import glob
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
IMG = 64

def preprocess_map(wm, size=IMG):
    """Wafer map (2D, values 0=background, 1=pass die, 2=fail die) -> uint8 (size,size).
    Nearest-neighbour resize so the categorical values never get blended."""
    a = np.clip(np.asarray(wm).astype(np.int64), 0, 2)
    h, w = a.shape
    ri = np.minimum(((np.arange(size) + 0.5) * h / size).astype(int), h - 1)
    ci = np.minimum(((np.arange(size) + 0.5) * w / size).astype(int), w - 1)
    return a[np.ix_(ri, ci)].astype(np.uint8)

def onehot_batch(idx):
    """uint8/long tensor (N,H,W) -> float tensor (N,3,H,W) one-hot (background, pass, fail)."""
    idx = torch.as_tensor(idx).long()
    return F.one_hot(idx, 3).permute(0, 3, 1, 2).float()


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())


class WaferCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.b1 = nn.Sequential(_block(3, 48), nn.MaxPool2d(2))     # 64 -> 32
        self.b2 = nn.Sequential(_block(48, 96), nn.MaxPool2d(2))    # 32 -> 16
        self.b3 = nn.Sequential(_block(96, 192), nn.MaxPool2d(2))   # 16 -> 8
        self.b4 = _block(192, 192)                                  # Grad-CAM target layer (8x8)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(192, n_classes))

    def forward(self, x):
        return self.head(self.b4(self.b3(self.b2(self.b1(x)))))


# ----------------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------------
def gradcam(model, x, class_idx=None):
    """x: (1,3,H,W). Returns (cam HxW in [0,1], probs ndarray, class_idx used)."""
    model.eval()
    store = {}

    def fwd_hook(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    handle = model.b4.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            probs = F.softmax(logits, 1)
            if class_idx is None:
                class_idx = int(probs.argmax(1))
            model.zero_grad()
            logits[0, class_idx].backward()
    finally:
        handle.remove()
    w = store["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam, probs[0].detach().numpy(), class_idx


# ----------------------------------------------------------------------------
# Real dataset loader (WM-811K -> LSWMD.pkl)
# ----------------------------------------------------------------------------
def _read_pickle_compat(path):
    """LSWMD.pkl was written with an old pandas; this handles the 'pandas.indexes' error."""
    import pandas as pd
    try:
        return pd.read_pickle(path)
    except Exception as e:  # noqa
        print(f"pd.read_pickle failed ({e}); retrying with compatibility unpickler...")

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("pandas.indexes"):
                module = module.replace("pandas.indexes", "pandas.core.indexes")
            try:
                return super().find_class(module, name)
            except (AttributeError, ImportError):
                if name in ("Int64Index", "Float64Index", "UInt64Index"):
                    return pd.Index
                raise

    with open(path, "rb") as f:
        return U(f, encoding="latin1").load()


def find_pickle(path):
    if os.path.isfile(path):
        return path
    hits = glob.glob(os.path.join(path, "**", "LSWMD*.pkl"), recursive=True) or \
        glob.glob(os.path.join(path, "**", "*.pkl"), recursive=True)
    if not hits:
        raise FileNotFoundError(f"LSWMD.pkl not found under {path}")
    return hits[0]


def load_wm811k(path):
    """Returns (list_of_maps, labels(str array), lot_names, summary dict)."""
    df = _read_pickle_compat(find_pickle(path))
    total = len(df)

    def lab(x):
        try:
            return str(np.asarray(x).ravel()[0])
        except Exception:
            return None

    labels = df["failureType"].apply(lab)
    keep = labels.isin(CLASSES)
    summary = {"total_records": int(total), "labelled_records": int(keep.sum()),
               "unlabelled_records": int(total - keep.sum())}
    df = df[keep]
    maps = list(df["waferMap"].values)
    lots = df["lotName"].apply(lambda x: str(np.asarray(x).ravel()[0]) if np.size(x) else "").values \
        if "lotName" in df.columns else np.array([""] * len(df))
    return maps, labels[keep].values, lots, summary


# ----------------------------------------------------------------------------
# Synthetic wafer maps (ONLY for smoke-testing the code without the real dataset)
# ----------------------------------------------------------------------------
def synth_wafer(cls, rng):
    n = int(rng.integers(30, 52))
    yy, xx = np.mgrid[:n, :n]
    c = (n - 1) / 2
    r = np.hypot(yy - c, xx - c) / (n / 2)
    inside = r <= 1.0
    noise = rng.random((n, n))
    fail = noise < 0.02
    if cls == "Center":
        fail |= (r < 0.3) & (noise < 0.85)
    elif cls == "Donut":
        fail |= (r > 0.35) & (r < 0.65) & (noise < 0.85)
    elif cls == "Edge-Ring":
        fail |= (r > 0.85) & (noise < 0.85)
    elif cls == "Edge-Loc":
        th = np.arctan2(yy - c, xx - c)
        a0 = rng.uniform(-np.pi, np.pi)
        d = np.abs(np.angle(np.exp(1j * (th - a0))))
        fail |= (r > 0.75) & (d < 0.5) & (noise < 0.85)
    elif cls == "Loc":
        ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.5)
        cy, cx = c + rad * (n / 2) * np.sin(ang), c + rad * (n / 2) * np.cos(ang)
        fail |= (np.hypot(yy - cy, xx - cx) < n * 0.12) & (noise < 0.85)
    elif cls == "Random":
        fail |= noise < 0.15
    elif cls == "Near-full":
        fail |= noise < 0.85
    elif cls == "Scratch":
        p0 = rng.uniform(0.2 * n, 0.8 * n, 2)
        ang = rng.uniform(0, 2 * np.pi)
        ln = rng.uniform(0.4, 0.8) * n
        p1 = p0 + ln * np.array([np.sin(ang), np.cos(ang)])
        d = p1 - p0
        t = np.clip(((yy - p0[0]) * d[0] + (xx - p0[1]) * d[1]) / (d @ d), 0, 1)
        dist = np.hypot(yy - (p0[0] + t * d[0]), xx - (p0[1] + t * d[1]))
        fail |= dist < 1.0
    return np.where(inside, np.where(fail, 2, 1), 0).astype(np.uint8)


def make_synthetic(n_per_class=300, seed=0):
    rng = np.random.default_rng(seed)
    maps, labels = [], []
    for cls in CLASSES:
        for _ in range(n_per_class):
            maps.append(synth_wafer(cls, rng))
            labels.append(cls)
    return maps, np.array(labels), np.array([""] * len(maps)), \
        {"total_records": len(maps), "labelled_records": len(maps), "unlabelled_records": 0}
