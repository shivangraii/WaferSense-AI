"""WaferSense AI - full training + evaluation pipeline.

Real data:   python train.py --data /path/to/wm811k-folder-or-LSWMD.pkl
Smoke test:  python train.py --synthetic --epochs 3
Outputs go to ./artifacts (model.pt, metrics.json, samples.npz)
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

from common import CLASSES, IMG, WaferCNN, load_wm811k, make_synthetic, onehot_batch, preprocess_map

p = argparse.ArgumentParser()
p.add_argument("--data", default=None, help="folder containing LSWMD.pkl, or the .pkl file itself")
p.add_argument("--synthetic", action="store_true", help="use fake data (code test only!)")
p.add_argument("--n_per_class", type=int, default=300, help="synthetic only")
p.add_argument("--max_per_class", type=int, default=4000, help="cap per class (controls subset size)")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--patience", type=int, default=12)
p.add_argument("--out", default="artifacts")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
rng = np.random.default_rng(args.seed)
torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- 1. Load ---------------------------------------------------------------
if args.synthetic:
    maps, labels, lots, summary = make_synthetic(args.n_per_class, args.seed)
    dataset_name = "SYNTHETIC (code test only - not WM-811K)"
else:
    if not args.data:
        raise SystemExit("Give --data <path> or use --synthetic")
    maps, labels, lots, summary = load_wm811k(args.data)
    dataset_name = "WM-811K"
print(summary)
full_counts = {str(k): int(v) for k, v in Counter(labels).items()}
print("Labelled class counts (full):", full_counts)

# ---- 2. Cap per class (documented subset), preprocess ------------------------
sel = []
for c in CLASSES:
    ids = np.where(labels == c)[0]
    if len(ids) > args.max_per_class:
        ids = rng.choice(ids, args.max_per_class, replace=False)
    sel.extend(ids.tolist())
sel = np.array(sorted(sel))
X = np.stack([preprocess_map(maps[i]) for i in sel])          # (N,64,64) uint8
y = np.array([CLASSES.index(labels[i]) for i in sel])
print("Subset used:", X.shape, dict(Counter(labels[sel])))

# ---- 3. Stratified split 70/15/15 -------------------------------------------
idx = np.arange(len(X))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)
va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=args.seed)
split_counts = {n: {CLASSES[k]: int((y[s] == k).sum()) for k in range(len(CLASSES))}
                for n, s in [("train", tr), ("val", va), ("test", te)]}
print({k: sum(v.values()) for k, v in split_counts.items()})

# ---- 4. Baseline: Random Forest on pooled features --------------------------
def rf_features(a):
    n = len(a)
    fail = (a == 2).reshape(n, 16, 4, 16, 4).mean((2, 4)).reshape(n, -1)
    pres = (a > 0).reshape(n, 16, 4, 16, 4).mean((2, 4)).reshape(n, -1)
    return np.hstack([fail, pres])

t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=args.seed)
rf.fit(rf_features(X[tr]), y[tr])
rf_pred = rf.predict(rf_features(X[te]))
baseline = {"accuracy": accuracy_score(y[te], rf_pred),
            "macro_f1": f1_score(y[te], rf_pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y[te], rf_pred)}
print(f"Baseline RF ({time.time()-t0:.0f}s):", baseline)

# ---- 5. CNN with class-weighted loss ----------------------------------------
counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(float)
cw = torch.tensor(len(tr) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
model = WaferCNN().to(device)
crit = nn.CrossEntropyLoss(weight=cw.to(device))
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps_per_ep = int(np.ceil(len(tr) / args.batch))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_ep)
Xt = torch.from_numpy(X)
yt = torch.from_numpy(y)


def augment(b):
    """Per-sample label-preserving augmentation: random 90-degree rotation + random flip."""
    out = b.clone()
    k = torch.randint(0, 4, (len(b),))
    for r in range(1, 4):
        m = k == r
        if m.any():
            out[m] = torch.rot90(b[m], r, dims=(1, 2))
    f = torch.rand(len(b)) < 0.5
    out[f] = torch.flip(out[f], dims=(2,))
    return out


@torch.no_grad()
def predict(ids, bs=512):
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        xb = onehot_batch(Xt[ids[i:i + bs]]).to(device)
        out.append(model(xb).argmax(1).cpu())
    return torch.cat(out).numpy()


history = {"train_loss": [], "val_macro_f1": [], "val_acc": []}
best_f1, best_state, bad = -1, None, 0
for ep in range(1, args.epochs + 1):
    model.train()
    perm = torch.from_numpy(rng.permutation(tr))
    tot = 0.0
    for i in range(0, len(perm), args.batch):
        b = perm[i:i + args.batch]
        xb = onehot_batch(augment(Xt[b])).to(device)
        yb = yt[b].to(device)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item() * len(b)
    vp = predict(va)
    vf1 = f1_score(y[va], vp, average="macro")
    history["train_loss"].append(tot / len(tr))
    history["val_macro_f1"].append(vf1)
    history["val_acc"].append(accuracy_score(y[va], vp))
    print(f"Epoch {ep:02d} | loss {tot/len(tr):.4f} | val macro-F1 {vf1:.4f} | val acc {history['val_acc'][-1]:.4f}")
    if vf1 > best_f1:
        best_f1, bad = vf1, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= args.patience:
            print("Early stopping.")
            break

# ---- 6. Final evaluation on untouched test set (once) ------------------------
model.load_state_dict(best_state)
tp = predict(te)
rep = classification_report(y[te], tp, labels=range(len(CLASSES)), target_names=CLASSES,
                            output_dict=True, zero_division=0)
cnn = {"accuracy": accuracy_score(y[te], tp), "macro_f1": f1_score(y[te], tp, average="macro"),
       "balanced_accuracy": balanced_accuracy_score(y[te], tp)}
print("CNN test:", cnn)

torch.save({"state_dict": best_state, "classes": CLASSES, "img": IMG}, os.path.join(args.out, "model.pt"))
metrics = {
    "dataset": dataset_name, "dataset_summary": summary, "full_class_counts": full_counts,
    "subset_size": int(len(X)), "max_per_class": args.max_per_class, "split_counts": split_counts,
    "classes": CLASSES, "cnn": cnn, "baseline_rf": baseline, "per_class": rep,
    "confusion_matrix": confusion_matrix(y[te], tp, labels=range(len(CLASSES))).tolist(),
    "history": history, "epochs_run": len(history["train_loss"]),
}
with open(os.path.join(args.out, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# a few test maps per class for the app demo
smp_x, smp_y = [], []
for k in range(len(CLASSES)):
    ids = te[y[te] == k][:8]
    smp_x.extend(X[ids]); smp_y.extend([k] * len(ids))
np.savez_compressed(os.path.join(args.out, "samples.npz"), x=np.stack(smp_x), y=np.array(smp_y))
print(f"Saved model.pt, metrics.json, samples.npz in ./{args.out}")
