"""Export the trained WaferSense model, metrics and demo wafers into ONE json that the website runs in the browser.

    python export_web.py                 -> writes artifacts/webdata.json
    python export_web.py --static        -> also writes index.html (a standalone copy of the whole site)

BatchNorm is folded into the convolutions, so the browser only needs conv + ReLU + pooling + one linear layer.
Grad-CAM for a global-average-pooling + linear head is exactly  ReLU(sum_k w_ck * A_k), so the site computes
the same heatmap PyTorch's Grad-CAM does (checked numerically).
"""
import base64
import json
import os
import sys
import numpy as np


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def build_webdata(art="artifacts"):
    import torch
    import torch.nn as nn
    from common import WaferCNN

    ck = torch.load(os.path.join(art, "model.pt"), map_location="cpu")
    classes = list(ck["classes"])
    m = WaferCNN(len(classes))
    m.load_state_dict(ck["state_dict"])
    m.eval()

    chunks, layers, off = [], [], 0
    for name in ("b1", "b2", "b3", "b4"):
        blk = getattr(m, name)
        pool = isinstance(blk[0], nn.Sequential)
        seq = blk[0] if pool else blk
        conv, bn = seq[0], seq[1]
        with torch.no_grad():
            s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = (conv.weight * s[:, None, None, None]).numpy().astype("<f4")
            b = ((conv.bias - bn.running_mean) * s + bn.bias).numpy().astype("<f4")
        layers.append({"cin": conv.in_channels, "cout": conv.out_channels, "pool": pool, "w": off, "b": off + w.size})
        chunks += [w.ravel(), b.ravel()]
        off += w.size + b.size
    fc = m.head[-1]
    fw = fc.weight.detach().numpy().astype("<f4")
    fb = fc.bias.detach().numpy().astype("<f4")
    fcm = {"n": fw.shape[0], "k": fw.shape[1], "w": off, "b": off + fw.size}
    chunks += [fw.ravel(), fb.ravel()]
    buf = np.concatenate(chunks).astype("<f4")

    sm = np.load(os.path.join(art, "samples.npz"))
    sx = sm["x"].astype(np.uint8)
    with open(os.path.join(art, "metrics.json")) as f:
        metrics = json.load(f)
    return {"classes": classes, "layers": layers, "fc": fcm, "weights": _b64(buf),
            "samples": {"n": int(len(sx)), "x": _b64(sx), "y": [int(v) for v in sm["y"]]}, "metrics": metrics}


def inject(template_text, data_json):
    token = "/*__WS_DATA__*/null"
    assert template_text.count(token) == 1, "template placeholder missing"
    return template_text.replace(token, data_json)


if __name__ == "__main__":
    art = os.environ.get("WAFER_ARTIFACTS", "artifacts")
    data = json.dumps(build_webdata(art), separators=(",", ":"))
    with open(os.path.join(art, "webdata.json"), "w") as f:
        f.write(data)
    print(f"wrote {art}/webdata.json ({len(data) // 1024} KB)")
    if "--static" in sys.argv:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "site_template.html"), encoding="utf-8") as f:
            html = inject(f.read(), data)
        with open("index.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("wrote index.html")
