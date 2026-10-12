# Quick timing of one training step (data copy / augmentation / forward+backward).
# Used this to check why epochs were slow - turned out it's just compute, not the data loading.
#
# usage: python src/bench.py

import time

import torch
import torch.nn.functional as F

from common import build_model, get_device, load_images, load_split, make_batch

dev = get_device()
if dev.type == "mps":
    sync = torch.mps.synchronize
elif dev.type == "cuda":
    sync = torch.cuda.synchronize
else:
    sync = lambda: None

id2row, X = load_images()
rows = load_split(id2row)["train"][0]
y = torch.zeros(64, dtype=torch.long, device=dev)

for name in ["cnn", "resnet18"]:
    m = build_model(name, 0.3).to(dev)
    opt = torch.optim.AdamW(m.parameters())
    t = dict(data=0.0, aug=0.0, step=0.0)
    for i in range(25):
        idx = rows[torch.randint(0, len(rows), (64,))]
        a = time.time(); x = X[idx].to(dev).float().div_(255); sync()
        b = time.time(); x = make_batch(X, idx, dev, True); sync()
        c = time.time(); loss = F.cross_entropy(m(x), y); opt.zero_grad(); loss.backward(); opt.step(); sync()
        d = time.time()
        if i >= 5:  # skip warmup
            t["data"] += b - a; t["aug"] += c - b; t["step"] += d - c
    per = {k: v / 20 * 1000 for k, v in t.items()}
    print(f"{name:9s} ms/batch: " + "  ".join(f"{k} {v:.0f}" for k, v in per.items())
          + f"  -> ~{sum(per.values()) * 112 / 1000:.0f}s per epoch")
