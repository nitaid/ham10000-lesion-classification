# Shared code: data loading, augmentation, models and metrics.

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from tqdm import tqdm

CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
N_CLASSES = len(CLASSES)
PROC = Path("data/processed")
SPLITS = Path("data/splits")
CACHE = PROC / "images_uint8.pt"
CROP = 224
# ImageNet stats, used for both models so they get the exact same input
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---- data ----

def load_images():
    # whole dataset as one uint8 tensor [N, 3, 256, 341] (~2.6GB), cached after the first run
    if CACHE.exists():
        d = torch.load(CACHE)
        return {i: r for r, i in enumerate(d["ids"])}, d["x"]
    ids = list(pd.read_csv(PROC / "metadata.csv").image_id)
    imgs = [np.asarray(Image.open(PROC / "images" / f"{i}.jpg").convert("RGB"))
            for i in tqdm(ids, desc="loading images (first run only)")]
    assert len({a.shape for a in imgs}) == 1
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()
    torch.save({"ids": ids, "x": x}, CACHE)
    return {i: r for r, i in enumerate(ids)}, x


def load_split(id2row, which="lesion"):
    # returns {"train"/"val"/"test": (row indices, labels, image ids)}
    df = pd.read_csv(SPLITS / f"{which}_split.csv")
    out = {}
    for s in ["train", "val", "test"]:
        part = df[df.split == s]
        rows = torch.tensor([id2row[i] for i in part.image_id], dtype=torch.long)
        out[s] = (rows, torch.tensor(part.label.values, dtype=torch.long), list(part.image_id))
    return out


# ---- augmentation (runs on the GPU, per batch) ----

def center_crop(x):
    _, _, h, w = x.shape
    t, l = (h - CROP) // 2, (w - CROP) // 2
    return x[:, :, t:t + CROP, l:l + CROP]


def augment(x):
    # random 224 crop, flips + transpose (= all 8 rotations/reflections, a lesion has no "up"),
    # +-20% brightness and contrast
    b, _, h, w = x.shape
    tops = torch.randint(0, h - CROP + 1, (b,)).tolist()
    lefts = torch.randint(0, w - CROP + 1, (b,)).tolist()
    x = torch.stack([x[i, :, t:t + CROP, l:l + CROP] for i, (t, l) in enumerate(zip(tops, lefts))])

    def maybe(op, x):
        mask = (torch.rand(b, device=x.device) < 0.5).view(b, 1, 1, 1)
        return torch.where(mask, op(x), x)

    x = maybe(lambda z: z.flip(3), x)
    x = maybe(lambda z: z.flip(2), x)
    x = maybe(lambda z: z.transpose(2, 3), x)

    x = x * (1 + 0.2 * (2 * torch.rand(b, 1, 1, 1, device=x.device) - 1))
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * (1 + 0.2 * (2 * torch.rand(b, 1, 1, 1, device=x.device) - 1)) + m
    return x.clamp(0, 1)


def make_batch(X, rows, device, train_aug):
    x = X[rows].to(device, non_blocking=True).float().div_(255)
    x = augment(x) if train_aug else center_crop(x)
    return (x - MEAN.to(device)) / STD.to(device)


# ---- models ----

class SmallCNN(nn.Module):
    # baseline: 4 x [conv-BN-ReLU, conv-BN-ReLU, maxpool], 32->64->128->256 channels,
    # global avg pool, linear. ~1.2M params, receptive field 76px after the last pool.
    def __init__(self, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, dropout, pretrained=True):
    if name == "cnn":
        return SmallCNN(dropout=dropout)
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, N_CLASSES))  # 1000 -> 7 classes
        return m
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ---- metrics ----

def compute_metrics(y, prob):
    pred = prob.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(N_CLASSES))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    try:
        auc = roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=range(N_CLASSES))
    except ValueError:
        auc = float("nan")
    return {
        "acc": float((pred == y).mean()),
        "bal_acc": float(recall.mean()),  # main metric: every class counts the same
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(N_CLASSES), zero_division=0)),
        "macro_auc": float(auc),
        "mel_recall": float(recall[CLASSES.index("mel")]),
        "per_class_recall": {c: float(r) for c, r in zip(CLASSES, recall)},
        "confusion": cm.tolist(),
    }


@torch.no_grad()
def predict(model, X, rows, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        x = make_batch(X, rows[i:i + bs], device, train_aug=False)
        out.append(torch.softmax(model(x), 1).float().cpu())
    return torch.cat(out).numpy()
