# Confusion matrix, most common mistakes, and Grad-CAM examples for one trained run.
# Grad-CAM uses the last conv block of ResNet-18 (layer4).
# Pick the run by validation score, not test (resnet18_s1 had the best val).
#
# usage: python src/interpret.py --run resnet18_s1
# saves results/confusion_matrix.png, results/gradcam_grid.png, results/errors.csv

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import confusion_matrix

from common import CLASSES, MEAN, STD, build_model, center_crop, load_images

OUT = Path("results")
# (true, predicted) pairs to show: 3 correct, 3 typical mistakes
PANELS = [("mel", "mel"), ("nv", "nv"), ("bcc", "bcc"), ("mel", "nv"), ("bkl", "mel"), ("nv", "mel")]


def load_model(run):
    model = build_model("resnet18", dropout=0.0, pretrained=False)
    model.load_state_dict(torch.load(Path("runs") / run / "best.pt", map_location="cpu"))
    return model.eval()


def plot_confusion(preds, path):
    cm = confusion_matrix(preds.label, preds.pred, labels=range(len(CLASSES)))
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix (row = recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="resnet18_s1")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    preds = pd.read_csv(Path("runs") / args.run / "test_preds.csv")
    preds["true_cls"] = preds.label.map(lambda i: CLASSES[i])
    preds["pred_cls"] = preds.pred.map(lambda i: CLASSES[i])
    preds["p_pred"] = preds.apply(lambda r: r[f"p_{r.pred_cls}"], axis=1)

    cm = plot_confusion(preds, OUT / "confusion_matrix.png")
    off = [(cm[i, j], CLASSES[i], CLASSES[j])
           for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j]
    print("most common mistakes (true -> predicted):")
    for n, t, p in sorted(off, reverse=True)[:6]:
        print(f"  {t:6s} -> {p:6s} {n:4d}  ({n / cm[CLASSES.index(t)].sum():.0%} of {t})")

    wrong = preds[preds.label != preds.pred].sort_values("p_pred", ascending=False)
    wrong[["image_id", "true_cls", "pred_cls", "p_pred"]].head(30).to_csv(OUT / "errors.csv", index=False)

    id2row, X = load_images()
    model = load_model(args.run)
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    # for each panel take the most confident example of that kind
    picks = []
    for t, p in PANELS:
        cand = preds[(preds.true_cls == t) & (preds.pred_cls == p)].sort_values("p_pred", ascending=False)
        if len(cand):
            picks.append(cand.iloc[0])

    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6), squeeze=False)
    for k, r in enumerate(picks):
        img = center_crop(X[[id2row[r.image_id]]].float() / 255)
        heat = cam(input_tensor=(img - MEAN) / STD, targets=[ClassifierOutputTarget(int(r.pred))])[0]
        rgb = img[0].permute(1, 2, 0).numpy()
        axes[0, k].imshow(rgb)
        axes[1, k].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
        ok = r.true_cls == r.pred_cls
        title = f"correct: {r.true_cls}" if ok else f"WRONG: {r.true_cls} → {r.pred_cls}"
        axes[0, k].set_title(f"{title}\np = {r.p_pred:.2f}", fontsize=9, color="black" if ok else "#B5482E")
        for a in axes[:, k]:
            a.axis("off")
    fig.suptitle(f"Grad-CAM for the predicted class · {args.run}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gradcam_grid.png", dpi=200)
    print(f"\nsaved figures to {OUT}/")


if __name__ == "__main__":
    main()
