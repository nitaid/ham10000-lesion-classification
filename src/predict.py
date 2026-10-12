# Demo: classify a single image and show the Grad-CAM heat map.
# (research demo, not a medical tool)
#
# usage: python src/predict.py path/to/image.jpg [--run resnet18_s1]

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from common import CLASSES, CROP, MEAN, STD, build_model

NAMES = {"nv": "melanocytic nevus", "mel": "melanoma", "bkl": "benign keratosis",
         "bcc": "basal cell carcinoma", "akiec": "actinic keratosis", "vasc": "vascular lesion",
         "df": "dermatofibroma"}


def preprocess(path):
    # same as evaluation: short side 256, center crop 224, ImageNet normalization
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    l, t = (w - CROP) // 2, (h - CROP) // 2
    rgb = np.asarray(img.crop((l, t, l + CROP, t + CROP)), dtype=np.float32) / 255
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return rgb, (x - MEAN) / STD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--run", default="resnet18_s1")
    args = ap.parse_args()

    model = build_model("resnet18", dropout=0.0, pretrained=False)
    model.load_state_dict(torch.load(Path("runs") / args.run / "best.pt", map_location="cpu"))
    model.eval()

    rgb, x = preprocess(args.image)
    with torch.no_grad():
        prob = torch.softmax(model(x), 1)[0].numpy()
    order = prob.argsort()[::-1]
    print(f"\n{Path(args.image).name}")
    for i in order[:3]:
        print(f"  {CLASSES[i]:6s} {prob[i]:6.1%}  {NAMES[CLASSES[i]]}")

    top = int(order[0])
    heat = GradCAM(model=model, target_layers=[model.layer4[-1]])(
        input_tensor=x, targets=[ClassifierOutputTarget(top)])[0]
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.8))
    ax[0].imshow(rgb)
    ax[0].set_title("input")
    ax[1].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
    ax[1].set_title(f"{CLASSES[top]} · {prob[top]:.0%}")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    out = Path(args.image).with_name(Path(args.image).stem + "_gradcam.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
