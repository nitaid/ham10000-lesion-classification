# Resize every HAM10000 image once (short side -> 256 px) so training doesn't have to
# decode the full 600x450 JPEGs every epoch. Also adds an integer label column.
#
# usage: python src/prepare_data.py

from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW = Path("data/raw")
OUT = Path("data/processed")
SHORT_SIDE = 256
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]


def find_images(root):
    # the Kaggle zip has every image folder twice (upper/lower case), keep one path per id
    paths = {}
    for p in sorted(root.rglob("*.jpg")):
        paths.setdefault(p.stem, p)
    return paths


def resize(img, short=SHORT_SIDE):
    w, h = img.size
    s = short / min(w, h)
    return img.resize((round(w * s), round(h * s)), Image.BICUBIC)


def main():
    meta = pd.read_csv(RAW / "HAM10000_metadata.csv")
    paths = find_images(RAW)
    missing = set(meta.image_id) - set(paths)
    assert not missing, f"{len(missing)} images missing, e.g. {list(missing)[:3]}"

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(meta.image_id, desc="resizing"):
        dst = OUT / "images" / f"{image_id}.jpg"
        if dst.exists():
            continue
        with Image.open(paths[image_id]) as img:
            resize(img.convert("RGB")).save(dst, quality=95)

    meta["label"] = meta.dx.map({c: i for i, c in enumerate(CLASSES)})
    assert meta.label.notna().all()
    meta.to_csv(OUT / "metadata.csv", index=False)

    # should print 10015 images / 7470 lesions (same as the paper)
    counts = meta.dx.value_counts().reindex(CLASSES)
    print(f"\nimages:  {len(meta)}")
    print(f"lesions: {meta.lesion_id.nunique()}")
    print(f"lesions with 2+ images: {(meta.groupby('lesion_id').size() > 1).mean():.1%}")
    for c, n in counts.items():
        print(f"  {c:6s} {n:5d}  ({n / len(meta):.1%})")
    print(f"imbalance: {counts.max() / counts.min():.0f}:1")


if __name__ == "__main__":
    main()
