# Train/val/test split (~71/14/14), stratified by class.
#
# lesion_split.csv - grouped by lesion_id, so all photos of a lesion end up on the same
#                    side. This is the split all results use.
# image_split.csv  - split by image (the common mistake). Only used for the leakage
#                    experiment.
#
# usage: python src/splits.py

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

META = Path("data/processed/metadata.csv")
OUT = Path("data/splits")
SEED = 42
CLASSES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
COLS = ["image_id", "lesion_id", "dx", "label", "split"]


def hold_out(df, k, by_lesion):
    # hold out 1 of k folds
    if by_lesion:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
        rest, held = next(cv.split(df, df.label, groups=df.lesion_id))
    else:
        cv = StratifiedShuffleSplit(n_splits=1, test_size=1 / k, random_state=SEED)
        rest, held = next(cv.split(df, df.label))
    return df.index[rest], df.index[held]


def make_split(meta, by_lesion):
    meta = meta.copy()
    rest, test = hold_out(meta, 7, by_lesion)               # 1/7 -> test
    train, val = hold_out(meta.loc[rest], 6, by_lesion)     # 1/6 of the rest -> val
    meta["split"] = "train"
    meta.loc[val, "split"] = "val"
    meta.loc[test, "split"] = "test"
    return meta


def shared_lesions(df, a, b):
    return set(df[df.split == a].lesion_id) & set(df[df.split == b].lesion_id)


def report(df, name):
    print(f"\n--- {name} ---")
    t = pd.crosstab(df.dx, df.split).reindex(index=CLASSES, columns=["train", "val", "test"])
    t.loc["TOTAL"] = t.sum()
    print(t)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        s = shared_lesions(df, a, b)
        n = df[(df.split == b) & df.lesion_id.isin(s)].shape[0]
        print(f"  lesions in both {a}/{b}: {len(s):4d}  ({n} {b} images)")


def main():
    meta = pd.read_csv(META)
    OUT.mkdir(parents=True, exist_ok=True)

    good = make_split(meta, by_lesion=True)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not shared_lesions(good, a, b), f"leakage between {a} and {b}"
    good[COLS].to_csv(OUT / "lesion_split.csv", index=False)
    report(good, "split by lesion")

    leaky = make_split(meta, by_lesion=False)
    leaky[COLS].to_csv(OUT / "image_split.csv", index=False)
    report(leaky, "split by image (leaky)")


if __name__ == "__main__":
    main()
