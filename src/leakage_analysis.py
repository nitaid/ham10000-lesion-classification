# Is the higher score on the image-level split just memorization?
# Inside the leaky test set, split images into "seen" (the lesion also has photos in the
# leaky train set) and "unseen", and score each group separately. Same model, same test
# images, so the comparison is fair.
#
# usage: python src/leakage_analysis.py

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common import CLASSES

OUT = Path("results")


def score(df):
    mel = df[df.label == CLASSES.index("mel")]
    return {"n_images": len(df),
            "acc": (df.label == df.pred).mean(),
            "bal_acc": balanced_accuracy_score(df.label, df.pred),
            "mel_recall": (mel.pred == mel.label).mean() if len(mel) else np.nan}


def main():
    split = pd.read_csv("data/splits/image_split.csv")
    train_lesions = set(split[split.split == "train"].lesion_id)
    lesion_of = dict(zip(split.image_id, split.lesion_id))

    rows = []
    for run in sorted(Path("runs").glob("resnet18_LEAKY_s*")):
        p = pd.read_csv(run / "test_preds.csv")
        p["seen"] = p.image_id.map(lesion_of).isin(train_lesions)
        for subset, part in [("all", p), ("seen", p[p.seen]), ("unseen", p[~p.seen])]:
            rows.append({"run": run.name, "subset": subset, **score(part)})
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "leakage_seen_unseen.csv", index=False)

    print(df.round(3).to_string(index=False))
    print("\nmean over seeds:")
    for subset, g in df.groupby("subset", sort=False):
        print(f"{subset:7s} n={int(g.n_images.mean()):4d}  bal_acc {g.bal_acc.mean():.3f}+-{g.bal_acc.std():.3f}  "
              f"acc {g.acc.mean():.3f}  mel_recall {g.mel_recall.mean():.3f}")


if __name__ == "__main__":
    main()
