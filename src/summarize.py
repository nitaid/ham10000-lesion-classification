# Collect all runs in runs/ into one table: mean +- std over seeds per experiment.
# Runs are named <experiment>_s<seed>. Writes results/runs.csv and results/summary.csv.
#
# usage: python src/summarize.py

import json
import re
from pathlib import Path

import pandas as pd

OUT = Path("results")


def main():
    rows = []
    for f in sorted(Path("runs").glob("*/metrics.json")):
        m = json.loads(f.read_text())
        name = f.parent.name
        rows.append({
            "run": name, "experiment": re.sub(r"_s\d+$", "", name), "seed": m["config"]["seed"],
            "best_epoch": m["best_epoch"], "epochs_run": len(m["history"]), "minutes": round(m["minutes"], 1),
            "val_bal_acc": m["val"]["bal_acc"], "val_mel_recall": m["val"]["mel_recall"],
            "test_acc": m["test"]["acc"], "test_bal_acc": m["test"]["bal_acc"],
            "test_macro_f1": m["test"]["macro_f1"], "test_auc": m["test"]["macro_auc"],
            "test_mel_recall": m["test"]["mel_recall"],
            **{f"rec_{c}": r for c, r in m["test"]["per_class_recall"].items()},
        })
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "runs.csv", index=False)

    pd.set_option("display.width", 200)
    print(df[["run", "epochs_run", "best_epoch", "minutes", "val_bal_acc", "test_bal_acc",
              "test_mel_recall"]].round(3).to_string(index=False))

    metrics = ["val_bal_acc", "test_bal_acc", "test_acc", "test_macro_f1", "test_auc", "test_mel_recall"]
    g = df.groupby("experiment")
    summary = g[metrics].agg(["mean", "std"]).round(3)
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_seeds"] = g.size()
    summary.to_csv(OUT / "summary.csv")

    print("\nmean +- std over seeds:")
    for exp, r in summary.iterrows():
        cells = "  ".join(f"{m.replace('test_', '')} {r[m + '_mean']:.3f}+-{0 if pd.isna(r[m + '_std']) else r[m + '_std']:.3f}"
                          for m in ["val_bal_acc", "test_bal_acc", "test_mel_recall"])
        print(f"{exp:22s} n={int(r['n_seeds'])}  {cells}")


if __name__ == "__main__":
    main()
