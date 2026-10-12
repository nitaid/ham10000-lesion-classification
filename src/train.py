# Train one model, keep the epoch with the best validation balanced accuracy,
# then evaluate it once on the test set.
#
# examples:
#   python src/train.py --overfit 20 --model cnn       # sanity check, loss should go to ~0
#   python src/train.py --model cnn --name baseline_cnn
#   python src/train.py --model resnet18 --seed 0 --name resnet18_s0
#   python src/train.py --model resnet18 --split image --name resnet18_LEAKY_s0
#
# each run saves to runs/<name>/: metrics.json, test_preds.csv, best.pt

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from common import (CLASSES, N_CLASSES, build_model, compute_metrics, count_params, get_device,
                    load_images, load_split, make_batch, predict, seed_everything)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    p.add_argument("--no-pretrained", action="store_true", help="resnet18 from random weights")
    p.add_argument("--split", choices=["lesion", "image"], default="lesion")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=None, help="default 1e-3 for cnn, 3e-4 for resnet18")
    p.add_argument("--wd", type=float, default=1e-4, help="weight decay, 0 = off")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--no-early-stop", action="store_true", help="keep the last epoch instead of the best")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit", type=int, default=0, help="sanity check on N training images")
    p.add_argument("--name", default=None)
    a = p.parse_args()
    if a.lr is None:
        a.lr = 1e-3 if a.model == "cnn" else 3e-4
    if a.name is None:
        a.name = f"{a.model}_{a.split}_s{a.seed}"
    return a


def make_optimizer(model, lr, wd):
    # weight decay only on weights, not on biases / batchnorm (like in lecture 7)
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def overfit_check(args, X, data, device):
    # if everything is wired correctly the model should be able to memorize a few images
    rows, labels, _ = data["train"]
    pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(args.seed))[:args.overfit]
    rows, y = rows[pick], labels[pick].to(device)
    model = build_model(args.model, dropout=0.0, pretrained=not args.no_pretrained).to(device)
    opt = make_optimizer(model, args.lr, 0.0)
    model.train()
    x = make_batch(X, rows, device, train_aug=False)
    for step in range(1, 201):
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 0:
            acc = (model(x).argmax(1) == y).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.4f}  train acc {acc:.2f}")
    print("\nPASS" if loss.item() < 0.05 else "\nFAIL - loss didn't go to ~0, something is wrong")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    print(f"device: {device}")

    id2row, X = load_images()
    data = load_split(id2row, args.split)
    if args.overfit:
        return overfit_check(args, X, data, device)

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    tr_rows, tr_y, _ = data["train"]
    va_rows, va_y, _ = data["val"]
    te_rows, te_y, te_ids = data["test"]

    # class weights N / (K * n_c) so rare classes count as much as nv in the loss
    counts = torch.bincount(tr_y, minlength=N_CLASSES).float()
    cw = None if args.no_class_weights else (counts.sum() / (N_CLASSES * counts)).to(device)

    model = build_model(args.model, args.dropout, pretrained=not args.no_pretrained).to(device)
    opt = make_optimizer(model, args.lr, args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"{args.model}: {count_params(model) / 1e6:.2f}M params, "
          f"train/val/test {len(tr_rows)}/{len(va_rows)}/{len(te_rows)}")

    history, best, best_epoch, bad = [], -1.0, -1, 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep = time.time()
        perm = torch.randperm(len(tr_rows))
        total, n = 0.0, 0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            x = make_batch(X, tr_rows[idx], device, train_aug=not args.no_aug)
            y = tr_y[idx].to(device)
            loss = F.cross_entropy(model(x), y, weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            n += len(idx)
        sched.step()

        vm = compute_metrics(va_y.numpy(), predict(model, X, va_rows, device))
        history.append({"epoch": epoch, "train_loss": total / n,
                        **{k: vm[k] for k in ["acc", "bal_acc", "macro_f1", "macro_auc", "mel_recall"]}})
        improved = vm["bal_acc"] > best
        if improved:
            best, best_epoch, bad = vm["bal_acc"], epoch, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        print(f"ep {epoch:2d}  loss {total / n:.3f}  val acc {vm['acc']:.3f}  bal_acc {vm['bal_acc']:.3f}  "
              f"mel_rec {vm['mel_recall']:.3f}  {time.time() - t_ep:.0f}s{'  *' if improved else ''}")
        if not args.no_early_stop and bad >= args.patience:
            print(f"early stop (no improvement for {args.patience} epochs)")
            break

    if args.no_early_stop:
        torch.save(model.state_dict(), out / "best.pt")
        best_epoch = epoch
    else:
        model.load_state_dict(torch.load(out / "best.pt"))

    # test set is used only here, once
    val_m = compute_metrics(va_y.numpy(), predict(model, X, va_rows, device))
    te_prob = predict(model, X, te_rows, device)
    test_m = compute_metrics(te_y.numpy(), te_prob)

    preds = pd.DataFrame(te_prob, columns=[f"p_{c}" for c in CLASSES])
    preds.insert(0, "image_id", te_ids)
    preds.insert(1, "label", te_y.numpy())
    preds.insert(2, "pred", te_prob.argmax(1))
    preds.to_csv(out / "test_preds.csv", index=False)

    result = {"config": vars(args), "params": count_params(model), "best_epoch": best_epoch,
              "minutes": (time.time() - t0) / 60, "history": history, "val": val_m, "test": test_m}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== {args.name}  (epoch {best_epoch}, {result['minutes']:.1f} min) ===")
    print(f"TEST  acc {test_m['acc']:.3f}  bal_acc {test_m['bal_acc']:.3f}  macro_f1 {test_m['macro_f1']:.3f}  "
          f"auc {test_m['macro_auc']:.3f}  mel_recall {test_m['mel_recall']:.3f}")
    print("per-class recall: " + "  ".join(f"{c} {r:.2f}" for c, r in test_m["per_class_recall"].items()))


if __name__ == "__main__":
    main()
