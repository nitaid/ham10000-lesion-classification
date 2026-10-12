#!/bin/bash
# Runs all the experiments (23 runs, about 6 hours on my M4 MacBook).
# Finished runs are skipped, so it's safe to restart if something stops.
#
# run from the repo root:
#   caffeinate -i ./scripts/run_all.sh 2>&1 | tee logs/run_all.txt

cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p logs

run() {
  name=$1; shift
  if [ -f runs/$name/metrics.json ]; then echo "skip $name"; return; fi
  echo "=== $(date +%H:%M) $name"
  python src/train.py --name $name "$@" > logs/$name.log 2>&1 \
    && tail -3 logs/$name.log | head -2 \
    || echo "!!! $name failed, see logs/$name.log"
}

run baseline_cnn --model cnn --seed 0

for s in 0 1 2; do
  run resnet18_s$s         --model resnet18 --seed $s
  run resnet18_LEAKY_s$s   --model resnet18 --seed $s --split image
  [ $s = 0 ] && run resnet18_scratch_s0 --model resnet18 --no-pretrained --seed 0

  # ablation: remove one thing at a time
  run abl_no_aug_s$s       --model resnet18 --seed $s --no-aug
  run abl_no_dropout_s$s   --model resnet18 --seed $s --dropout 0
  run abl_no_wd_s$s        --model resnet18 --seed $s --wd 0
  run abl_no_earlystop_s$s --model resnet18 --seed $s --no-early-stop
  run abl_no_classw_s$s    --model resnet18 --seed $s --no-class-weights
done

echo "=== $(date +%H:%M) done"
