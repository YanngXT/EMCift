#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?usage: bash scripts/run.sh DATASET DATA_ROOT [PHYSICAL_GPU] [SEED ...]}
data_root=${2:?data root required}
gpu=${3:-0}
if [ "$#" -ge 3 ]
then
  shift 3
else
  shift 2
fi
if [ "$#" -eq 0 ]
then
  set -- 42
fi
root=$(dirname "$(dirname "$(readlink -f "$0")")")
cd "$root"
case "$dataset" in
  iDigits|DomainNet|CORe50) config="configs/$dataset.json" ;;
  PACS|iWildCamTinyPlus) config="experiments/supplemental/$dataset.json" ;;
  *) printf 'Unsupported dataset: %s\n' "$dataset" >&2; exit 2 ;;
esac
for seed in "$@"
do
  CUDA_VISIBLE_DEVICES="$gpu" python -u main.py --config "$config" \
    --data-root "$data_root" --device 0 --seed "$seed" \
    --output-dir "outputs/$dataset/seed_$seed"
done
