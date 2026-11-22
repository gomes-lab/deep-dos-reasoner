#!/bin/bash
# Train Stage II (CPPR) on the frozen Stage I model with the same benchmark and seed.
#
#   bash scripts/train_stage2.sh <edos|edos_stable|phdos> <seed> [train_stage2.py options]
#   sbatch [-p ... --qos ...] scripts/train_stage2.sh ...      (from the repository root)
#
# Needs TRAINED/stage1/<benchmark>_stage1_seed<seed>/model.ckpt.
# Writes TRAINED/stage2/<benchmark>_stage2_seed<seed>/model.ckpt and log.txt.
# --cfl_courant 0.5 caps the Courant number at 1/2 so every transport step stays
# non-negative (DOS_Reasoner/stage2.py).
#SBATCH -J ddr_stage2
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -N 1 -n 1 -c 16
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -G 1

set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <benchmark> <seed> [extra args]" >&2
    exit 1
fi

source "$(dirname "$0")/common.sh" 2>/dev/null || source "${SLURM_SUBMIT_DIR:?}/scripts/common.sh"
cd_repo_root "$0"

BENCH=$1
SEED=$2
shift 2
set_benchmark "$BENCH"

python train_stage2.py \
    --data_src "$DATA_SRC" \
    --label_dim "$LABEL_DIM" \
    --scaling_law 1.0 \
    --train \
    --lr 1. \
    --num-epochs 20 \
    --batch-size 128 \
    --d_model 512 \
    --graph_dropout 0.2 \
    --dec_dropout 0.0 \
    --dec_layers 6 \
    --h 8 \
    --accum_step 1 \
    --weight_decay 1e-3 \
    --CPPR_lr 1e-3 \
    --k_steps 5 \
    --cfl_courant 0.5 \
    --seed "$SEED" \
    --stage1_ckpt "${BENCH}_stage1_seed${SEED}" \
    --save_dir "${BENCH}_stage2_seed${SEED}" \
    "$@"
