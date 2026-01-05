#!/bin/bash

#SBATCH -J svgvae_interp
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o .../vecssl_svgvae_interp_%j.txt

cd ...

source .venv/bin/activate

# 1. Set path variables
DATA_ROOT="..."
SVG_DIR="${DATA_ROOT}/svg"
IMG_DIR="${DATA_ROOT}/img"
META_CSV="${DATA_ROOT}/metadata.csv"

# Checkpoint points to the version with the Decoder
CKPT_PATH="..."

# Rename output directory for distinction
OUT_DIR="interp_output_b_to_0_16"

echo "Running svgvae Decoder interpolation..."
echo "Checkpoint: $CKPT_PATH"

# 2. Run Python script
# ⚠️ Key modification: changed --model-type to ae ⚠️
python scripts/eval_interpolation_universal.py \
    --ckpt "$CKPT_PATH" \
    --model-type ae \
    --svg-dir "$SVG_DIR" \
    --img-dir "$IMG_DIR" \
    --meta "$META_CSV" \
    --idx-a 3 \
    --idx-b 0 \
    --num-steps 16 \
    --out-dir "$OUT_DIR" \
    --device cuda \
    --log-level INFO \
    --max-num-groups 8 \
    --max-seq-len 40

echo "Done. Results saved to $OUT_DIR"
