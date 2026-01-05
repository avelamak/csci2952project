#!/bin/bash

#SBATCH -J multimae_interp          # Job name
#SBATCH -p gpu                      # SLURM partition name (keep as 'gpu' if that is your cluster's partition)
#SBATCH --gres=gpu:1                # Request 1 GPU resource
#SBATCH --mem=64G                   # Request 64GB memory
#SBATCH -t 01:00:00                 # Run time limit set to 1 hour
#SBATCH -o .../vecssl_multimae_interp_eval_%j.txt  # Standard output file path

cd ...

# 1. Load environment
source .venv/bin/activate

# 2. Set path variables (Update based on your ls results)
# Dataset root directory
DATA_ROOT="..."

# Specific subdirectories and files
SVG_DIR="${DATA_ROOT}/svg"
IMG_DIR="${DATA_ROOT}/img"
META_CSV="${DATA_ROOT}/metadata.csv"

# Model Checkpoint path
CKPT_PATH="..."

# Output results directory
OUT_DIR="results/interpolation_test"

echo "Running interpolation..."
echo "Checkpoint: $CKPT_PATH"
echo "Dataset: $DATA_ROOT"

# 3. Run Python script
# Here we use --idx-a and --idx-b (refer to your CSV head for specific characters)
# You can modify the idx values to test different characters
python scripts/eval_interpolation_multimae.py \
    --ckpt ... \
    --svg-dir ... \
    --img-dir ... \
    --meta ... \
    --idx-a 3 \
    --idx-b 0 \
    --num-steps 8 \
    --out-dir interp_output_multimae_test_b_to_0_16 \
    --device cuda \
    --log-level INFO \
    --save-png \
    --viewbox-size 256 \
    --max-num-groups 8 \
    --max-seq-len 40

echo "Done. Results saved to $OUT_DIR"
