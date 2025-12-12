#!/bin/bash
#SBATCH -J jepa_7_linprobe_family
#SBATCH -p 3090-gcondo
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:2
#SBATCH --mem=240G
#SBATCH -t 2-00:00:00
#SBATCH -o /oscar/scratch/zzhan215/sbatchout/jepa_7_linprobe_family_%j.txt

echo "Starting eval at: $(date)"
cd /oscar/home/zzhan215/csci2952project

source .venv/bin/activate

accelerate launch --num_processes=2 scripts/eval_linear_probe.py \
  --task family_label \
  --encoder-type jepa \
  --checkpoint /oscar/scratch/zzhan215/exps/jepa_7/checkpoint_0190.pt \
  --svg-dir /oscar/scratch/zzhan215/google_fonts_processed_reduced/svg/ \
  --img-dir /oscar/scratch/zzhan215/google_fonts_processed_reduced/img/ \
  --meta /oscar/scratch/zzhan215/new/metadata.csv \
  --epochs 100 --num-workers 0 \
  --wandb-project svgvae --wandb-entity vecssl \
  --wandb-name jepa_7_linprobe_family \
  --checkpoint-dir /oscar/scratch/zzhan215/evals/jepa_7_linprobe_family \
  --cache

echo "Finished eval at: $(date)"
