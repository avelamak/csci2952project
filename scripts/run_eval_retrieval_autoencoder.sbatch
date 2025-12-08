#!/bin/bash
#SBATCH -J vecssl_eval_auto
#SBATCH -p gpu               # 如果你们 Oscar 分区名不同，改这里
#SBATCH --gres=gpu:1
#SBATCH --mem=128G          # max i can use is 128
#SBATCH -t 06:00:00
#SBATCH -o /oscar/home/jshen95/vecssl_eval_auto_%j.txt

echo "Starting eval at: $(date)"
cd /oscar/home/jshen95/csci2952project

# 激活虚拟环境
source .venv/bin/activate

python scripts/eval_retrieval.py \
  --encoder-type autoencoder \
  --checkpoint /oscar/scratch/zzhan215/exp_fonts_multi/best_model.pt \
  --svg-dir /oscar/scratch/zzhan215/google_fonts_processed_reduced/svg \
  --img-dir /oscar/scratch/zzhan215/google_fonts_processed_reduced/img \
  --meta /oscar/scratch/zzhan215/google_fonts_processed_reduced/metadata.csv \
  --batch-size 4 \
  --max-seq-len 60 \
  --max-num-groups 8 \
  --max-eval-samples 2000 \
  --device cuda \
  --num-workers 0


echo "Finished eval at: $(date)"
