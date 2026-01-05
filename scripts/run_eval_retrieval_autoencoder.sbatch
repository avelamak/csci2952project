#!/bin/bash
#SBATCH -J vecssl_eval_auto
#SBATCH -p gpu               
#SBATCH --gres=gpu:1
#SBATCH --mem=128G          # max i can use is 128
#SBATCH -t 06:00:00
#SBATCH -o .../vecssl_eval_auto_%j.txt

echo "Starting eval at: $(date)"
cd ...

source .venv/bin/activate

python scripts/eval_retrieval.py \
  --encoder-type autoencoder \
  --checkpoint ... \
  --svg-dir ... \
  --img-dir ... \
  --meta ... \
  --batch-size 4 \
  --max-seq-len 60 \
  --max-num-groups 8 \
  --max-eval-samples 2000 \
  --device cuda \
  --num-workers 0


echo "Finished eval at: $(date)"
