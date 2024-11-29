#!/bin/bash
#SBATCH --time=6-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -o slurm_logs/slurm-%A_%x.out
#SBATCH -e slurm_logs/slurm-%A_%x.err

python src/train/train_sub.py \
    --config-path ./configs/vimeo_lpips/local_v15_lpips_01.yaml \
    ---resume-path ./ckpt/init_local.ckpt \
    ---gpus 4 \
    ---batch-size 2 \
    ---logdir ./logs/vimeo_lpips_01/local/ \
    --checkpoint-dirpath ./checkpoints/vimeo_lpips_01/ \
    ---max-epochs 4 \
    ---num-workers 8


