#!/bin/bash
#SBATCH --time=6-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH -o slurm_logs/slurm-%A_%x.out
#SBATCH -e slurm_logs/slurm-%A_%x.err

python src/train/train.py \
    --config-path ./configs/vimeo_lpips/local_v15_lpips_01.yaml \
    ---resume-path ./checkpoints/vimeo_lpips_s_01/local-best-checkpoint-v1.ckpt \
    ---gpus 4 \
    ---batch-size 3 \
    ---logdir ./logs/vimeo_lpips_s_01/local/ \
    --checkpoint-dirpath ./checkpoints/vimeo_lpips_s_01/ \
    ---max-epochs 3 \
    ---num-workers 8


