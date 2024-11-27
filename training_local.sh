#!/bin/bash
#SBATCH --time=6-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v10
#SBATCH -o slurm_logs/slurm-%A_%x.out
#SBATCH -e slurm_logs/slurm-%A_%x.err

python src/train/train.py \
    --config-path ./configs/local_v15.yaml \
    ---resume-path ./checkpoints/vimeo_8/local-best-checkpoint.ckpt \
    ---gpus 7 \
    ---batch-size 3 \
    ---logdir ./logs/vimeo_8/ \
    --checkpoint-dirpath ./checkpoints/vimeo/ \
    ---max-epochs 4 \
    ---num-workers 8

