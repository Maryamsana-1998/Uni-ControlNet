#!/bin/bash
#SBATCH --time=4-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH -o slurm_logs/slurm-%A_%x.out
#SBATCH -e slurm_logs/slurm-%A_%x.err

python src/train/train.py \
    --config-path ./configs/spring_op/local_v15.yaml \
    ---resume-path ckpt/vimeo/init_local.ckpt \
    ---max-epochs 1 \
    ---batch-size 4 \
    ---gpus 4 \
    ---logdir logs/spring/local/ \
    --checkpoint-dirpath checkpoints/spring/local

