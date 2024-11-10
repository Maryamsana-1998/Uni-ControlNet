#!/bin/bash
#SBATCH --time=4-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v13
#SBATCH -o slurm_logs/slurm-%A_%x.out
#SBATCH -e slurm_logs/slurm-%A_%x.err

python src/train/train.py \
    --config-path configs/vimeo_vgg_percep/local_v15_percep.yaml \
    ---resume-path ckpt/vimeo/init_local.ckpt \
    ---max-epochs 1 \
    ---batch-size 3 \
    ---gpus 8 \
    ---logdir logs/vgg/ \
    --checkpoint-dirpath checkpoints/vimeo_vgg_0.05/

