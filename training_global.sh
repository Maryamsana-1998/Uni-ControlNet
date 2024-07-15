#!/bin/bash
#SBATCH --time=4-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v8
#SBATCH -o slurm_logs/slurm-laion-%A_%x.out

python src/train/train.py --config-path ./configs/laicon/global_v15.yaml  ---resume-path ckpt/laicon/init_global.ckpt ---max-epochs 3  ---gpus 8 ---logdir global_logs/
