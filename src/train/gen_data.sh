#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v12
#SBATCH -o slurm.out
#SBATCH -e slurm.err

python gen_depthmaps.py