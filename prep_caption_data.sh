#!/bin/bash
#SBATCH --time=6-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -o slurm_logs/SLURM-%A_%x.out
#SBATCH -e slurm_logs/SLURM-%A_%x.err

#Your command to run the Python script
#pip install torch torchvision compressai
#source activate compressails
python prepare_dataset.py