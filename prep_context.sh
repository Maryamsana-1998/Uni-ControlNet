#!/bin/bash
#SBATCH --time=4-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v8

#Your command to run the Python script
#pip install torch torchvision compressai
#source activate compressails
python prepare_global_features.py