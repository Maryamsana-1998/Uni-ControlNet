#!/bin/bash
#SBATCH --time=4-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v13

#Your command to run the Python script
#pip install torch torchvision compressai
#source activate compressails
python src/train/train.py --config-path ./configs/local_v15.yaml ---resume-path ckpt/init_local.ckpt --ckpt /checkpoints/local_all ---training-steps 200000 ---gpus 6 ---logdir local_logs/

