#!/bin/bash
#SBATCH --time=4-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2

#Your command to run the Python script
#pip install torch torchvision compressai
#source activate compressails
python src/train/train_video.py --config-path configs/video_uni_v15/local_v15-seg.yaml ---training-steps 50000 --ckpt checkpoints/video_local/ ---gpus 4
