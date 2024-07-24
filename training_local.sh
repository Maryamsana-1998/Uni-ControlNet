#!/bin/bash
#SBATCH --time=6-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v4
#SBATCH -o slurm_logs/slurm-vimeo-%A_%x.out
#SBATCH -e slurm_logs/slurm-vimeo-%A_%x.err


#Your command to run the Python script
#pip install torch torchvision compressai
#source activate compressails
python src/train/train.py --config-path ./configs/vimeo_img2img/local_v15.yaml ---resume-path ckpt/vimeo_img2img/init_local.ckpt --ckpt ./checkpoints/vimeo_img2img/local/ ---gpus 8 ---logdir logs/vimeo_img2img/local/

