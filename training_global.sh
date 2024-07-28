#!/bin/bash
#SBATCH --time=6-0  # Set the time limit to 3 hours
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v8
#SBATCH -o slurm_logs/slurm-vimeo-%A_%x.out
#SBATCH -e slurm_logs/slurm-vimeo-%A_%x.err

#Your command to run the Python script
#pip install torch torchvision compressai
#source activate compressails
python src/train/train_global.py --config-path ./configs/vimeo_img2img/global_v15.yaml ---resume-path ./ckpt/vimeo_img2img/init_global.ckpt ---gpus 6 ---logdir logs/vimeo_img2img/global
