#!/bin/bash
#SBATCH --time=6-0
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH -o experiment_3/slurm.out
#SBATCH -e experiment_3/slurm.err

# Set up directories
EXPERIMENT_DIR="experiment_3"
LOCAL_CKPT_DIR="${EXPERIMENT_DIR}/local_ckpt"
LOGS_DIR="${EXPERIMENT_DIR}/logs"
PRED_DIR="${EXPERIMENT_DIR}/preds"


mkdir -p ${EXPERIMENT_DIR} ${LOCAL_CKPT_DIR} ${LOGS_DIR}

# Training parameters
CONFIG_PATH="./configs/vimeo_lpips/local_v15_lpips_01_static.yaml"
INIT_CKPT="./ckpt/init_local.ckpt"
NUM_GPUS=4
BATCH_SIZE=2
NUM_WORKERS=8
MAX_EPOCHS=3


# Copy config file to experiment directory
cp ${CONFIG_PATH} ${EXPERIMENT_DIR}/local_v15.yaml
echo "Config file copied to ${EXPERIMENT_DIR}/local_v15.yaml"

# Create a JSON file of training hyperparameters
HYPERPARAM_FILE="${EXPERIMENT_DIR}/hyperparams.json"

cat <<EOF > ${HYPERPARAM_FILE}
{
    "num_gpus": ${NUM_GPUS},
    "batch_size": ${BATCH_SIZE},
    "num_workers": ${NUM_WORKERS},
    "max_epochs": ${MAX_EPOCHS},
    "config":${CONFIG_PATH},
    "init_ckpt": ${INIT_CKPT},
    "loss":"baseline+lpips",
    "data": "30"
}
EOF

echo "Hyperparameters JSON saved at ${HYPERPARAM_FILE}"

# Run Training
python src/train/train_sub.py \
    --config-path ${CONFIG_PATH} \
    ---resume-path ${INIT_CKPT} \
    ---gpus ${NUM_GPUS} \
    ---batch-size ${BATCH_SIZE} \
    ---logdir ${LOGS_DIR} \
    --checkpoint-dirpath ${LOCAL_CKPT_DIR} \
    ---max-epochs ${MAX_EPOCHS} \
    ---num-workers ${NUM_WORKERS}

# After training, prepare uni weights
LOCAL_BEST="${LOCAL_CKPT_DIR}/local-best-checkpoint.ckpt"
UNI_CKPT="${EXPERIMENT_DIR}/uni.ckpt"
UNI_CONFIG="configs/vimeo_lpips/uni_v15.yaml"

python utils/prepare_weights.py integrate \
       ${LOCAL_BEST} ckpt/init_global.ckpt  \
       ${UNI_CONFIG} ${UNI_CKPT} 

echo "Unified weights prepared and stored at ${UNI_CKPT}."
echo "Experiment finished successfully."

python eval_uvg.py --original_root /data/maryam.sana/vimeo_unicontrol/Uni-ControlNet/data/UVG/ \
                   --pred_root ${PRED_DIR}  \
                   --config /path/to/config.yaml \
                   --ckpt ${UNI_CKPT}

