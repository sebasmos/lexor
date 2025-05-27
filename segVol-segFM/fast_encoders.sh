#!/bin/bash -l
####################################
# SLURM Job Configuration
####################################
#SBATCH --job-name=SegVol_FastEnc             # Updated job name to reflect fast encoder experiments
#SBATCH -N 1                                  # Number of nodes (multi-node job)
#SBATCH --partition=csgpu                     # Use GPU partition for CS users
#SBATCH --gres=gpu:2                          # Request 2 GPUs per node
#SBATCH --ntasks-per-node=2                   # Match number of GPUs per node
#SBATCH -t 00:50:00                           # Maximum run time (30 hours)
#SBATCH --mail-type=ALL                       # Send email on start, end, and failure
#SBATCH --mail-user=viraj.aher@ucd.ie         # Email address for notifications
#SBATCH --output=logs/%x_%j.out               # Log file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err                # Error log file
####################################
# Environment Setup
####################################
# Create logs directory if it doesn't exist
mkdir -p logs

# Print job details
echo "==============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Submitted from: $SLURM_SUBMIT_DIR"
echo "Allocated GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_GPUS_ON_NODE))"
echo "Start time: $(date)"
echo "==============================================="

# Load necessary modules (CUDA 12.6)
# module load cuda/12.6  # Uncomment if needed

# Activate the conda environment containing SegVol dependencies
conda activate segvol

# Change to the directory where the script was submitted from
cd $SLURM_SUBMIT_DIR

# Optional: Limit number of OpenMP threads to avoid CPU oversubscription
export OMP_NUM_THREADS=1

####################################
# Fast Encoder Configuration
####################################
# Set the encoder type for this experiment
# Available options: original, fast_vit, ultra_fast_vit, mobilenet_2_5d, 
#                   efficientnet_2_5d, mobilenet_3d, hybrid_cnn_vit, fast_resnet3d
ENCODER_TYPE="${ENCODER_TYPE:-mobilenet_2_5d}"  # Default to fastest encoder
EXPERIMENT_NAME="segvol_${ENCODER_TYPE}_$(date +%Y%m%d_%H%M%S)"

echo "Fast Encoder Configuration:"
echo "  - Encoder Type: $ENCODER_TYPE"
echo "  - Experiment Name: $EXPERIMENT_NAME"

####################################
# PyTorch DDP Configuration
####################################
# Set up distributed training environment
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export NODE_RANK=$SLURM_NODEID

echo "PyTorch DDP Configuration:"
echo "  - MASTER_ADDR: $MASTER_ADDR"
echo "  - MASTER_PORT: $MASTER_PORT"
echo "  - WORLD_SIZE: $WORLD_SIZE"
echo "  - NODE_RANK: $NODE_RANK"
echo "  - Processes per node: $SLURM_NTASKS_PER_NODE"
echo "==============================================="

####################################
# Training Arguments
####################################
# Configure training parameters based on encoder type
if [[ "$ENCODER_TYPE" == "mobilenet_2_5d" || "$ENCODER_TYPE" == "ultra_fast_vit" ]]; then
    # Fast encoders can handle larger batch sizes
    BATCH_SIZE=4
    LOG_TIMING=true
elif [[ "$ENCODER_TYPE" == "original" ]]; then
    # Original encoder - conservative settings
    BATCH_SIZE=4
    LOG_TIMING=false
else
    # Default settings for other encoders
    BATCH_SIZE=4
    LOG_TIMING=true
fi

echo "Training Configuration:"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Log Timing: $LOG_TIMING"
echo "  - Experiment Directory: ./ckpts_${EXPERIMENT_NAME}"
echo "==============================================="

####################################
# Launch Training via torchrun
####################################
echo "Launching DDP training with fast encoder: $ENCODER_TYPE"
echo "Command: torchrun --nproc_per_node=$SLURM_NTASKS_PER_NODE train_fast2.py [args...]"
echo "==============================================="

torchrun \
  --nproc_per_node=$SLURM_NTASKS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train_fast2.py \
    --fast_encoder_type $ENCODER_TYPE \
    --batch_size $BATCH_SIZE \
    --save_dir "./ckpts_${EXPERIMENT_NAME}" \
    --benchmark_encoders \
    $([ "$LOG_TIMING" = true ] && echo "--log_inference_time") \
    --num_epochs 3000 \
    --initial_lr 1e-5 \
    --train_root "/scratch/vaher/SegFM3D/3D_train_npz_random_10percent_16G" \
    --resume_ckpt "SegVol_v1.pth" \
    --model_dir "./segvol"

torchrun --nproc_per_node=$2 \
  train_fast2.py \
    --fast_encoder_type mobilenet_2_5d \
    --batch_size 4 \
    --save_dir "./ckpts_mobilenet_2_5d" \
    --num_epochs 3000 \
    --initial_lr 1e-5 \
    --train_root "./3D_train_npz_random_10percent_16G" \
    --resume_ckpt "SegVol_v1.pth" \
    --model_dir "./segvol"


####################################
# Post-run Info
####################################
TRAINING_EXIT_CODE=$?

echo "==============================================="
echo "Training job finished at: $(date)"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "Encoder used: $ENCODER_TYPE"
echo "Experiment name: $EXPERIMENT_NAME"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Optional: Run a quick validation/test on the best model
    echo "Latest checkpoint: ./ckpts_${EXPERIMENT_NAME}/latest.pth"
    echo "Best model: ./ckpts_${EXPERIMENT_NAME}/best_model_${ENCODER_TYPE}.pth"
else
    echo "Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "Check the error logs for details."
fi

echo "==============================================="

# Optional: Clean up temporary files or send notification
# You can add cleanup commands here
