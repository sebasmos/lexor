#!/bin/bash -l
#SBATCH --job-name=SegVol_Train
#SBATCH -N 2
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:2                     # 2 GPUs per node
#SBATCH --ntasks-per-node=2              # One task per GPU
#SBATCH -t 06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=viraj.aher@ucd.ie
#SBATCH --output=logs/%x_%j.out
# Load required module
module load cuda/12.6

conda activate segvol

# Set torchrun environment variables
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
NODE_RANK=$SLURM_NODEID

# Run your script using torchrun
torchrun \
  --nproc_per_node=2 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py  # Replace with your actual script and any needed arguments
