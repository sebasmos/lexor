#!/bin/bash -l

####################################
# SLURM Job Configuration
####################################

#SBATCH --job-name=SegVol_Train              # Descriptive job name
#SBATCH -N 2                                  # Number of nodes (multi-node job)
#SBATCH --partition=csgpu                     # Use GPU partition for CS users
#SBATCH --gres=gpu:2                          # Request 2 GPUs per node
#SBATCH --ntasks-per-node=2                   # Match number of GPUs per node
#sbatch --nodelist=sonicmem3 cluster
#SBATCH -t 30:00:00                           # Maximum run time (6 hours)
#SBATCH --mail-type=ALL                       # Send email on start, end, and failure
#SBATCH --mail-user=viraj.aher@ucd.ie         # Email address for notifications
#SBATCH --output=logs/%x_%j.out               # Log file (%x = job name, %j = job ID)

####################################
# Environment Setup
####################################

# Print job details
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Submitted from: $SLURM_SUBMIT_DIR"
echo "Allocated GPUs per node: $SLURM_GPUS_ON_NODE"

# Load necessary modules (CUDA 12.6)


# Activate the conda environment containing SegVol dependencies
conda activate segvol

# Change to the directory where the script was submitted from
cd $SLURM_SUBMIT_DIR

# Optional: Limit number of OpenMP threads to avoid CPU oversubscription


####################################
# PyTorch DDP Configuration
####################################

# Master node and port for communication between processes

# Rank of the current node (needed by torchrun)

# Total number of processes = nproc_per_node * nnodes = 2 * 2 = 4

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_RANK: $NODE_RANK"
echo "Launching DDP with 2 processes per node across 2 nodes (4 GPUs total)..."

####################################
# Launch Training via torchrun
####################################

torchrun \
  --nproc_per_node=2 \
  train_fast2.py  # Replace with your training script and arguments as needed

####################################
# Post-run Info
####################################

echo "Training job finished at: $(date)"
