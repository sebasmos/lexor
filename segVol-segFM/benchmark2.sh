#!/bin/bash -l
#SBATCH --job-name=VISTA3D_Validation
#SBATCH -N 1
#SBATCH --partition=csgpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=3
#SBATCH -t 06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=viraj.aher@ucd.ie

# Load apptainer module
module load apptainer/1.3.4-gcc-11.5.0-ojp6nts

# Define paths relative to working directory
WORK_DIR="/home/people/vaher/scratch/VISTA/vista3d/cvpr_workshop"
INPUT_DIR="${WORK_DIR}/validation_data/3D_val_npz"
GT_DIR="${WORK_DIR}/validation_data/3D_val_gt/3D_val_gt/3D_val_gt_interactive"
OUTPUT_DIR="${WORK_DIR}/validation_output"
CONTAINER_PATH="${WORK_DIR}/vista3d.sif"
MODEL_PATH="${WORK_DIR}/CPRR25_vista3D_model_final_10percent_data.pth"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Print job info
echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "Input directory: ${INPUT_DIR}"
echo "Ground truth directory: ${GT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Container path: ${CONTAINER_PATH}"
echo "Model path: ${MODEL_PATH}"

# Check if input NPZ files exist
echo "Checking input files:"
INPUT_COUNT=$(find ${INPUT_DIR} -name "*.npz" | wc -l)
echo "Found ${INPUT_COUNT} input NPZ files"
if [ "${INPUT_COUNT}" -eq 0 ]; then
  echo "ERROR: No input NPZ files found!"
  exit 1
fi

# Check if GT files exist
echo "Checking GT files:"
GT_COUNT=$(find ${GT_DIR} -name "*.npz" | wc -l)
echo "Found ${GT_COUNT} GT NPZ files"
if [ "${GT_COUNT}" -eq 0 ]; then
  echo "ERROR: No GT NPZ files found!"
  exit 1
fi

# First file for debugging (optional)
FIRST_INPUT=$(find ${INPUT_DIR} -name "*.npz" | head -1)
echo "First input file: ${FIRST_INPUT}"

# Debug mode: Test a single file first (optional)
echo "Running debug test on first file..."
apptainer exec --nv \
  --bind ${WORK_DIR}:/workspace \
  ${CONTAINER_PATH} \
  python3 /workspace/validation_infer.py \
  --input $(echo ${FIRST_INPUT} | sed "s|${WORK_DIR}|/workspace|") \
  --gt_dir /workspace/validation_data/3D_val_gt/3D_val_gt/3D_val_gt_interactive \
  --output /workspace/validation_output \
  --model /workspace/CPRR25_vista3D_model_final_10percent_data.pth \
  --save_nifti \
  --debug

# Run container with proper workspace binding for full validation
echo "Running full validation with parallel processing..."
apptainer exec --nv \
  --bind ${WORK_DIR}:/workspace \
  ${CONTAINER_PATH} \
  python3 /workspace/validation_infer.py \
  --input /workspace/validation_data/3D_val_npz \
  --gt_dir /workspace/validation_data/3D_val_gt/3D_val_gt/3D_val_gt_interactive \
  --output /workspace/validation_output \
  --model /workspace/CPRR25_vista3D_model_final_10percent_data.pth \
  --save_nifti \
  --num_workers 3

# Check if job completed successfully
if [ $? -eq 0 ]; then
  echo "Validation job completed successfully at $(date)"
  
  # Show summary
  echo "Dice Score Summary:"
  cat ${OUTPUT_DIR}/summary.txt
else
  echo "Validation job failed at $(date)"
  echo "Error details may be above."
fi
