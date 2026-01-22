#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=dino_lunit
#SBATCH --output=dino_%j.out
#SBATCH --error=dino_%j.err

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

############################
# 1. Load required modules
############################
module purge
module load cuda/12.1.1
module load anaconda3/2024.06

########################################
# 2. Initialize conda
########################################
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh

########################################
# 3. Activate zerosim_env
########################################
echo "Activating conda environment: zerosim_env"
conda activate zerosim_env

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate environment zerosim_env"
    conda env list
    exit 1
fi

echo "Active environment: $CONDA_DEFAULT_ENV"

########################################
# 4. Quick environment check (no cuDNN test)
########################################
echo "=========================================="
echo "Environment check..."
echo "=========================================="

python -c "import sys; print('Python:', sys.version.split()[0])"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import timm; print('timm:', timm.__version__)"

echo "=========================================="

########################################
# 5. Navigate to project directory
########################################
cd /home/senthilkumar.m/Dino_Pathology

if [ ! -f train.py ]; then
    echo "ERROR: train.py not found in $(pwd)"
    exit 1
fi

if [ ! -f lunit_vit_small_dino.pth ]; then
    echo "ERROR: Lunit weights not found!"
    echo "Expected: $(pwd)/lunit_vit_small_dino.pth"
    exit 1
fi

echo "=========================================="
echo "Starting DINO Continued SSL Training"
echo "Working directory: $(pwd)"
echo "Lunit weights: ✓ Found"
echo "=========================================="

########################################
# 6. Run training
########################################
python train.py

########################################
# 7. Job completion
########################################
EXIT_CODE=$?
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE