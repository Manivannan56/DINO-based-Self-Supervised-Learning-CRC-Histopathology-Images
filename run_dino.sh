#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=dino_ssl
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
# 3. Activate zerosim_env (already set up)
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
# 4. Verify all packages are available
########################################
echo "=========================================="
echo "Verifying packages..."
echo "=========================================="

python - << 'EOF'
import sys
import torch
import torchvision
import numpy
import PIL
import tqdm
import timm
import matplotlib
import seaborn
import sklearn

print("✓ Python:", sys.version.split()[0])
print("✓ PyTorch:", torch.__version__)
print("✓ torchvision:", torchvision.__version__)
print("✓ numpy:", numpy.__version__)
print("✓ Pillow:", PIL.__version__)
print("✓ tqdm:", tqdm.__version__)
print("✓ timm:", timm.__version__)
print("✓ matplotlib:", matplotlib.__version__)
print("✓ seaborn:", seaborn.__version__)
print("✓ scikit-learn:", sklearn.__version__)
print()
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
    print("CUDA version:", torch.version.cuda)
else:
    print("ERROR: CUDA not available!")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed"
    exit 1
fi

########################################
# 5. Navigate to project directory
########################################
cd /home/senthilkumar.m/Dino_Pathology

if [ ! -f train.py ]; then
    echo "ERROR: train.py not found in $(pwd)"
    exit 1
fi

echo "=========================================="
echo "Starting DINO training..."
echo "Working directory: $(pwd)"
echo "=========================================="

########################################
# 6. Run training
########################################
python train.py

########################################
# 7. Job completion
########################################
echo "=========================================="
echo "Job finished at: $(date)"
echo "Exit code: $?"
echo "=========================================="