#!/bin/bash
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -A class_eee59881079fall2025
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH -t 0-8:00:00
#SBATCH -o gaudi_lazy_unet_%j.out
#SBATCH -e gaudi_lazy_unet_%j.err
#SBATCH --job-name=unet_gaudi_lazy
#SBATCH --export=NONE

source ~/.bashrc

echo "============================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "============================================="

# Load mamba module and activate Gaudi PyTorch kernel
module load mamba/latest

hl-smi

echo "\n\n"

# Start hl-smi monitoring in background
watch -n 2 hl-smi > hl_smi_lazy_${SLURM_JOB_ID}.log 2>&1 &
MONITOR_PID=$!

# Prevent user-installed packages from conflicting with Gaudi kernel
unset PYTHONPATH
unset PYTHONUSERBASE
export PYTHONNOUSERSITE=1

# Install required packages in the mamba environment
echo "Installing required packages..."
mamba run -n gaudi-pytorch-diffusion-1.22.0.740 pip install --user scikit-learn tqdm kagglehub

# Set environment variable for HPU Lazy Mode
export PT_HPU_LAZY_MODE=1

# Run the training script (Lazy Mode) using Gaudi kernel with -s flag to ignore user site-packages
mamba run -n gaudi-pytorch-diffusion-1.22.0.740 python3 -s unet_gaudi_lazy.py 2>&1 | tee unet_gaudi_lazy_${SLURM_JOB_ID}.log

# Stop hl-smi monitoring
kill $MONITOR_PID 2>/dev/null

echo "============================================="
echo "Job finished at: $(date)"
echo "============================================="
