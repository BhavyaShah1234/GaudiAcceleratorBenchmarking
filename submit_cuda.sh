#!/bin/bash
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_eee59881079fall2025
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=64GB
#SBATCH -t 0-8:00:00
#SBATCH -o cuda_unet_%j.out
#SBATCH -e cuda_unet_%j.err
#SBATCH --job-name=unet_cuda
#SBATCH --export=NONE

source ~/.bashrc

echo "============================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "============================================="

# Load mamba module and activate shah kernel for CUDA
module load mamba/latest

nvidia-smi

echo "\n\n"

watch -n 2 nvidia-smi > nvidia_smi_${SLURM_JOB_ID}.log 2>&1 &
MONITOR_PID=$!

# Run the training script using shah kernel
mamba run -n shah python3 unet_cuda.py 2>&1 | tee unet_cuda_${SLURM_JOB_ID}.log

# Stop nvidia-smi monitoring
kill $MONITOR_PID 2>/dev/null

echo "============================================="
echo "Job finished at: $(date)"
echo "============================================="
