#!/bin/bash
#
# Master Benchmark Orchestration Script
# ======================================
# This script:
# 1. Prepares the dataset once (download + crop + split)
# 2. Submits all three training jobs (CUDA, Gaudi Lazy, Gaudi Eager)
#

module load mamba/latest
source activate gaudi-pytorch-diffusion-1.22.0.740

echo "======================================================================="
echo "  UNet Semantic Segmentation Benchmark Suite"
echo "  Comparing NVIDIA A100 vs Intel Gaudi (Lazy & Eager Modes)"
echo "======================================================================="
echo ""

# Step 1: Prepare dataset
echo "[1/4] Preparing dataset..."
echo "-----------------------------------------------------------------------"
if [ -f "dataset_splits.json" ]; then
    echo "✓ Dataset already prepared (dataset_splits.json found)"
else
    echo "Running prepare_dataset.py..."
    python3 prepare_dataset.py
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Dataset preparation failed!"
        exit 1
    fi
    echo "✓ Dataset preparation complete"
fi
echo ""

# Step 2: Submit CUDA training job
echo "[2/4] Submitting CUDA training job..."
echo "-----------------------------------------------------------------------"
CUDA_JOB=$(sbatch submit_cuda.sh | awk '{print $4}')
if [ -z "$CUDA_JOB" ]; then
    echo "❌ ERROR: Failed to submit CUDA job"
    exit 1
fi
echo "✓ CUDA job submitted: $CUDA_JOB"
echo ""

# Step 3: Submit Gaudi Lazy Mode training job
echo "[3/4] Submitting Gaudi (Lazy Mode) training job..."
echo "-----------------------------------------------------------------------"
LAZY_JOB=$(sbatch submit_gaudi_lazy.sh | awk '{print $4}')
if [ -z "$LAZY_JOB" ]; then
    echo "❌ ERROR: Failed to submit Gaudi Lazy job"
    exit 1
fi
echo "✓ Gaudi Lazy job submitted: $LAZY_JOB"
echo ""

# Step 4: Submit Gaudi Eager Mode training job
echo "[4/4] Submitting Gaudi (Eager Mode) training job..."
echo "-----------------------------------------------------------------------"
EAGER_JOB=$(sbatch submit_gaudi_eager.sh | awk '{print $4}')
if [ -z "$EAGER_JOB" ]; then
    echo "❌ ERROR: Failed to submit Gaudi Eager job"
    exit 1
fi
echo "✓ Gaudi Eager job submitted: $EAGER_JOB"
echo ""

# Summary
echo "======================================================================="
echo "  All jobs submitted successfully!"
echo "======================================================================="
echo ""
echo "Job IDs:"
echo "  CUDA (A100):        $CUDA_JOB"
echo "  Gaudi Lazy Mode:    $LAZY_JOB"
echo "  Gaudi Eager Mode:   $EAGER_JOB"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $CUDA_JOB,$LAZY_JOB,$EAGER_JOB"
echo ""
echo "Check outputs:"
echo "  tail -f *_unet_*.out"
echo ""
echo "After completion, compare results:"
echo "  python3 compare_results.py"
echo ""
echo "======================================================================="
