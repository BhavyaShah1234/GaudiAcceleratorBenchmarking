#!/bin/bash
#
# Verification script for enhanced UNet benchmark suite
#

echo "======================================================================="
echo "  UNet Benchmark Suite - File Verification"
echo "======================================================================="
echo ""

check_file() {
    if [ -f "$1" ]; then
        echo "  ✓ $1"
        return 0
    else
        echo "  ✗ $1 (MISSING)"
        return 1
    fi
}

check_executable() {
    if [ -x "$1" ]; then
        echo "  ✓ $1 (executable)"
        return 0
    else
        echo "  ✗ $1 (not executable or missing)"
        return 1
    fi
}

errors=0

echo "Dataset Preparation:"
check_file "prepare_dataset.py" || ((errors++))
echo ""

echo "Training Scripts:"
check_file "unet_cuda.py" || ((errors++))
check_file "unet_gaudi_lazy.py" || ((errors++))
check_file "unet_gaudi_eager.py" || ((errors++))
echo ""

echo "SBATCH Submission Scripts:"
check_executable "submit_cuda.sh" || ((errors++))
check_executable "submit_gaudi_lazy.sh" || ((errors++))
check_executable "submit_gaudi_eager.sh" || ((errors++))
echo ""

echo "Orchestration Script:"
check_executable "run_benchmark.sh" || ((errors++))
echo ""

echo "Utility Scripts:"
check_file "compare_results.py" || ((errors++))
echo ""

echo "Documentation:"
check_file "ENHANCEMENT_SUMMARY.md" || ((errors++))
echo ""

echo "Backup Files:"
check_file "unet_cuda.py.backup" || echo "  ℹ unet_cuda.py.backup (optional)"
check_file "unet_gaudi_lazy.py.backup" || echo "  ℹ unet_gaudi_lazy.py.backup (optional)"
check_file "unet_gaudi_eager.py.backup" || echo "  ℹ unet_gaudi_eager.py.backup (optional)"
echo ""

echo "======================================================================="
if [ $errors -eq 0 ]; then
    echo "✓ All required files present!"
    echo ""
    echo "Quick Start:"
    echo "  1. Copy to Sol:       scp *.py *.sh <asurite>@sol.asu.edu:~/Gaudi/"
    echo "  2. Run benchmark:     ./run_benchmark.sh"
    echo "  3. Monitor jobs:      squeue -u \$USER"
    echo "  4. Compare results:   python3 compare_results.py"
else
    echo "✗ $errors file(s) missing!"
    echo ""
    echo "Please ensure all required files are present."
fi
echo "======================================================================="
