# UNet Semantic Segmentation Benchmark Suite

Comprehensive benchmarking of UNet training on NVIDIA A100 (CUDA) vs Intel Gaudi (Lazy & Eager modes).

## Files

### Training Scripts
- `unet_cuda.py` - NVIDIA A100 / CUDA training
- `unet_gaudi_lazy.py` - Intel Gaudi Lazy mode training
- `unet_gaudi_eager.py` - Intel Gaudi Eager mode training

### SBATCH Submission Scripts
- `submit_cuda.sh` - Submit CUDA job (A100, public QOS)
- `submit_gaudi_lazy.sh` - Submit Gaudi Lazy job (class_gaudi QOS)
- `submit_gaudi_eager.sh` - Submit Gaudi Eager job (class_gaudi QOS)

### Utilities
- `prepare_dataset.py` - Download CARLA dataset, create crops, generate splits
- `run_benchmark.sh` - Master script: prepare data + submit all jobs
- `compare_results.py` - Compare results from all three platforms
- `verify_enhanced.sh` - Verify all files are present

## Quick Start

### 1. Copy to Sol
```bash
scp *.py *.sh <asurite>@sol.asu.edu:~/Gaudi/
```

### 2. Run Benchmark Suite
```bash
ssh <asurite>@sol.asu.edu
cd ~/Gaudi
./run_benchmark.sh
```

This will:
1. Download CARLA dataset (if needed)
2. Generate random crops (384x512)
3. Create train/val/test splits
4. Submit all 3 training jobs to SLURM

### 3. Monitor Jobs
```bash
watch -n 5 'squeue -u $USER'
```

### 4. Check Logs
```bash
tail -f cuda_unet_*.out
tail -f gaudi_lazy_unet_*.out
tail -f gaudi_eager_unet_*.out
```

### 5. Compare Results
After all jobs complete:
```bash
python3 compare_results.py
```

## Manual Workflow

If you prefer to run steps manually:

```bash
# 1. Prepare dataset once
python3 prepare_dataset.py

# 2. Submit training jobs
sbatch submit_cuda.sh
sbatch submit_gaudi_lazy.sh
sbatch submit_gaudi_eager.sh

# 3. After completion
python3 compare_results.py
```

## Metrics Tracked

### Timing
- Total training time (sec, min)
- Average epoch time
- Average/Min/Max batch time
- Throughput (samples/sec)

### Memory
- Model memory footprint (GB)
- Peak memory allocated (GB)
- Peak memory reserved (GB)
- Average memory allocated (GB)

### Model Performance
- Best train/val loss and IoU
- Per-epoch: loss, accuracy, IoU, F1, learning rate
- Test: accuracy, IoU, precision, recall, F1, per-class IoU

## Output Files

After training completes:
- `results_cuda.json` - Comprehensive CUDA results
- `results_gaudi_lazy.json` - Gaudi Lazy mode results
- `results_gaudi_eager.json` - Gaudi Eager mode results
- `best_unet_cuda.pth` - Best CUDA model checkpoint
- `best_unet_gaudi_lazy.pth` - Best Gaudi Lazy checkpoint
- `best_unet_gaudi_eager.pth` - Best Gaudi Eager checkpoint
- `dataset_splits.json` - Train/val/test split information

## Configuration

All scripts use identical configuration:
- Image size: 384x512
- Batch size: 16
- Epochs: 20
- Learning rate: 1e-4
- Workers: 4
- Classes: 20 (CARLA semantic segmentation)
- Random seed: 42 (for reproducibility)

## Notes

- Requires Kaggle credentials in `~/.kaggle/kaggle.json`
- Dataset: CARLA Vision Dataset from Kaggle
- All platforms use identical train/val/test splits for fair comparison
- Results saved in JSON format for easy analysis
