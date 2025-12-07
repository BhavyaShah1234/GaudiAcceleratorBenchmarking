# UNet Benchmark Quick Reference

## Enhanced Metrics Now Tracked

### ✅ Timing Metrics
- **Total training time** (sec, min)
- **Average epoch time**
- **Average batch time**
- **Min/Max batch time**
- **Throughput** (samples/sec)

### ✅ Memory Metrics (GPU/HPU)
- **Model memory footprint**
- **Peak memory allocated**
- **Peak memory reserved**
- **Average memory allocated**

### ✅ Best Metrics Tracked
- **Best train loss** & **Best train IoU**
- **Best validation loss** & **Best validation IoU**

### ✅ Per-Epoch History
- Train/Val Loss
- Train/Val Accuracy
- Train/Val IoU
- Train/Val F1 Score
- Learning rates

### ✅ Test Metrics
- Accuracy, IoU, Precision, Recall, F1
- Per-class IoU

## Workflow Changes

### Before (Old Workflow):
```bash
# Each script downloaded dataset separately
python3 unet_cuda.py
python3 unet_gaudi_lazy.py
python3 unet_gaudi_eager.py
```

### After (New Workflow):
```bash
# Single command runs everything
./run_benchmark.sh

# Or manually:
python3 prepare_dataset.py  # Run once
sbatch submit_cuda.sh
sbatch submit_gaudi_lazy.sh
sbatch submit_gaudi_eager.sh
```

## Key Files

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Downloads CARLA dataset, creates crops, saves splits |
| `dataset_splits.json` | Contains train/val/test split info (created by prepare_dataset.py) |
| `unet_cuda.py` | Enhanced CUDA training with comprehensive metrics |
| `unet_gaudi_lazy.py` | Enhanced Gaudi Lazy mode training |
| `unet_gaudi_eager.py` | Enhanced Gaudi Eager mode training |
| `run_benchmark.sh` | Master script that runs dataset prep + submits all jobs |
| `results_cuda.json` | Comprehensive CUDA benchmark results |
| `results_gaudi_lazy.json` | Comprehensive Gaudi Lazy results |
| `results_gaudi_eager.json` | Comprehensive Gaudi Eager results |

## Results JSON Structure

Each `results_*.json` contains:
```json
{
  "device": "Device name",
  "pytorch_version": "X.X.X",
  "model_info": {
    "total_params": 31037780,
    "model_memory_gb": 0.118
  },
  "timing": {
    "total_training_time_sec": ...,
    "avg_epoch_time_sec": ...,
    "avg_batch_time_sec": ...,
    "throughput_samples_per_sec": ...
  },
  "memory": {
    "peak_memory_allocated_gb": ...,
    "peak_memory_reserved_gb": ...,
    "avg_memory_allocated_gb": ...
  },
  "best_metrics": {
    "train": {"loss": ..., "iou": ...},
    "validation": {"loss": ..., "iou": ...}
  },
  "test_metrics": {...},
  "history": {...}
}
```

## Commands

### On Sol Supercomputer:

```bash
# 1. Copy files
scp *.py *.sh bshah43@sol.asu.edu:~/Gaudi/

# 2. Login and run
ssh bshah43@sol.asu.edu
cd ~/Gaudi
./run_benchmark.sh

# 3. Monitor
watch -n 5 'squeue -u $USER'

# 4. Check logs
tail -f cuda_unet_*.out
tail -f gaudi_lazy_unet_*.out
tail -f gaudi_eager_unet_*.out

# 5. Compare results (after jobs complete)
python3 compare_results.py
```

### Locally (for testing CUDA only):

```bash
python3 prepare_dataset.py
python3 unet_cuda.py
```

## What Changed?

1. **Separated dataset preparation** - No more downloading dataset 3 times
2. **Comprehensive metrics** - All timing, memory, and performance metrics now tracked
3. **Best metrics tracking** - Automatically tracks best train/val loss and IoU
4. **Consistent splits** - All platforms use identical train/val/test splits
5. **Structured output** - Results saved in detailed JSON format
6. **Master script** - Single command to run entire benchmark suite

## Comparison

Use `compare_results.py` to generate:
- Side-by-side metric comparison
- Performance speedup calculations
- Memory usage comparison
- Training curves plots
- Markdown summary report
