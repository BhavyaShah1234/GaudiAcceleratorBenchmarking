"""
Benchmark Results Comparison and Analysis Tool

This script loads all three result JSON files and generates a comprehensive
comparison report for the UNet semantic segmentation benchmark.

Usage:
    python3 compare_results.py

Make sure you have run all three training scripts and have:
    - results_cuda.json
    - results_gaudi_lazy.json
    - results_gaudi_eager.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not parse {filepath}")
        return None

def print_section(title: str):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def compare_performance(results: Dict[str, Dict]):
    """Compare performance metrics across all devices"""
    print_section("PERFORMANCE COMPARISON")
    
    # Training time
    print("\n📊 Training Time:")
    print(f"{'Device':<25} {'Total Time':<15} {'Avg Epoch':<15} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = None
    for name, data in results.items():
        if data is None:
            continue
        total_time = data['total_training_time']
        avg_epoch = data['avg_epoch_time']
        
        if baseline_time is None:
            baseline_time = total_time
            speedup = "1.00x (baseline)"
        else:
            speedup = f"{baseline_time / total_time:.2f}x"
        
        print(f"{name:<25} {total_time:>10.2f}s {avg_epoch:>10.2f}s     {speedup}")
    
    # Memory usage
    print("\n💾 Memory Usage:")
    print(f"{'Device':<25} {'Peak Memory (GB)':<20}")
    print("-" * 70)
    
    for name, data in results.items():
        if data is None:
            continue
        memory = data['test_metrics']['max_memory_gb']
        print(f"{name:<25} {memory:>15.2f} GB")

def compare_accuracy(results: Dict[str, Dict]):
    """Compare model quality metrics"""
    print_section("MODEL QUALITY COMPARISON")
    
    metrics = ['accuracy', 'mean_iou', 'mean_precision', 'mean_recall', 'mean_f1']
    
    print(f"\n{'Device':<25} {'Accuracy':<12} {'Mean IoU':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 95)
    
    for name, data in results.items():
        if data is None:
            continue
        
        test = data['test_metrics']
        print(f"{name:<25} "
              f"{test['accuracy']:>10.4f}  "
              f"{test['mean_iou']:>10.4f}  "
              f"{test['mean_precision']:>10.4f}  "
              f"{test['mean_recall']:>10.4f}  "
              f"{test['mean_f1']:>10.4f}")

def print_recommendations(results: Dict[str, Dict]):
    """Print recommendations based on results"""
    print_section("RECOMMENDATIONS")
    
    # Find fastest
    fastest = None
    fastest_time = float('inf')
    for name, data in results.items():
        if data is None:
            continue
        if data['total_training_time'] < fastest_time:
            fastest_time = data['total_training_time']
            fastest = name
    
    # Find most accurate
    most_accurate = None
    best_iou = 0
    for name, data in results.items():
        if data is None:
            continue
        if data['test_metrics']['mean_iou'] > best_iou:
            best_iou = data['test_metrics']['mean_iou']
            most_accurate = name
    
    # Find most memory efficient
    most_efficient = None
    lowest_memory = float('inf')
    for name, data in results.items():
        if data is None:
            continue
        if data['test_metrics']['max_memory_gb'] < lowest_memory:
            lowest_memory = data['test_metrics']['max_memory_gb']
            most_efficient = name
    
    print(f"\n🏆 Fastest Training: {fastest} ({fastest_time:.2f}s)")
    print(f"🎯 Best Accuracy: {most_accurate} (IoU: {best_iou:.4f})")
    print(f"💡 Most Memory Efficient: {most_efficient} ({lowest_memory:.2f} GB)")
    
    # Analysis
    print("\n📝 Analysis:")
    
    if fastest and 'Gaudi' in fastest:
        print(f"   • Intel Gaudi shows faster training than NVIDIA A100")
        if 'Lazy' in fastest:
            print(f"   • Lazy mode provides best performance for graph-optimizable workloads")
    
    if all(data is not None for data in results.values()):
        iou_diff = max(d['test_metrics']['mean_iou'] for d in results.values() if d) - \
                   min(d['test_metrics']['mean_iou'] for d in results.values() if d)
        
        if iou_diff < 0.01:
            print(f"   • All devices achieve similar accuracy (IoU diff: {iou_diff:.4f})")
            print(f"   • Model quality is hardware-independent ✓")
        else:
            print(f"   • Accuracy varies across devices (IoU diff: {iou_diff:.4f})")
            print(f"   • Consider investigating source of variation")

def print_detailed_metrics(results: Dict[str, Dict]):
    """Print detailed breakdown of metrics"""
    print_section("DETAILED METRICS")
    
    for name, data in results.items():
        if data is None:
            continue
        
        print(f"\n{name}:")
        print(f"  Device: {data['device']}")
        print(f"  Model Parameters: {data['model_params']['total']:,}")
        print(f"  Training Configuration:")
        print(f"    - Batch Size: {data['config']['batch_size']}")
        print(f"    - Epochs: {data['config']['num_epochs']}")
        print(f"    - Learning Rate: {data['config']['learning_rate']}")
        print(f"  Performance:")
        print(f"    - Total Training Time: {data['total_training_time']:.2f}s")
        print(f"    - Average Epoch Time: {data['avg_epoch_time']:.2f}s")
        print(f"    - Throughput: {data['config']['batch_size'] * len(data['history']['train_loss']) / data['total_training_time']:.2f} batches/s")
        print(f"  Test Metrics:")
        for metric, value in data['test_metrics'].items():
            if isinstance(value, float):
                print(f"    - {metric}: {value:.4f}")

def generate_plot_commands(results: Dict[str, Dict]):
    """Generate Python code for plotting results"""
    print_section("VISUALIZATION")
    
    print("""
To visualize the results, run this Python code:

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results_cuda.json') as f:
    cuda = json.load(f)
with open('results_gaudi_lazy.json') as f:
    gaudi_lazy = json.load(f)
with open('results_gaudi_eager.json') as f:
    gaudi_eager = json.load(f)

# Plot training time comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Training time comparison
devices = ['CUDA\\n(A100)', 'Gaudi\\n(Lazy)', 'Gaudi\\n(Eager)']
times = [cuda['total_training_time'], 
         gaudi_lazy['total_training_time'], 
         gaudi_eager['total_training_time']]
colors = ['#76b900', '#0071c5', '#ed1c24']

axes[0, 0].bar(devices, times, color=colors)
axes[0, 0].set_ylabel('Time (seconds)')
axes[0, 0].set_title('Total Training Time')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Memory usage comparison
memory = [cuda['test_metrics']['max_memory_gb'],
          gaudi_lazy['test_metrics']['max_memory_gb'],
          gaudi_eager['test_metrics']['max_memory_gb']]

axes[0, 1].bar(devices, memory, color=colors)
axes[0, 1].set_ylabel('Memory (GB)')
axes[0, 1].set_title('Peak Memory Usage')
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Accuracy metrics comparison
metrics = ['Accuracy', 'IoU', 'Precision', 'Recall', 'F1']
cuda_metrics = [cuda['test_metrics']['accuracy'],
                cuda['test_metrics']['mean_iou'],
                cuda['test_metrics']['mean_precision'],
                cuda['test_metrics']['mean_recall'],
                cuda['test_metrics']['mean_f1']]
gaudi_lazy_metrics = [gaudi_lazy['test_metrics']['accuracy'],
                       gaudi_lazy['test_metrics']['mean_iou'],
                       gaudi_lazy['test_metrics']['mean_precision'],
                       gaudi_lazy['test_metrics']['mean_recall'],
                       gaudi_lazy['test_metrics']['mean_f1']]
gaudi_eager_metrics = [gaudi_eager['test_metrics']['accuracy'],
                        gaudi_eager['test_metrics']['mean_iou'],
                        gaudi_eager['test_metrics']['mean_precision'],
                        gaudi_eager['test_metrics']['mean_recall'],
                        gaudi_eager['test_metrics']['mean_f1']]

x = np.arange(len(metrics))
width = 0.25

axes[1, 0].bar(x - width, cuda_metrics, width, label='CUDA', color=colors[0])
axes[1, 0].bar(x, gaudi_lazy_metrics, width, label='Gaudi Lazy', color=colors[1])
axes[1, 0].bar(x + width, gaudi_eager_metrics, width, label='Gaudi Eager', color=colors[2])
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Model Quality Metrics')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics, rotation=45)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Training loss over epochs
epochs = range(1, len(cuda['history']['train_loss']) + 1)

axes[1, 1].plot(epochs, cuda['history']['train_loss'], 
                marker='o', label='CUDA', color=colors[0])
axes[1, 1].plot(epochs, gaudi_lazy['history']['train_loss'], 
                marker='s', label='Gaudi Lazy', color=colors[1])
axes[1, 1].plot(epochs, gaudi_eager['history']['train_loss'], 
                marker='^', label='Gaudi Eager', color=colors[2])
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Training Loss Over Time')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'benchmark_comparison.png'")
```
""")

def main():
    """Main comparison function"""
    print("\n" + "="*70)
    print("  UNet Semantic Segmentation Benchmark - Results Analysis")
    print("="*70)
    
    # Load all results
    results = {
        'CUDA (A100)': load_results('results_cuda.json'),
        'Intel Gaudi (Lazy Mode)': load_results('results_gaudi_lazy.json'),
        'Intel Gaudi (Eager Mode)': load_results('results_gaudi_eager.json')
    }
    
    # Check which results are available
    available = [name for name, data in results.items() if data is not None]
    missing = [name for name, data in results.items() if data is None]
    
    print(f"\n✓ Found results for: {', '.join(available)}")
    if missing:
        print(f"✗ Missing results for: {', '.join(missing)}")
        print(f"\nNote: Run the corresponding training scripts to generate missing results.")
    
    if not available:
        print("\n❌ No result files found!")
        print("\nExpected files:")
        print("  - results_cuda.json")
        print("  - results_gaudi_lazy.json")
        print("  - results_gaudi_eager.json")
        print("\nRun the training scripts first:")
        print("  python3 unet_cuda.py")
        print("  python3 unet_gaudi_lazy.py")
        print("  python3 unet_gaudi_eager.py")
        sys.exit(1)
    
    # Run comparisons
    if len(available) >= 2:
        compare_performance(results)
        compare_accuracy(results)
        print_recommendations(results)
    
    print_detailed_metrics(results)
    
    if len(available) >= 2:
        generate_plot_commands(results)
    
    print("\n" + "="*70)
    print("  Analysis Complete")
    print("="*70)
    print()

if __name__ == '__main__':
    main()
