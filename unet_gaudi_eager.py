"""
=============================================================
UNet Semantic Segmentation Training on Intel Gaudi (Eager Mode)
=============================================================
Enhanced version with comprehensive benchmarking metrics.
=============================================================
"""

import os
import sys
import time
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Habana imports
try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hthpu
    HABANA_AVAILABLE = True
except ImportError:
    HABANA_AVAILABLE = False
    print("ERROR: Habana frameworks not available!")
    sys.exit(1)

# =============================================================
# 1. Setup and Configuration
# =============================================================

def move_to_cpu(obj):
    """Recursively move all tensors in nested structure to CPU"""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    else:
        return obj

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HABANA_AVAILABLE:
        torch.hpu.manual_seed_all(seed)

set_seed(42)

# Enable Eager Mode
if HABANA_AVAILABLE:
    os.environ["PT_HPU_LAZY_MODE"] = "0"

# Device configuration
if HABANA_AVAILABLE and torch.hpu.is_available():
    device = torch.device("hpu")
    print(f"\n{'='*60}")
    print(f"Device: HPU (Intel Gaudi - Eager Mode)")
    print(f"PyTorch: {torch.__version__}")
    print(f"{'='*60}\n")
else:
    print("ERROR: Habana Gaudi not available!")
    sys.exit(1)

# Configuration
CONFIG = {
    'image_height': 384,
    'image_width': 512,
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'num_workers': 8,
    'num_classes': 20,
    'device_name': 'Gaudi_Eager'
}

# Load dataset splits
if not Path('dataset_splits.json').exists():
    print("ERROR: dataset_splits.json not found!")
    print("Run: python3 prepare_dataset.py")
    sys.exit(1)

with open('dataset_splits.json', 'r') as f:
    DATASET_INFO = json.load(f)

CLASSES = DATASET_INFO['classes']

# =============================================================
# 2. Dataset (Same as CUDA)
# =============================================================

class CARLADataset(Dataset):
    def __init__(self, image_paths, mask_paths, classes):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.classes = classes
    
    def __len__(self):
        return len(self.image_paths)
    
    def transform_mask(self, mask):
        masks = []
        for color in self.classes:
            m = (mask == color).all(axis=-1)
            masks.append(m)
        return np.stack(masks, axis=-1).astype(np.float32)
    
    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert('RGB')).astype(np.float32) / 255.0
        
        mask = np.array(Image.open(self.mask_paths[idx]).convert('RGB'))
        mask = self.transform_mask(mask)
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        
        return img, mask

# =============================================================
# 3. UNet Model (Same as CUDA)
# =============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.out = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.out(d1)

# =============================================================
# 4. Performance Metrics (Same as CUDA)
# =============================================================

class PerformanceMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.batch_times = []
        self.iteration_times = []
        self.memory_allocated = []
        self.memory_reserved = []
    
    def update(self, y_true, y_pred, batch_time, iter_time, mem_alloc, mem_res):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        for i in range(y_true.shape[0]):
            for t, p in zip(y_true[i].flatten(), y_pred[i].flatten()):
                self.confusion_matrix[int(t), int(p)] += 1
        
        self.batch_times.append(batch_time)
        self.iteration_times.append(iter_time)
        self.memory_allocated.append(mem_alloc)
        self.memory_reserved.append(mem_res)
    
    def compute_metrics(self):
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        precision = np.divide(tp, tp + fp, where=(tp + fp) != 0, out=np.zeros_like(tp, dtype=float))
        recall = np.divide(tp, tp + fn, where=(tp + fn) != 0, out=np.zeros_like(tp, dtype=float))
        f1 = np.divide(2 * precision * recall, precision + recall, 
                       where=(precision + recall) != 0, out=np.zeros_like(precision, dtype=float))
        
        accuracy = tp.sum() / cm.sum()
        iou = np.divide(tp, tp + fp + fn, where=(tp + fp + fn) != 0, out=np.zeros_like(tp, dtype=float))
        
        return {
            'accuracy': float(accuracy),
            'mean_precision': float(precision.mean()),
            'mean_recall': float(recall.mean()),
            'mean_f1': float(f1.mean()),
            'mean_iou': float(iou.mean()),
            'per_class_iou': iou.tolist(),
            'avg_batch_time': float(np.mean(self.batch_times)) if self.batch_times else 0,
            'min_batch_time': float(np.min(self.batch_times)) if self.batch_times else 0,
            'max_batch_time': float(np.max(self.batch_times)) if self.batch_times else 0,
            'avg_iteration_time': float(np.mean(self.iteration_times)) if self.iteration_times else 0,
            'throughput_samples_per_sec': CONFIG['batch_size'] / np.mean(self.batch_times) if self.batch_times else 0,
            'peak_memory_allocated_gb': float(np.max(self.memory_allocated)) if self.memory_allocated else 0,
            'peak_memory_reserved_gb': float(np.max(self.memory_reserved)) if self.memory_reserved else 0,
            'avg_memory_allocated_gb': float(np.mean(self.memory_allocated)) if self.memory_allocated else 0,
        }

# =============================================================
# 5. Training & Validation (Modified for Gaudi Eager Mode)
# =============================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    epoch_loss = 0
    epoch_start = time.time()
    metrics = PerformanceMetrics(CONFIG['num_classes'])
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [TRAIN]")
    
    for batch_idx, (images, masks) in enumerate(pbar):
        iter_start = time.time()
        
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # EAGER MODE: No mark_step needed (immediate execution)
        
        batch_time = time.time() - iter_start
        iter_time = batch_time
        
        mem_alloc = torch.hpu.memory_allocated() / 1024**3 if HABANA_AVAILABLE else 0
        mem_res = torch.hpu.memory_reserved() / 1024**3 if HABANA_AVAILABLE else 0
        
        with torch.no_grad():
            pred = torch.argmax(outputs, dim=1)
            true = torch.argmax(masks, dim=1)
            metrics.update(true, pred, batch_time, iter_time, mem_alloc, mem_res)
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{epoch_loss / (batch_idx + 1):.4f}', 'mem': f'{mem_alloc:.2f}GB'})
    
    epoch_metrics = metrics.compute_metrics()
    epoch_metrics['epoch_time'] = time.time() - epoch_start
    epoch_metrics['avg_loss'] = epoch_loss / len(loader)
    return epoch_metrics

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    metrics = PerformanceMetrics(CONFIG['num_classes'])
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            batch_start = time.time()
            
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # EAGER MODE: No mark_step needed
            
            batch_time = time.time() - batch_start
            mem_alloc = torch.hpu.memory_allocated() / 1024**3 if HABANA_AVAILABLE else 0
            mem_res = torch.hpu.memory_reserved() / 1024**3 if HABANA_AVAILABLE else 0
            
            pred = torch.argmax(outputs, dim=1)
            true = torch.argmax(masks, dim=1)
            metrics.update(true, pred, batch_time, batch_time, mem_alloc, mem_res)
            
            val_loss += loss.item()
    
    val_metrics = metrics.compute_metrics()
    val_metrics['avg_loss'] = val_loss / len(loader)
    return val_metrics

# =============================================================
# 6. Main (Same structure as CUDA)
# =============================================================

def main():
    print(f"\n{'='*60}")
    print("UNet Training - Intel Gaudi (Eager Mode)")
    print(f"{'='*60}\n")
    
    # Load data
    train_dataset = CARLADataset(
        DATASET_INFO['splits']['train']['images'],
        DATASET_INFO['splits']['train']['masks'],
        CLASSES
    )
    val_dataset = CARLADataset(
        DATASET_INFO['splits']['val']['images'],
        DATASET_INFO['splits']['val']['masks'],
        CLASSES
    )
    test_dataset = CARLADataset(
        DATASET_INFO['splits']['test']['images'],
        DATASET_INFO['splits']['test']['masks'],
        CLASSES
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=False)
    
    # Model
    model = UNet(3, CONFIG['num_classes']).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    if HABANA_AVAILABLE:
        model_memory = torch.hpu.memory_allocated() / 1024**3
        print(f"Model Memory: {model_memory:.3f} GB\n")
    else:
        model_memory = 0
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    print(f"{'='*60}")
    print("Training Started")
    print(f"{'='*60}\n")
    
    training_start = time.time()
    best_val_loss = float('inf')
    best_val_iou = 0.0
    best_train_loss = float('inf')
    best_train_iou = 0.0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': [],
        'epoch_times': [], 'learning_rates': []
    }
    
    for epoch in range(CONFIG['num_epochs']):
        lr = optimizer.param_groups[0]['lr']
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, CONFIG['num_epochs'])
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_metrics['avg_loss'])
        
        # Track best
        best_train_loss = min(best_train_loss, train_metrics['avg_loss'])
        best_train_iou = max(best_train_iou, train_metrics['mean_iou'])
        best_val_loss = min(best_val_loss, val_metrics['avg_loss'])
        best_val_iou = max(best_val_iou, val_metrics['mean_iou'])
        
        # History
        history['train_loss'].append(train_metrics['avg_loss'])
        history['val_loss'].append(val_metrics['avg_loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_iou'].append(train_metrics['mean_iou'])
        history['val_iou'].append(val_metrics['mean_iou'])
        history['train_f1'].append(train_metrics['mean_f1'])
        history['val_f1'].append(val_metrics['mean_f1'])
        history['epoch_times'].append(train_metrics['epoch_time'])
        history['learning_rates'].append(lr)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'='*60}")
        print(f"Loss:       Train={train_metrics['avg_loss']:.4f}  Val={val_metrics['avg_loss']:.4f}")
        print(f"Accuracy:   Train={train_metrics['accuracy']:.4f}  Val={val_metrics['accuracy']:.4f}")
        print(f"IoU:        Train={train_metrics['mean_iou']:.4f}  Val={val_metrics['mean_iou']:.4f}")
        print(f"F1:         Train={train_metrics['mean_f1']:.4f}  Val={val_metrics['mean_f1']:.4f}")
        print(f"Time:       {train_metrics['epoch_time']:.2f}s")
        print(f"Batch:      {train_metrics['avg_batch_time']:.4f}s")
        print(f"Throughput: {train_metrics['throughput_samples_per_sec']:.2f} samples/s")
        print(f"Memory:     {train_metrics['peak_memory_allocated_gb']:.2f} GB")
        print(f"LR:         {lr:.6f}")

        # Save best
        if val_metrics['mean_iou'] == best_val_iou:
            # Recursively move all tensors to CPU to avoid HPU tensor operation errors
            model_state = move_to_cpu(model.state_dict())
            optimizer_state = move_to_cpu(optimizer.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'val_loss': val_metrics['avg_loss'],
                'val_iou': val_metrics['mean_iou'],
            }, 'best_unet_gaudi_eager.pth')
            print("✓ Saved best model")
    
    total_training_time = time.time() - training_start
    
    # Test
    print(f"\n{'='*60}")
    print("Test Evaluation")
    print(f"{'='*60}\n")
    model.load_state_dict(torch.load('best_unet_gaudi_eager.pth')['model_state_dict'])
    test_metrics = validate(model, test_loader, criterion, device)
    
    # Final Results
    print(f"\n{'='*60}")
    print("FINAL RESULTS - Intel Gaudi (Eager Mode)")
    print(f"{'='*60}")
    print(f"\nTiming:")
    print(f"  Total Training:  {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
    print(f"  Avg Epoch:       {np.mean(history['epoch_times']):.2f}s")
    print(f"  Avg Batch:       {train_metrics['avg_batch_time']:.4f}s")
    print(f"  Min Batch:       {train_metrics['min_batch_time']:.4f}s")
    print(f"  Max Batch:       {train_metrics['max_batch_time']:.4f}s")
    print(f"  Throughput:      {train_metrics['throughput_samples_per_sec']:.2f} samples/s")
    
    print(f"\nBest Training:")
    print(f"  Loss: {best_train_loss:.4f}")
    print(f"  IoU:  {best_train_iou:.4f}")
    
    print(f"\nBest Validation:")
    print(f"  Loss: {best_val_loss:.4f}")
    print(f"  IoU:  {best_val_iou:.4f}")
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  IoU:       {test_metrics['mean_iou']:.4f}")
    print(f"  Precision: {test_metrics['mean_precision']:.4f}")
    print(f"  Recall:    {test_metrics['mean_recall']:.4f}")
    print(f"  F1:        {test_metrics['mean_f1']:.4f}")
    
    print(f"\nMemory:")
    print(f"  Model:         {model_memory:.3f} GB")
    print(f"  Peak Alloc:    {test_metrics['peak_memory_allocated_gb']:.2f} GB")
    print(f"  Peak Reserved: {test_metrics['peak_memory_reserved_gb']:.2f} GB")
    print(f"  Avg Alloc:     {test_metrics['avg_memory_allocated_gb']:.2f} GB")
    
    # Save results
    results = {
        'device': 'Intel Gaudi (Eager Mode)',
        'pytorch_version': torch.__version__,
        'config': CONFIG,
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_memory_gb': model_memory,
        },
        'timing': {
            'total_training_time_sec': total_training_time,
            'total_training_time_min': total_training_time / 60,
            'avg_epoch_time_sec': float(np.mean(history['epoch_times'])),
            'avg_batch_time_sec': train_metrics['avg_batch_time'],
            'min_batch_time_sec': train_metrics['min_batch_time'],
            'max_batch_time_sec': train_metrics['max_batch_time'],
            'avg_iteration_time_sec': train_metrics['avg_iteration_time'],
            'throughput_samples_per_sec': train_metrics['throughput_samples_per_sec'],
        },
        'memory': {
            'model_memory_gb': model_memory,
            'peak_memory_allocated_gb': test_metrics['peak_memory_allocated_gb'],
            'peak_memory_reserved_gb': test_metrics['peak_memory_reserved_gb'],
            'avg_memory_allocated_gb': test_metrics['avg_memory_allocated_gb'],
        },
        'best_metrics': {
            'train': {'loss': best_train_loss, 'iou': best_train_iou},
            'validation': {'loss': best_val_loss, 'iou': best_val_iou}
        },
        'test_metrics': test_metrics,
        'history': history,
        'dataset_info': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset)
        }
    }
    
    with open('results_gaudi_eager.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to results_gaudi_eager.json")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
