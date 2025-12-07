"""
=============================================================
CARLA Dataset Preparation Script
=============================================================
Downloads and prepares the CARLA Vision Dataset with random crops.
Saves dataset split information for consistent training across all platforms.
=============================================================
"""

import os
import sys
import json
import random
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import zipfile
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

# Configuration
CONFIG = {
    'image_height': 384,
    'image_width': 512,
    'crops_per_img': 10,
    'data_dir': './carla_data',
    'num_classes': 20,
    'test_size': 0.3,
    'val_size': 0.5,  # Of the test_size portion
    'random_seed': 42
}

# Class definitions for CARLA dataset
CLASSES = [
    [0, 0, 142], [45, 60, 150], [70, 70, 70], [70, 130, 180], [81, 0, 81],
    [100, 40, 40], [102, 102, 156], [107, 142, 35], [110, 190, 160], [128, 64, 128],
    [145, 170, 100], [150, 100, 100], [153, 153, 153], [157, 234, 50], [170, 120, 50],
    [220, 20, 60], [220, 220, 0], [230, 150, 140], [244, 35, 232], [250, 170, 30]
]

def download_kaggle_dataset():
    """Download and extract CARLA dataset from Kaggle"""
    print("\n" + "="*60)
    print("Downloading CARLA Dataset from Kaggle")
    print("="*60)
    
    data_dir = Path(CONFIG['data_dir'])
    if data_dir.exists() and (data_dir / 'images').exists() and (data_dir / 'masks').exists():
        print("✓ Dataset already exists, skipping download.")
        return
    
    # Setup Kaggle credentials
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    if not kaggle_json.exists():
        print("\n❌ Kaggle credentials not found!")
        print("Please ensure kaggle.json exists in ~/.kaggle/")
        print("Download from: https://www.kaggle.com/settings")
        sys.exit(1)
    
    os.chmod(kaggle_json, 0o600)
    
    # Download dataset using kagglehub
    print("Downloading dataset...")
    try:
        import kagglehub
        download_path = kagglehub.dataset_download("maelstro/carla-vd-dataset")
        print(f"✓ Dataset downloaded to: {download_path}")
        
        # Organize files from kagglehub cache to our structure
        data_dir.mkdir(exist_ok=True)
        download_path = Path(download_path)
        
        # Copy/move files to our data directory structure
        images_src = download_path / '_out'
        masks_src = download_path / '_out_seg_city'
        
        if images_src.exists():
            import shutil
            print("Copying images...")
            if (data_dir / 'images').exists():
                shutil.rmtree(data_dir / 'images')
            shutil.copytree(images_src, data_dir / 'images')
        else:
            print("❌ '_out' directory not found in downloaded dataset!")
            sys.exit(1)
        
        if masks_src.exists():
            import shutil
            print("Copying masks...")
            if (data_dir / 'masks').exists():
                shutil.rmtree(data_dir / 'masks')
            shutil.copytree(masks_src, data_dir / 'masks')
        else:
            print("❌ '_out_seg_city' directory not found in downloaded dataset!")
            sys.exit(1)
        
        print("✓ Dataset organized successfully!")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        sys.exit(1)
    
    print("✓ Dataset downloaded and extracted successfully!")

def random_crops(img_dir, mask_dir, crops_per_img, crop_h, crop_w, output_dir):
    """Generate random crops from the original images"""
    print("\n" + "="*60)
    print(f"Generating Random Crops ({crop_h}x{crop_w})")
    print("="*60)
    
    output_dir = Path(output_dir)
    if output_dir.exists() and len(list((output_dir / 'images').glob('*.png'))) > 0:
        print("✓ Cropped images already exist, skipping...")
        return
    
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'masks').mkdir(exist_ok=True)
    
    img_files = sorted(list(img_dir.glob('*.png')))
    mask_files = sorted(list(mask_dir.glob('*.png')))
    
    print(f"Processing {len(img_files)} images with {crops_per_img} crops each...")
    
    for i, (img_path, mask_path) in enumerate(tqdm(zip(img_files, mask_files), total=len(img_files), desc="Cropping")):
        img = np.array(Image.open(str(img_path)))
        mask = np.array(Image.open(str(mask_path)))
        
        img_h, img_w, _ = img.shape
        
        for j in range(crops_per_img):
            x1 = random.randint(0, img_w - crop_w - 1)
            y1 = random.randint(0, img_h - crop_h - 1)
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            
            img_crop = img[y1:y2, x1:x2]
            mask_crop = mask[y1:y2, x1:x2]
            
            Image.fromarray(img_crop).save(str(output_dir / 'images' / f'img_{i}_{j}.png'))
            Image.fromarray(mask_crop).save(str(output_dir / 'masks' / f'mask_{i}_{j}.png'))
    
    print(f"✓ Generated {len(img_files) * crops_per_img} cropped images!")

def create_dataset_splits(crop_dir):
    """Create train/val/test splits and save to JSON"""
    print("\n" + "="*60)
    print("Creating Dataset Splits")
    print("="*60)
    
    img_files = sorted(list((crop_dir / 'images').glob('*.png')))
    mask_files = sorted(list((crop_dir / 'masks').glob('*.png')))
    
    img_paths = [str(f) for f in img_files]
    mask_paths = [str(f) for f in mask_files]
    
    # Train/Val/Test split
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        img_paths, mask_paths, test_size=CONFIG['test_size'], random_state=CONFIG['random_seed']
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks, test_size=CONFIG['val_size'], random_state=CONFIG['random_seed']
    )
    
    dataset_info = {
        'config': CONFIG,
        'classes': CLASSES,
        'splits': {
            'train': {
                'images': train_imgs,
                'masks': train_masks,
                'count': len(train_imgs)
            },
            'val': {
                'images': val_imgs,
                'masks': val_masks,
                'count': len(val_imgs)
            },
            'test': {
                'images': test_imgs,
                'masks': test_masks,
                'count': len(test_imgs)
            }
        },
        'total_samples': len(img_paths)
    }
    
    # Save to JSON
    with open('dataset_splits.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✓ Dataset splits created:")
    print(f"  Train: {len(train_imgs)} samples")
    print(f"  Val:   {len(val_imgs)} samples")
    print(f"  Test:  {len(test_imgs)} samples")
    print(f"  Total: {len(img_paths)} samples")
    print(f"\n✓ Dataset split info saved to: dataset_splits.json")

def main():
    print("\n" + "="*60)
    print("CARLA Dataset Preparation for UNet Benchmarking")
    print("="*60)
    
    # Download dataset
    download_kaggle_dataset()
    
    # Generate crops
    data_dir = Path(CONFIG['data_dir'])
    crop_dir = Path(f"cropped_{CONFIG['image_height']}x{CONFIG['image_width']}")
    
    random_crops(
        data_dir / 'images',
        data_dir / 'masks',
        CONFIG['crops_per_img'],
        CONFIG['image_height'],
        CONFIG['image_width'],
        crop_dir
    )
    
    # Create and save dataset splits
    create_dataset_splits(crop_dir)
    
    print("\n" + "="*60)
    print("✓ Dataset Preparation Complete!")
    print("="*60)
    print("\nYou can now run the training scripts.")
    print("They will automatically load the dataset splits from dataset_splits.json")

if __name__ == '__main__':
    main()
