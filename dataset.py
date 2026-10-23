"""
Dataset preparation and DataLoader utilities for QAgroNet.
Supports Blackgram Plant Leaf Disease (BPLD) and PlantVillage datasets.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from config import cfg

# Official Blackgram disease classes as referenced in the paper
BPLD_CLASSES = [
    "Anthracnose",
    "Leaf_Crinkle",
    "Powdery_Mildew",
    "Yellow_Mosaic",
    "Healthy"
]

def get_transforms():
    """
    Standard data transformations:
    - Train: Random horizontal flip, random rotation (15 deg), color jitter, normalization
    - Val/Test: Resize (224x224), normalization
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf

def create_synthetic_dataset(root_dir, class_names=BPLD_CLASSES, images_per_class=20):
    """Generates synthetic leaf images to allow immediate execution and validation."""
    os.makedirs(root_dir, exist_ok=True)
    for c in class_names:
        cdir = os.path.join(root_dir, c)
        os.makedirs(cdir, exist_ok=True)
        for i in range(images_per_class):
            img_path = os.path.join(cdir, f"sample_{i:03d}.png")
            if not os.path.exists(img_path):
                # Generate realistic-looking colored texture (leaf green/brown tones)
                base = np.zeros((224, 224, 3), dtype=np.uint8)
                base[:, :, 1] = np.random.randint(90, 200, (224, 224)) # Green channel
                base[:, :, 0] = np.random.randint(30, 100, (224, 224)) # Red channel
                base[:, :, 2] = np.random.randint(20, 80, (224, 224))  # Blue channel
                Image.fromarray(base).save(img_path)
    print(f"[Dataset] Generated {len(class_names) * images_per_class} sample images in {root_dir}")

def load_split_dataset(root_dir, train_tf, val_tf, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS):
    """
    Loads dataset and splits into Train (70%), Val (15%), Test (15%) using stratified sampling.
    Asserts every class is present in each split.
    """
    if not os.path.exists(root_dir) or len(os.listdir(root_dir)) == 0:
        print(f"[Dataset] Notice: '{root_dir}' not found or empty. Creating initial synthetic data...")
        create_synthetic_dataset(root_dir)

    full = datasets.ImageFolder(root=root_dir)
    targets = np.array([t for _, t in full.samples])
    idx = np.arange(len(full))

    # Stratified split: Train (70%) vs Val+Test (30%)
    tr_idx, tmp_idx = train_test_split(
        idx,
        test_size=(cfg.VAL_RATIO + cfg.TEST_RATIO),
        stratify=targets,
        random_state=cfg.SEED
    )

    # Split remaining 30% equally into Val (15%) and Test (15%)
    va_idx, te_idx = train_test_split(
        tmp_idx,
        test_size=0.5,
        stratify=targets[tmp_idx],
        random_state=cfg.SEED
    )

    # Class presence verification
    for name, split in [('Train', tr_idx), ('Val', va_idx), ('Test', te_idx)]:
        present = set(targets[split].tolist())
        missing = set(range(len(full.classes))) - present
        assert not missing, f"{name} split missing classes: {missing}"

    tr_ds = Subset(datasets.ImageFolder(root_dir, transform=train_tf), tr_idx.tolist())
    va_ds = Subset(datasets.ImageFolder(root_dir, transform=val_tf), va_idx.tolist())
    te_ds = Subset(datasets.ImageFolder(root_dir, transform=val_tf), te_idx.tolist())

    pin_mem = (cfg.DEVICE.type == 'cuda')
    kw = dict(num_workers=num_workers, pin_memory=pin_mem)

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, **kw)
    val_loader   = DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, full.classes
