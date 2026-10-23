"""
Cross-dataset transfer learning module for QAgroNet.
Transfers pretrained MobileNetV2 + Variational Quantum Circuit weights
from PlantVillage (38 classes) to BPLD (5 classes).
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from config import cfg
from model import QAgroNet
from dataset import get_transforms, load_split_dataset
from train import train_one_epoch, evaluate, make_scaler

def load_matching_state(model, src_state):
    """
    Safely loads matching parameters from src_state into model.
    Filters out keys with mismatched tensor shapes (e.g. final linear layer with different num_classes).
    """
    tgt = model.state_dict()
    kept, dropped_shape, dropped_key = {}, [], []

    for k, v in src_state.items():
        if k not in tgt:
            dropped_key.append(k)
        elif tgt[k].shape != v.shape:
            dropped_shape.append((k, tuple(v.shape), tuple(tgt[k].shape)))
        else:
            kept[k] = v

    print(f"[Transfer] Loaded {len(kept)} matching parameter tensors.")
    if dropped_shape:
        print(f"[Transfer] Reinitialized due to class-count mismatch: {[x[0] for x in dropped_shape]}")
    model.load_state_dict(kept, strict=False)

def transfer_learning(src_ckpt_path, target_root=cfg.BPLD_ROOT, target_name='BPLD', epochs=10, lr=1.5e-4):
    """
    Fine-tunes QAgroNet on target dataset using pre-trained quantum and classical representations.
    """
    print(f"\n{'='*64}\n CROSS-DATASET TRANSFER LEARNING -> {target_name}\n{'='*64}")
    tr_tf, va_tf = get_transforms()
    tr_ld, va_ld, te_ld, class_names = load_split_dataset(target_root, tr_tf, va_tf)

    model = QAgroNet(num_classes=len(class_names), config=cfg).to(cfg.DEVICE)

    if os.path.exists(src_ckpt_path):
        print(f"[Transfer] Loading source checkpoint from: {src_ckpt_path}")
        ckpt = torch.load(src_ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        src_state = ckpt.get('model_state_dict', ckpt)
        load_matching_state(model, src_state)
    else:
        print(f"[Transfer] Source checkpoint '{src_ckpt_path}' not found. Training from ImageNet initialization.")

    # Freeze backbone and quantum circuit; fine-tune classifier head
    for name, param in model.named_parameters():
        param.requires_grad = ('classifier' in name)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg.WEIGHT_DECAY
    )
    scaler = make_scaler()

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, tr_ld, criterion, optimizer, scaler, cfg.DEVICE)
        va_loss, va_acc, _, _ = evaluate(model, va_ld, criterion, cfg.DEVICE)

        if va_acc > best_acc:
            best_acc = va_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"FT Epoch {ep:02d}/{epochs:02d} | Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

    model.load_state_dict(best_weights)
    _, te_acc, te_pred, te_true = evaluate(model, te_ld, criterion, cfg.DEVICE)

    print(f"\n{'='*64}")
    print(f" TRANSFER TEST ACCURACY on {target_name}: {te_acc:.4f}")
    print(f"{'='*64}")
    print(classification_report(te_true, te_pred, target_names=class_names, digits=4))

    save_path = os.path.join(cfg.CKPT_DIR, f"qagronet_transfer_{target_name.lower()}.pth")
    torch.save({
        'model_state_dict': best_weights,
        'class_names': class_names,
        'test_acc': te_acc,
        'source_checkpoint': src_ckpt_path
    }, save_path)
    print(f"[Transfer] Checkpoint saved to {save_path}")
    return te_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cross-Dataset Transfer Learning for QAgroNet")
    parser.add_argument('--source_ckpt', type=str, default='./checkpoints/qagronet_plantvillage.pth')
    parser.add_argument('--target_root', type=str, default=cfg.BPLD_ROOT)
    parser.add_argument('--target_name', type=str, default='BPLD')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    transfer_learning(
        src_ckpt_path=args.source_ckpt,
        target_root=args.target_root,
        target_name=args.target_name,
        epochs=args.epochs
    )
