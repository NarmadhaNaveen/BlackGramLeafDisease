"""
Training engine for QAgroNet with Automatic Mixed Precision (AMP),
differential learning rate for quantum gates, and Cosine Annealing.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import copy
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from config import cfg, _AMP_DEVICE, _AMP_ENABLED, _AMP_DTYPE
from model import QAgroNet, self_test
from dataset import get_transforms, load_split_dataset

def make_scaler():
    """Returns GradScaler configured for CUDA AMP or no-op on CPU."""
    return GradScaler(_AMP_DEVICE, enabled=_AMP_ENABLED)

def build_optimizer(model, lr=cfg.LR, q_lr_mult=cfg.Q_LR_MULT, weight_decay=cfg.WEIGHT_DECAY):
    """
    Differential learning rate:
    Assigns higher learning rate multiplier to quantum weights (RY/RZ gate angles)
    so the variational circuit learns effectively alongside classical weights.
    """
    quantum_params = []
    classical_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'q_weights' in name:
            quantum_params.append(param)
        else:
            classical_params.append(param)

    return optim.AdamW([
        {'params': classical_params, 'lr': lr},
        {'params': quantum_params,   'lr': lr * q_lr_mult}
    ], weight_decay=weight_decay)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Executes one epoch of training with gradient accumulation & AMP."""
    model.train()
    total_loss, n_seen = 0.0, 0
    preds, labels_ = [], []
    accum = cfg.GRAD_ACCUM_STEPS
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc='Train', leave=False, ncols=100)
    for step, (imgs, labels) in enumerate(pbar):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(_AMP_DEVICE, enabled=_AMP_ENABLED, dtype=_AMP_DTYPE):
            logits = model(imgs)
            loss   = criterion(logits, labels) / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_sz = imgs.size(0)
        total_loss += loss.item() * accum * batch_sz
        n_seen     += batch_sz
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        labels_.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item() * accum:.4f}")

    return total_loss / n_seen, accuracy_score(labels_, preds)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluates the model on validation or test sets."""
    model.eval()
    total_loss, n_seen = 0.0, 0
    preds, labels_ = [], []
    for imgs, labels in tqdm(loader, desc='Eval ', leave=False, ncols=100):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(_AMP_DEVICE, enabled=_AMP_ENABLED, dtype=_AMP_DTYPE):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        batch_sz = imgs.size(0)
        total_loss += loss.item() * batch_sz
        n_seen     += batch_sz
        preds.extend(logits.argmax(1).cpu().numpy())
        labels_.extend(labels.cpu().numpy())

    return (total_loss / n_seen, accuracy_score(labels_, preds), preds, labels_)

def train(dataset_name='BPLD', data_root=cfg.BPLD_ROOT, epochs=cfg.EPOCHS, lr=cfg.LR, batch_size=cfg.BATCH_SIZE):
    """Full training pipeline for QAgroNet."""
    print(f"\n{'='*64}\n Training QAgroNet on {dataset_name}\n{'='*64}")
    tr_tf, va_tf = get_transforms()
    tr_ld, va_ld, te_ld, class_names = load_split_dataset(data_root, tr_tf, va_tf, batch_size=batch_size)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Dataset split: Train={len(tr_ld.dataset)} | Val={len(va_ld.dataset)} | Test={len(te_ld.dataset)}")

    model     = QAgroNet(num_classes=len(class_names), config=cfg).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    optimizer = build_optimizer(model, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = make_scaler()

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, tr_ld, criterion, optimizer, scaler, cfg.DEVICE)
        va_loss, va_acc, _, _ = evaluate(model, va_ld, criterion, cfg.DEVICE)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            best_weights = copy.deepcopy(model.state_dict())

        vram = (torch.cuda.max_memory_allocated() / 1e9 if cfg.DEVICE.type == 'cuda' else 0.0)
        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} "
              f"| Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | Time: {time.time()-t0:.1f}s | Peak VRAM: {vram:.2f}GB")

    # Final evaluation on held-out test split
    model.load_state_dict(best_weights)
    te_loss, te_acc, te_pred, te_true = evaluate(model, te_ld, criterion, cfg.DEVICE)

    print(f"\n{'='*64}")
    print(f" FINAL TEST EVALUATION on {dataset_name}: Accuracy = {te_acc:.4f}")
    print(f"{'='*64}")
    print(classification_report(te_true, te_pred, target_names=class_names, digits=4))

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    ckpt_file = os.path.join(cfg.CKPT_DIR, f"qagronet_{dataset_name.lower()}.pth")
    torch.save({
        'model_state_dict': best_weights,
        'class_names': class_names,
        'history': history,
        'test_acc': te_acc,
        'config': {
            'n_qubits': cfg.N_QUBITS,
            'n_layers': cfg.N_LAYERS,
            'embed_dim': cfg.EMBED_DIM
        }
    }, ckpt_file)
    print(f"[Checkpoint] Model saved to {ckpt_file}")
    return model, history, te_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train QAgroNet")
    parser.add_argument('--dataset', type=str, default='BPLD', choices=['BPLD', 'PlantVillage'], help="Dataset name")
    parser.add_argument('--data_root', type=str, default=cfg.BPLD_ROOT, help="Path to dataset root directory")
    parser.add_argument('--epochs', type=int, default=cfg.EPOCHS, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=cfg.LR, help="Base learning rate")
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train(
        dataset_name=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )
