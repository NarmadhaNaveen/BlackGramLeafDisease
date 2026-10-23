# ==============================================================
# QAgroNet — two-stage paper-faithful recipe + CLI stages
#   --stage all          : PlantVillage pretrain → BPLD transfer
#   --stage pv-only      : only PlantVillage pretrain
#   --stage bpld-direct  : only BPLD, full unfreeze (no PV)
#   --stage smoke        : self-test + smoke test only
# ==============================================================
import os, sys, copy, time, math, argparse
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import pennylane as qml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---- Windows-safe stdout (avoid charmap UnicodeEncodeError) ----
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# --------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------
class Config:
    # Quantum
    N_QUBITS    = 8
    N_LAYERS    = 4
    EMBED_DIM   = 8

    # Classical
    FEATURE_DIM = 1280

    # Batch / loaders
    BATCH_SIZE       = 64
    GRAD_ACCUM_STEPS = 1
    NUM_WORKERS      = 0     # Windows-safe (no worker spawn deadlocks)

    # Two-stage schedule
    EPOCHS      = 30         # overridden per stage
    LR          = 3e-4       # overridden per stage

    PV_EPOCHS   = 30
    PV_LR       = 3e-4

    BPLD_EPOCHS = 60
    BPLD_LR     = 1e-4

    BPLD_DIRECT_EPOCHS = 80  # longer since it starts cold
    BPLD_DIRECT_LR     = 3e-4

    # Optimization
    Q_LR_MULT     = 5.0
    WEIGHT_DECAY  = 1e-5
    LABEL_SMOOTH  = 0.0
    SEED          = 42

    USE_AMP       = True
    FREEZE_FRAC   = 0.0

    TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

    BPLD_ROOT         = './data/BPLD'
    PLANTVILLAGE_ROOT = './data/PlantVillage'
    CKPT_DIR          = './checkpoints'

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

# Reproducibility + speed
torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')

# Device-aware AMP
_AMP_DEVICE  = cfg.DEVICE.type
_AMP_ENABLED = bool(cfg.USE_AMP and _AMP_DEVICE == 'cuda')
_AMP_DTYPE   = torch.float16 if _AMP_ENABLED else torch.float32

print(f"Device: {cfg.DEVICE}")
if cfg.DEVICE.type == 'cuda':
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | VRAM: {p.total_memory/1e9:.1f} GB")
print(f"AMP: device={_AMP_DEVICE} enabled={_AMP_ENABLED} dtype={_AMP_DTYPE}")
print(f"NUM_WORKERS={cfg.NUM_WORKERS}")

# --------------------------------------------------------------
# 2. QUANTUM
# --------------------------------------------------------------
QDEV = qml.device("default.qubit", wires=cfg.N_QUBITS)

def _cnot_ring(wires):
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])

@qml.qnode(QDEV, interface='torch', diff_method='backprop')
def quantum_circuit_single(inputs, weights):
    wires = list(range(cfg.N_QUBITS))
    qml.AngleEmbedding(inputs, wires=wires, rotation='Y')
    for layer in range(cfg.N_LAYERS):
        for i, w in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=w)
            qml.RZ(weights[layer, i, 1], wires=w)
        _cnot_ring(wires)
    return [qml.expval(qml.PauliZ(w)) for w in wires]

def quantum_forward(z_batch, weights):
    """
    z_batch: (B, N_QUBITS) fp32, differentiable.
    Device-safe: transfers to CPU for PennyLane simulation, then transfers
    results back to original device. PyTorch autograd tracks graph across .cpu() and .to().
    Returns: (B, N_QUBITS) with dtype forced back to fp32 (PennyLane returns fp64).
    """
    orig_device = z_batch.device
    z_cpu = z_batch.cpu()
    w_cpu = weights.cpu()
    outs = [torch.stack(quantum_circuit_single(z_cpu[i], w_cpu))
            for i in range(z_cpu.size(0))]
    out = torch.stack(outs, dim=0)
    return out.to(device=orig_device, dtype=z_batch.dtype)

# --------------------------------------------------------------
# 3. MODEL
# --------------------------------------------------------------
class QAgroNet(nn.Module):
    def __init__(self, num_classes, cfg=cfg):
        super().__init__()
        self.cfg = cfg

        bb = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = bb.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)

        if cfg.FREEZE_FRAC > 0:
            n = len(self.backbone)
            n_frz = int(n * cfg.FREEZE_FRAC)
            for i, blk in enumerate(self.backbone):
                if i < n_frz:
                    for p in blk.parameters():
                        p.requires_grad = False

        self.compress = nn.Sequential(
            nn.Linear(cfg.FEATURE_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, cfg.EMBED_DIM),
            nn.Tanh(),
        )
        self.register_buffer('angle_scale', torch.tensor(math.pi / 2))

        self.q_weights = nn.Parameter(
            torch.randn(cfg.N_LAYERS, cfg.N_QUBITS, 2) * 0.1
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.N_QUBITS, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.avgpool(f).flatten(1)
        z = self.compress(f) * self.angle_scale
        z_fp32 = z.float()
        q_out = quantum_forward(z_fp32, self.q_weights)
        return self.classifier(q_out)

# --------------------------------------------------------------
# 4. SELF-TEST
# --------------------------------------------------------------
def self_test():
    print("\n" + "=" * 62)
    print(" SELF-TEST — gradient flow through every submodule")
    print("=" * 62)
    m = QAgroNet(num_classes=5, cfg=cfg).to(cfg.DEVICE)
    x = torch.randn(4, 3, 224, 224, device=cfg.DEVICE)
    y = torch.tensor([0, 1, 2, 3], device=cfg.DEVICE)
    logits = m(x)
    nn.CrossEntropyLoss()(logits, y).backward()

    def has_grad(params):
        return any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in params if p.requires_grad)

    checks = {
        'backbone (unfrozen half)': has_grad(m.backbone.parameters()),
        'compressor (1280->8)'     : has_grad(m.compress.parameters()),
        'quantum weights'         : (m.q_weights.grad is not None
                                     and m.q_weights.grad.abs().sum().item() > 0),
        'classifier head'         : has_grad(m.classifier.parameters()),
    }
    for k, ok in checks.items():
        print(f"  [{'OK ' if ok else 'FAIL'}] grad reaches {k}")

    z_probe = torch.randn(2, cfg.EMBED_DIM, device=cfg.DEVICE,
                          requires_grad=True)
    quantum_forward(z_probe, m.q_weights).sum().backward()
    z_ok = z_probe.grad is not None and z_probe.grad.abs().sum().item() > 0
    print(f"  [{'OK ' if z_ok else 'FAIL'}] grad reaches quantum INPUT (z)")

    all_ok = all(checks.values()) and z_ok
    print("=" * 62)
    print(" SELF-TEST: " + ("PASSED ✓" if all_ok else "FAILED ✗"))
    print("=" * 62 + "\n")
    if not all_ok:
        raise RuntimeError("Self-test failed — do not proceed to training")
    return m

# --------------------------------------------------------------
# 5. DATA
# --------------------------------------------------------------
def get_transforms():
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf

def load_split_dataset(root_dir, train_tf, val_tf):
    # Support case-insensitive directory resolution
    if not os.path.exists(root_dir):
        parent = os.path.dirname(root_dir) or '.'
        base = os.path.basename(root_dir).lower()
        if os.path.exists(parent):
            for candidate in os.listdir(parent):
                if candidate.lower() == base:
                    root_dir = os.path.join(parent, candidate)
                    break

    full    = datasets.ImageFolder(root=root_dir)
    targets = np.array([t for _, t in full.samples])
    idx     = np.arange(len(full))

    tr_idx, tmp_idx = train_test_split(
        idx, test_size=(cfg.VAL_RATIO + cfg.TEST_RATIO),
        stratify=targets, random_state=cfg.SEED)
    va_idx, te_idx = train_test_split(
        tmp_idx, test_size=0.5,
        stratify=targets[tmp_idx], random_state=cfg.SEED)

    for name, split in [('train', tr_idx), ('val', va_idx), ('test', te_idx)]:
        present = set(targets[split].tolist())
        missing = set(range(len(full.classes))) - present
        assert not missing, f"{name} split missing classes: {missing}"

    tr_ds = Subset(datasets.ImageFolder(root_dir, transform=train_tf),
                   tr_idx.tolist())
    va_ds = Subset(datasets.ImageFolder(root_dir, transform=val_tf),
                   va_idx.tolist())
    te_ds = Subset(datasets.ImageFolder(root_dir, transform=val_tf),
                   te_idx.tolist())

    kw = dict(num_workers=cfg.NUM_WORKERS, pin_memory=(cfg.DEVICE.type == 'cuda'),
              persistent_workers=False)
    return (DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  **kw),
            DataLoader(va_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, **kw),
            DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, **kw),
            full.classes, tr_ds, va_ds)

# --------------------------------------------------------------
# 6. DIAGNOSTICS
# --------------------------------------------------------------
def print_run_diagnostics(dataset_name, tr_ds, va_ds, te_ds,
                          class_names, tr_ld):
    print(f"\n--- Diagnostics for {dataset_name} ---")
    print(f"Batch size (cfg): {cfg.BATCH_SIZE}")
    xb, _ = next(iter(tr_ld))
    print(f"Actual batch shape from loader: {tuple(xb.shape)}")

    def counts_of(ds):
        base = ds.dataset
        return Counter(base.samples[i][1] for i in ds.indices)

    tr_c, va_c, te_c = counts_of(tr_ds), counts_of(va_ds), counts_of(te_ds)
    print(f"{'class':<25} {'train':>6} {'val':>6} {'test':>6}")
    for ci, name in enumerate(class_names):
        print(f"{name:<25} {tr_c.get(ci,0):>6} {va_c.get(ci,0):>6} {te_c.get(ci,0):>6}")
    print()

# --------------------------------------------------------------
# 7. CLASS WEIGHTS
# --------------------------------------------------------------
def class_weights_from_subset(ds, num_classes, device):
    base = ds.dataset
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for i in ds.indices:
        counts[base.samples[i][1]] += 1
    counts = counts.clamp(min=1.0)
    w = counts.sum() / counts
    w = w / w.mean()
    print(f"[class weights] counts={counts.int().tolist()}  "
          f"weights={[round(x,3) for x in w.tolist()]}")
    return w.to(device)

# --------------------------------------------------------------
# 8. TRAIN / EVAL
# --------------------------------------------------------------
from torch.amp import autocast, GradScaler

def make_scaler():
    return GradScaler(_AMP_DEVICE, enabled=_AMP_ENABLED)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, n_seen = 0.0, 0
    preds, labels_ = [], []
    accum = cfg.GRAD_ACCUM_STEPS
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc='Train', leave=False, ncols=110)
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

        total_loss += loss.item() * accum * imgs.size(0)
        n_seen     += imgs.size(0)
        preds.extend(logits.argmax(1).detach().float().cpu().numpy())
        labels_.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item()*accum:.4f}")

    return total_loss / n_seen, accuracy_score(labels_, preds)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_seen = 0.0, 0
    preds, labels_ = [], []
    for imgs, labels in tqdm(loader, desc='Eval ', leave=False, ncols=110):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(_AMP_DEVICE, enabled=_AMP_ENABLED, dtype=_AMP_DTYPE):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        n_seen     += imgs.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels_.extend(labels.cpu().numpy())
    return (total_loss / n_seen, accuracy_score(labels_, preds),
            preds, labels_)

# --------------------------------------------------------------
# 9. OPTIMIZER
# --------------------------------------------------------------
def build_optimizer(model):
    q, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (q if 'q_weights' in n else other).append(p)
    return optim.AdamW(
        [{'params': other, 'lr': cfg.LR},
         {'params': q,     'lr': cfg.LR * cfg.Q_LR_MULT}],
        weight_decay=cfg.WEIGHT_DECAY)

# --------------------------------------------------------------
# 10. TRAIN FULL
# --------------------------------------------------------------
def train_model(dataset_name, root_dir, use_class_weights=True):
    print(f"\n{'='*62}\n Training QAgroNet on {dataset_name}\n{'='*62}")
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {root_dir}\n"
            f"Expected ImageFolder layout: {root_dir}/<class_name>/*.jpg")

    tr_tf, va_tf = get_transforms()
    tr_ld, va_ld, te_ld, class_names, tr_ds, va_ds = load_split_dataset(
        root_dir, tr_tf, va_tf)
    print_run_diagnostics(dataset_name, tr_ds, va_ds, te_ld.dataset,
                          class_names, tr_ld)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train/Val/Test = {len(tr_ld.dataset)}/"
          f"{len(va_ld.dataset)}/{len(te_ld.dataset)}")
    print(f"Epochs={cfg.EPOCHS} LR={cfg.LR} WD={cfg.WEIGHT_DECAY} "
          f"LS={cfg.LABEL_SMOOTH} FREEZE_FRAC={cfg.FREEZE_FRAC}")

    model = QAgroNet(len(class_names), cfg).to(cfg.DEVICE)

    if use_class_weights:
        cw = class_weights_from_subset(tr_ds, len(class_names), cfg.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=cw,
                                        label_smoothing=cfg.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)

    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler    = make_scaler()

    best_acc, best_w = 0.0, copy.deepcopy(model.state_dict())
    hist = {'tr_loss': [], 'tr_acc': [], 'va_loss': [], 'va_acc': []}

    for ep in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        tr_l, tr_a = train_one_epoch(model, tr_ld, criterion, optimizer,
                                     scaler, cfg.DEVICE)
        va_l, va_a, _, _ = evaluate(model, va_ld, criterion, cfg.DEVICE)
        scheduler.step()

        hist['tr_loss'].append(tr_l); hist['tr_acc'].append(tr_a)
        hist['va_loss'].append(va_l); hist['va_acc'].append(va_a)

        if va_a > best_acc:
            best_acc, best_w = va_a, copy.deepcopy(model.state_dict())

        vram = (torch.cuda.max_memory_allocated()/1e9
                if cfg.DEVICE.type == 'cuda' else 0.0)
        print(f"Ep {ep:03d}/{cfg.EPOCHS} | tr {tr_l:.4f}/{tr_a:.4f} "
              f"| va {va_l:.4f}/{va_a:.4f} | {time.time()-t0:5.1f}s "
              f"| peak VRAM {vram:.2f} GB")

    model.load_state_dict(best_w)
    te_l, te_a, te_p, te_y = evaluate(model, te_ld, criterion, cfg.DEVICE)
    print(f"\n Test acc on {dataset_name}: {te_a:.4f}")
    print(classification_report(te_y, te_p, target_names=class_names, digits=4))

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(cfg.CKPT_DIR, f"qagronet_{dataset_name.lower()}.pth")
    torch.save({'model_state_dict': best_w,
                'class_names': class_names,
                'history': hist,
                'test_acc': te_a},
               ckpt_path)
    return model, hist, te_a

# --------------------------------------------------------------
# 11. TRANSFER
# --------------------------------------------------------------
def load_matching_state(model, src_state):
    tgt = model.state_dict()
    kept, dropped_shape, dropped_key = {}, [], []
    for k, v in src_state.items():
        if k not in tgt:
            dropped_key.append(k)
        elif tgt[k].shape != v.shape:
            dropped_shape.append((k, tuple(v.shape), tuple(tgt[k].shape)))
        else:
            kept[k] = v
    print(f"[load] kept {len(kept)}/{len(src_state)} keys")
    if dropped_key:
        print(f"[load] dropped (missing in target): {dropped_key}")
    if dropped_shape:
        print(f"[load] dropped (shape mismatch): {dropped_shape}")
    model.load_state_dict(kept, strict=False)

def cross_dataset_transfer(src_model, target_root, target_name,
                           epochs=60, fine_tune_head_only=False):
    print(f"\n{'='*62}\n Transfer → {target_name}\n{'='*62}")
    tr_tf, va_tf = get_transforms()
    tr_ld, va_ld, te_ld, class_names, tr_ds, va_ds = load_split_dataset(
        target_root, tr_tf, va_tf)
    print_run_diagnostics(target_name, tr_ds, va_ds, te_ld.dataset,
                          class_names, tr_ld)

    model = QAgroNet(len(class_names), cfg).to(cfg.DEVICE)
    if isinstance(src_model, dict):
        load_matching_state(model, src_model)
    elif hasattr(src_model, 'state_dict'):
        load_matching_state(model, src_model.state_dict())

    if fine_tune_head_only:
        for n, p in model.named_parameters():
            p.requires_grad = ('classifier' in n)
        lr = cfg.LR * 0.5
        print("[transfer] head-only fine-tune")
    else:
        for p in model.parameters():
            p.requires_grad = True
        lr = cfg.BPLD_LR
        print(f"[transfer] FULL network fine-tune, lr={lr}")

    cw = class_weights_from_subset(tr_ds, len(class_names), cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw,
                                    label_smoothing=cfg.LABEL_SMOOTH)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = make_scaler()

    best_acc, best_w = 0.0, copy.deepcopy(model.state_dict())
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_l, tr_a = train_one_epoch(model, tr_ld, criterion, optimizer,
                                     scaler, cfg.DEVICE)
        va_l, va_a, _, _ = evaluate(model, va_ld, criterion, cfg.DEVICE)
        scheduler.step()
        if va_a > best_acc:
            best_acc, best_w = va_a, copy.deepcopy(model.state_dict())
        print(f"FT ep {ep:03d}/{epochs} | tr {tr_l:.4f}/{tr_a:.4f} "
              f"| va {va_l:.4f}/{va_a:.4f} | {time.time()-t0:5.1f}s")

    model.load_state_dict(best_w)
    _, te_a, te_p, te_y = evaluate(model, te_ld, criterion, cfg.DEVICE)
    print(f"\n Transfer test acc on {target_name}: {te_a:.4f}")
    print(classification_report(te_y, te_p, target_names=class_names, digits=4))

    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(cfg.CKPT_DIR, f"qagronet_{target_name.lower()}_transfer.pth")
    torch.save({'model_state_dict': best_w,
                'class_names': class_names,
                'test_acc': te_a},
               ckpt_path)
    return te_a

# --------------------------------------------------------------
# 12. SMOKE TEST
# --------------------------------------------------------------
def smoke_test(dataset_name, root_dir, frac=0.10, epochs=2):
    print(f"\n{'='*62}\n SMOKE TEST on {dataset_name} "
          f"({frac:.0%} subset, {epochs} epochs)\n{'='*62}")
    tr_tf, va_tf = get_transforms()
    tr_ld, va_ld, _, class_names, tr_ds, va_ds = load_split_dataset(
        root_dir, tr_tf, va_tf)

    def _shrink(ds, f):
        n = max(4, int(len(ds) * f))
        idx = torch.randperm(len(ds),
                generator=torch.Generator().manual_seed(0))[:n].tolist()
        return Subset(ds, idx)

    kw = dict(num_workers=cfg.NUM_WORKERS, pin_memory=(cfg.DEVICE.type == 'cuda'))
    tr_ld = DataLoader(_shrink(tr_ds, frac), batch_size=cfg.BATCH_SIZE,
                       shuffle=True, **kw)
    va_ld = DataLoader(_shrink(va_ds, max(frac, 0.5)),
                       batch_size=cfg.BATCH_SIZE, shuffle=False, **kw)

    model = QAgroNet(len(class_names), cfg).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    optimizer = build_optimizer(model)
    scaler = make_scaler()

    for ep in range(1, epochs + 1):
        tr_l, tr_a = train_one_epoch(model, tr_ld, criterion, optimizer,
                                     scaler, cfg.DEVICE)
        va_l, va_a, _, _ = evaluate(model, va_ld, criterion, cfg.DEVICE)
        print(f"  smoke ep {ep}/{epochs} | tr {tr_l:.4f}/{tr_a:.4f} "
              f"| va {va_l:.4f}/{va_a:.4f}")

    print(f" SMOKE TEST on {dataset_name}: PASSED ✓")
    return True

# --------------------------------------------------------------
# 13. STAGE RUNNERS
# --------------------------------------------------------------
def run_stage_all():
    self_test()
    smoke_test('BPLD', cfg.BPLD_ROOT)

    # Stage 1 — PlantVillage
    cfg.EPOCHS = cfg.PV_EPOCHS
    cfg.LR     = cfg.PV_LR
    print(f"\n>>> STAGE 1: PlantVillage pretrain "
          f"({cfg.EPOCHS} epochs, lr={cfg.LR})")
    model_pv, _, acc_pv = train_model(
        'PlantVillage', cfg.PLANTVILLAGE_ROOT, use_class_weights=False)

    # Stage 2 — BPLD transfer
    cfg.EPOCHS = cfg.BPLD_EPOCHS
    cfg.LR     = cfg.BPLD_LR
    print(f"\n>>> STAGE 2: BPLD transfer "
          f"({cfg.EPOCHS} epochs, lr={cfg.LR}, full unfreeze)")
    acc_transfer = cross_dataset_transfer(
        model_pv, cfg.BPLD_ROOT, 'BPLD',
        epochs=cfg.BPLD_EPOCHS, fine_tune_head_only=False)

    print(f"\n{'='*62}\n FINAL\n{'='*62}")
    print(f" PlantVillage         : {acc_pv:.4f}")
    print(f" BPLD (after transfer): {acc_transfer:.4f}")

def run_stage_pv_only():
    self_test()
    cfg.EPOCHS = cfg.PV_EPOCHS
    cfg.LR     = cfg.PV_LR
    print(f"\n>>> STAGE pv-only: PlantVillage "
          f"({cfg.EPOCHS} epochs, lr={cfg.LR})")
    _, _, acc = train_model(
        'PlantVillage', cfg.PLANTVILLAGE_ROOT, use_class_weights=False)
    print(f"\nPlantVillage final: {acc:.4f}")

def run_stage_bpld_direct():
    self_test()
    smoke_test('BPLD', cfg.BPLD_ROOT)
    cfg.EPOCHS = cfg.BPLD_DIRECT_EPOCHS
    cfg.LR     = cfg.BPLD_DIRECT_LR
    print(f"\n>>> STAGE bpld-direct: BPLD "
          f"({cfg.EPOCHS} epochs, lr={cfg.LR}, full unfreeze, from ImageNet)")
    _, _, acc = train_model('BPLD', cfg.BPLD_ROOT, use_class_weights=True)
    print(f"\nBPLD direct final: {acc:.4f}")

def run_stage_smoke():
    self_test()
    smoke_test('BPLD', cfg.BPLD_ROOT)
    print("\nSmoke-only run complete.")

# --------------------------------------------------------------
# 14. CLI
# --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="QAgroNet training")
    ap.add_argument(
        '--stage', default='all',
        choices=['all', 'pv-only', 'bpld-direct', 'smoke'],
        help="which stage(s) to run (default: all)")
    args = ap.parse_args()

    print(f"\n=== QAgroNet — stage={args.stage} ===\n")
    if args.stage == 'all':
        run_stage_all()
    elif args.stage == 'pv-only':
        run_stage_pv_only()
    elif args.stage == 'bpld-direct':
        run_stage_bpld_direct()
    elif args.stage == 'smoke':
        run_stage_smoke()

if __name__ == '__main__':
    main()
