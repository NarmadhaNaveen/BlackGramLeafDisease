"""
QAgroNet Configuration Module
Centralized hyperparameters and settings for Quantum-Classical Hybrid Architecture.
"""

import math
import torch

class Config:
    # ---------------- Quantum Circuit Parameters ----------------
    N_QUBITS    = 8          # 8 qubits for 8-dimensional quantum state
    N_LAYERS    = 4          # 4 variational quantum layers
    EMBED_DIM   = 8          # Compressed feature dimension matching N_QUBITS
    ANGLE_SCALE = math.pi / 2 # Scale features into [-pi/2, pi/2] for AngleEmbedding

    # ---------------- Classical Backbone ----------------
    FEATURE_DIM = 1280       # MobileNetV2 penultimate feature dimension
    FREEZE_FRAC = 0.5        # Fraction of MobileNetV2 feature blocks frozen

    # ---------------- Training Hyperparameters ----------------
    BATCH_SIZE       = 64
    GRAD_ACCUM_STEPS = 1
    EPOCHS           = 30
    LR               = 3e-4
    Q_LR_MULT        = 5.0   # Multiplier for quantum parameter learning rate
    WEIGHT_DECAY     = 1e-4
    LABEL_SMOOTH     = 0.05
    SEED             = 42
    NUM_WORKERS      = 0     # Set to 0 for Windows compatibility & low latency

    # ---------------- Hardware & AMP ----------------
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True

    # ---------------- Dataset Splits ----------------
    TRAIN_RATIO = 0.70
    VAL_RATIO   = 0.15
    TEST_RATIO  = 0.15

    # ---------------- File Paths ----------------
    BPLD_ROOT         = './data/BPLD'
    PLANTVILLAGE_ROOT = './data/PlantVillage'
    CKPT_DIR          = './checkpoints'

cfg = Config()

_AMP_DEVICE  = cfg.DEVICE.type
_AMP_ENABLED = bool(cfg.USE_AMP and _AMP_DEVICE == 'cuda')
_AMP_DTYPE   = torch.float16 if _AMP_ENABLED else torch.float32
