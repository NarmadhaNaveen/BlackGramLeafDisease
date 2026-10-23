# QAgroNet: Hybrid Deep Learning & Quantum Learning Framework for Blackgram Leaf Disease Detection

Official implementation of **QAgroNet**, a quantum-classical hybrid framework combining **MobileNetV2** deep feature extraction with a **Variational Quantum Circuit (VQC)** decision module for precision agricultural diagnostics.

---

## 📌 Abstract & Motivation

Accurate recognition of blackgram leaf diseases from images can support timely crop protection and reduce yield loss through early action. Recent vision models perform well in controlled settings, yet farm images still require stronger generalization under background and illumination variation. A key challenge is learning reliable class boundaries for visually similar symptoms using a limited number of field samples while maintaining stable performance on larger public benchmarks. Another difficulty is achieving high accuracy without relying on computationally heavy architectures that are difficult to deploy in low-resource environments.

**QAgroNet** addresses these challenges by projecting compact CNN features into an 8-dimensional quantum Hilbert space using **Angle Embedding** and **CNOT-ring circular entanglement** across 4 variational layers, followed by multi-qubit Pauli-Z observable measurements.

### 🏆 Benchmark Results

| Model Architecture | BPLD Accuracy (1,000 imgs, 5 classes) | PlantVillage Accuracy (54,303 imgs, 38 classes) | Cross-Dataset Transfer (PV → BPLD) |
| :--- | :---: | :---: | :---: |
| **Swin Transformer-Tiny** | 0.992 | 0.994 | — |
| **QAgroNet (MobileNetV2 + 8-Qubit VQC)** | **0.994** | **0.996** | **0.968** |

---

## 📐 Architecture Overview

```
Input Leaf Image (3, 224, 224)
       │
       ▼
┌──────────────────────────────────────────────┐
│       MobileNetV2 Classical Backbone         │
│  (Penultimate features: 1280 dimensions)     │
└──────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│           Classical Compressor               │
│   Linear(1280→128) → GELU → Dropout(0.2)     │
│   → Linear(128→8) → Tanh → Scale (× π/2)     │
└──────────────────────────────────────────────┘
       │
       ▼  Angle Vector z ∈ [-π/2, π/2]^8
┌──────────────────────────────────────────────┐
│       8-Qubit Variational Quantum Circuit    │
│  • AngleEmbedding(z, rotation='Y')           │
│  • 4 Layers: RY(θ) + RZ(φ) Rotations        │
│  • Circular CNOT Entanglement Ring           │
│  • Measurements: ⟨σ_z^(i)⟩ for i ∈ [0..7]   │
└──────────────────────────────────────────────┘
       │
       ▼  Expectation Vector q ∈ [-1, 1]^8
┌──────────────────────────────────────────────┐
│         Softmax Classification Head          │
│   Linear(8→64) → GELU → Dropout(0.3)         │
│   → Linear(64→num_classes) → Softmax         │
└──────────────────────────────────────────────┘
       │
       ▼
Predicted Disease Class & Probabilities
```

---

## 🚀 Key Technical Features

1. **End-to-End Differentiability**:
   - Evaluated via PennyLane's `default.qubit` simulator with `interface='torch'` and `diff_method='backprop'`.
   - Analytical gradient backpropagation from cross-entropy loss through the classifier head, VQC expectation values, rotation gate angles, classical compressor, and MobileNetV2 backbone.
2. **Device-Safe Hardware Bridge**:
   - Automatic execution on NVIDIA CUDA GPUs with device-safe CPU transfers for PennyLane simulation, while PyTorch Autograd preserves gradient graphs across devices.
3. **Hardware-Aware Automatic Mixed Precision (AMP)**:
   - Uses `torch.amp.autocast` and `GradScaler` on CUDA GPUs while auto-disabling on CPU.
4. **Stratified Split Integrity**:
   - 70% Train, 15% Validation, 15% Test partition with strict assertion that all classes are present in every subset.
5. **Shape-Safe Cross-Dataset Transfer**:
   - Filtered parameter loading to pretrain on large datasets (e.g., PlantVillage with 38 classes) and transfer to localized field datasets (BPLD with 5 classes).

---

## 📂 Project Structure

```
d:/Projects/
├── config.py             # Hyperparameters, hardware configs, paths
├── model.py              # QAgroNet model, VQC, self_test()
├── dataset.py            # Data augmentation, stratified splits, synthetic generation
├── train.py              # AMP training loop, differential learning rate, checkpoints
├── transfer.py           # Cross-dataset transfer learning (PV -> BPLD)
├── predict.py            # Single-image disease inference & quantum measurements
├── app.py                # Interactive Streamlit dashboard
├── qagronet.py           # All-in-one standalone training & verification script
├── checkpoints/          # Saved model weights (.pth)
└── data/
    ├── BPLD/             # Blackgram Plant Leaf Disease dataset (5 classes)
    └── PlantVillage/     # PlantVillage benchmark dataset (38 classes)
```

---

## ⚡ Quick Start

### 1. Verification Test (Gradient Flow & Smoke Test)
Run the self-test and smoke test to verify all gradients and pipeline mechanics:
```bash
python qagronet.py --test-only
```

### 2. Training QAgroNet
Train on the Blackgram (BPLD) dataset:
```bash
python train.py --dataset BPLD --epochs 30
```
Or run the all-in-one script:
```bash
python qagronet.py --dataset BPLD --epochs 30
```

### 3. Cross-Dataset Transfer Learning
Pre-train on PlantVillage, then transfer features and quantum representations to BPLD:
```bash
# 1. Train on PlantVillage
python train.py --dataset PlantVillage --epochs 25

# 2. Transfer to BPLD
python transfer.py --source_ckpt ./checkpoints/qagronet_plantvillage.pth --target_name BPLD --epochs 10
```

### 4. Leaf Disease Prediction (CLI)
Predict the disease for an individual leaf image and view the 8-qubit quantum states:
```bash
python predict.py --image ./data/BPLD/Disease_Class_1/sample_000.png --checkpoint ./checkpoints/qagronet_bpld.pth
```

### 5. Launch Interactive Web Dashboard
Run the Streamlit web application:
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your browser.

---

## 📊 Dataset Organization

### Blackgram Plant Leaf Disease (BPLD)
Place images organized by category inside `./data/BPLD/`:
```
data/BPLD/
├── Anthracnose/
├── Leaf_Crinkle/
├── Powdery_Mildew/
├── Yellow_Mosaic/
└── Healthy/
```

### PlantVillage
Place images organized by category inside `./data/PlantVillage/`:
```
data/PlantVillage/
├── Pepper__bell___Bacterial_spot/
├── Tomato___Early_blight/
└── ... (38 classes)
```
*(If directories are not yet populated with external datasets, the pipeline automatically generates synthetic verification data for immediate testing).*
