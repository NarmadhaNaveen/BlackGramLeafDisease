"""
QAgroNet Architecture: Hybrid MobileNetV2 + Variational Quantum Circuit (VQC)
End-to-end differentiable pipeline with device-safe PennyLane boundary.
"""

import math
import torch
import torch.nn as nn
from torchvision import models
import pennylane as qml
from config import cfg

# Initialize PennyLane quantum device (8 qubits)
QDEV = qml.device("default.qubit", wires=cfg.N_QUBITS)

def _cnot_ring(wires):
    """Circular entanglement across all qubits using CNOT gates."""
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])

@qml.qnode(QDEV, interface='torch', diff_method='backprop')
def quantum_circuit_single(inputs, weights):
    """
    Simulates the VQC for a single sample.
    inputs : (N_QUBITS,)            — features encoded as angles in [-pi/2, pi/2]
    weights: (N_LAYERS, N_QUBITS, 2) — trainable RY and RZ parameters
    returns: list[N_QUBITS] of expectation values <PauliZ> in [-1.0, 1.0]
    """
    wires = list(range(cfg.N_QUBITS))
    # 1. Angle Embedding with Pauli-Y rotation
    qml.AngleEmbedding(inputs, wires=wires, rotation='Y')

    # 2. Variational Quantum Circuit Layers
    for layer in range(cfg.N_LAYERS):
        for i, w in enumerate(wires):
            qml.RY(weights[layer, i, 0], wires=w)
            qml.RZ(weights[layer, i, 1], wires=w)
        _cnot_ring(wires)

    # 3. Measurement: Pauli-Z expectation values across all 8 wires
    return [qml.expval(qml.PauliZ(w)) for w in wires]

def quantum_forward(z_batch, weights):
    """
    Batch wrapper for quantum circuit evaluation.
    Device-safe & dtype-safe:
    - Transfers inputs and weights to CPU for PennyLane simulation.
    - Transfers results back to original device (CUDA/CPU) with matching dtype (fp32).
    - PyTorch autograd tracks the computation graph seamlessly across devices.
    """
    orig_device = z_batch.device
    z_cpu = z_batch.cpu()
    w_cpu = weights.cpu()

    outs = [torch.stack(quantum_circuit_single(z_cpu[i], w_cpu))
            for i in range(z_cpu.size(0))]
    out = torch.stack(outs, dim=0)          # (B, N_QUBITS), fp64
    return out.to(device=orig_device, dtype=z_batch.dtype)


class QAgroNet(nn.Module):
    """
    QAgroNet Model:
    1. MobileNetV2 feature extractor (1280 features)
    2. Classical compression (1280 -> 128 -> 8) scaled by pi/2
    3. Variational Quantum Circuit (8 qubits, Angle Embedding, 4 layers CNOT ring)
    4. Softmax decision head (8 -> 64 -> num_classes)
    """
    def __init__(self, num_classes, config=cfg):
        super().__init__()
        self.cfg = config

        # Pretrained MobileNetV2 backbone
        bb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = bb.features
        self.avgpool  = nn.AdaptiveAvgPool2d(1)

        # Freeze early layers
        n_blocks = len(self.backbone)
        n_freeze = int(n_blocks * self.cfg.FREEZE_FRAC)
        for i, blk in enumerate(self.backbone):
            if i < n_freeze:
                for p in blk.parameters():
                    p.requires_grad = False

        # Classical compressor: 1280 -> 8
        self.compress = nn.Sequential(
            nn.Linear(self.cfg.FEATURE_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.cfg.EMBED_DIM),
            nn.Tanh(),
        )
        self.register_buffer('angle_scale', torch.tensor(self.cfg.ANGLE_SCALE))

        # Trainable variational quantum weights: (N_LAYERS, N_QUBITS, 2)
        self.q_weights = nn.Parameter(
            torch.randn(self.cfg.N_LAYERS, self.cfg.N_QUBITS, 2) * 0.1
        )

        # Classification head on top of quantum expectation values
        self.classifier = nn.Sequential(
            nn.Linear(self.cfg.N_QUBITS, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def extract_features(self, x):
        """Extract deep CNN features and compressed quantum angles."""
        f = self.backbone(x)
        f = self.avgpool(f).flatten(1)
        z = self.compress(f) * self.angle_scale
        return f, z

    def forward(self, x, return_quantum_state=False):
        """
        Forward pass:
        x: (B, 3, 224, 224)
        returns: logits (B, num_classes)
        """
        _, z = self.extract_features(x)
        z_fp32 = z.float()
        q_out = quantum_forward(z_fp32, self.q_weights)   # (B, 8)
        logits = self.classifier(q_out)
        if return_quantum_state:
            return logits, q_out
        return logits


def self_test(device=cfg.DEVICE):
    """
    Self-test unit: Verifies end-to-end differentiability through all submodules.
    """
    print("\n" + "=" * 62)
    print(" SELF-TEST: Verifying gradient flow across submodules")
    print("=" * 62)

    m = QAgroNet(num_classes=5, config=cfg).to(device)
    x = torch.randn(4, 3, 224, 224, device=device)
    y = torch.tensor([0, 1, 2, 3], device=device)

    logits = m(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    def has_grad(params):
        return any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in params if p.requires_grad)

    checks = {
        'MobileNetV2 unfrozen blocks': has_grad(m.backbone.parameters()),
        'Compressor (1280 -> 8)'     : has_grad(m.compress.parameters()),
        'Quantum circuit weights'    : (m.q_weights.grad is not None and m.q_weights.grad.abs().sum().item() > 0),
        'Softmax classifier head'    : has_grad(m.classifier.parameters()),
    }

    for k, ok in checks.items():
        status = "PASSED" if ok else "FAILED"
        print(f"  [{status:^6}] Gradient flows to {k}")

    # Probe quantum input gradient directly
    z_probe = torch.randn(2, cfg.EMBED_DIM, device=device, requires_grad=True)
    out = quantum_forward(z_probe, m.q_weights)
    out.sum().backward()
    z_ok = z_probe.grad is not None and z_probe.grad.abs().sum().item() > 0
    print(f"  [{'PASSED' if z_ok else 'FAILED':^6}] Gradient flows through quantum input (z)")

    all_ok = all(checks.values()) and z_ok
    print("=" * 62)
    print(f" SELF-TEST RESULT: {'SUCCESS' if all_ok else 'FAILED'}")
    print("=" * 62 + "\n")
    if not all_ok:
        raise RuntimeError("Self-test failed! Gradient flow is obstructed.")
    return m

if __name__ == '__main__':
    self_test()
