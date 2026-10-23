"""
Inference script for QAgroNet:
Classifies leaf images and outputs disease predictions along with 8-qubit quantum state measurements.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import cfg
from model import QAgroNet
from dataset import BPLD_CLASSES

def predict_single_image(image_path, checkpoint_path=None, class_names=None):
    """
    Runs end-to-end inference on a single image.
    Returns:
    - predicted_class: str
    - confidence: float
    - class_probabilities: dict
    - quantum_expectations: list of float (8 Pauli-Z expectation values)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Determine classes
    if class_names is None:
        class_names = BPLD_CLASSES

    # Initialize model
    model = QAgroNet(num_classes=len(class_names), config=cfg).to(cfg.DEVICE)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        if 'class_names' in ckpt:
            class_names = ckpt['class_names']
            # Rebuild model if class count changed
            if len(class_names) != model.classifier[-1].out_features:
                model = QAgroNet(num_classes=len(class_names), config=cfg).to(cfg.DEVICE)
        model.load_state_dict(state_dict)
        print(f"[Inference] Loaded weights from {checkpoint_path}")
    else:
        print("[Inference] Notice: No checkpoint provided or found. Running with initial weights.")

    model.eval()

    # Preprocessing
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = tf(img).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        logits, q_out = model(tensor, return_quantum_state=True)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        q_vals = q_out.squeeze(0).cpu().numpy().tolist()

    pred_idx = int(probs.argmax())
    predicted_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    return predicted_class, confidence, prob_dict, q_vals

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict Leaf Disease with QAgroNet")
    parser.add_argument('--image', type=str, required=True, help="Path to input leaf image")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to trained model checkpoint")
    args = parser.parse_args()

    pred_class, conf, all_probs, q_measurements = predict_single_image(args.image, args.checkpoint)

    print("\n" + "=" * 62)
    print(" QAgroNet INFERENCE RESULTS")
    print("=" * 62)
    print(f" Predicted Disease : {pred_class}")
    print(f" Confidence        : {conf * 100:.2f}%\n")

    print(" Class Probabilities:")
    for c, p in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(p * 30)
        print(f"  - {c:<20} : {p*100:5.2f}% | {bar}")

    print("\n Quantum Variational Circuit Measurements (8 Qubits Pauli-Z Expectations):")
    for i, val in enumerate(q_measurements):
        sign = "+" if val >= 0 else ""
        print(f"  - Qubit q[{i}] <sigma_z> : {sign}{val:.4f}")
    print("=" * 62 + "\n")
