"""
generate_intermediate_images.py
Generates high-resolution publication-quality intermediate visualization images
for the QAgroNet Plant Leaf Disease detection framework.

Intermediate Stages Visualized:
1. Raw & Preprocessed Leaf Images
2. Low-level Early Conv Feature Maps (Edges, Veins, Surface Texture)
3. Mid-level Bottleneck Feature Maps (Lesions, Spot Contours)
4. Deep Penultimate Conv Feature Maps (Disease Semantic Patterns)
5. Grad-CAM Class Activation Heatmap & Saliency Overlay
6. Quantum Hilbert Space Angle Vector (z) & 8-Qubit Pauli-Z Expectation Values (<sigma_z>)
7. Softmax Diagnostic Output & Probability Distribution
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Set UTF-8 safe output
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from config import cfg
from model import QAgroNet, quantum_forward

# Preprocessing transforms
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class GradCAM:
    """Computes Grad-CAM saliency for the penultimate convolutional layer."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self._hook_handles.extend([h1, h2])

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        # Forward pass
        f = self.model.backbone(input_tensor)
        f_pool = self.model.avgpool(f).flatten(1)
        z = (self.model.compress(f_pool) * self.model.angle_scale).float()
        q = quantum_forward(z, self.model.q_weights)
        logits = self.model.classifier(q)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Gradient-weighted feature saliency
        gradients = self.gradients[0]     # (1280, 7, 7)
        activations = self.activations[0] # (1280, 7, 7)
        weights = gradients.mean(dim=(1, 2), keepdim=True) # (1280, 1, 1)

        # Absolute gradient-weighted importance (preserves bidirectional feature impacts)
        cam = torch.abs(weights * activations).sum(dim=0) # (7, 7)
        cam = cam.detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, logits.detach(), q.detach(), z.detach()

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


def extract_intermediate_feature_maps(model, input_tensor):
    """Hooks into MobileNetV2 early, mid, and deep layers."""
    features = {}

    def get_hook(name):
        def hook(m, inp, out):
            features[name] = out.detach().cpu()
        return hook

    h1 = model.backbone[0].register_forward_hook(get_hook('early'))
    h2 = model.backbone[4].register_forward_hook(get_hook('mid'))
    h3 = model.backbone[18].register_forward_hook(get_hook('deep'))

    with torch.no_grad():
        _ = model.backbone(input_tensor)

    h1.remove()
    h2.remove()
    h3.remove()
    return features


def load_trained_model(ckpt_path=None):
    device = cfg.DEVICE
    default_classes = ['anthracnose', 'healthy', 'leaf crinckle', 'powdery mildew', 'yellow mosaic']
    class_names = default_classes

    if ckpt_path and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        if 'class_names' in ckpt:
            class_names = ckpt['class_names']
        model = QAgroNet(num_classes=len(class_names), config=cfg).to(device)
        model.load_state_dict(state_dict)
        print(f"[Model] Successfully loaded weights from {ckpt_path}")
    else:
        model = QAgroNet(num_classes=len(class_names), config=cfg).to(device)
        print("[Model] Using default initialization")

    model.eval()
    return model, class_names


def overlay_cam(img_pil, cam, alpha=0.55, colormap=cm.jet):
    """Smooths and overlays Grad-CAM heatmap onto leaf image."""
    cam_uint8 = np.uint8(255 * cam)
    cam_img = Image.fromarray(cam_uint8).resize(img_pil.size, Image.Resampling.BICUBIC)
    # Apply subtle Gaussian blur for smooth natural heatmap contour
    cam_img = cam_img.filter(ImageFilter.GaussianBlur(radius=8))
    cam_np = np.array(cam_img) / 255.0
    heatmap = colormap(cam_np)[:, :, :3] # RGB
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap))
    overlay = Image.blend(img_pil, heatmap_pil, alpha=alpha)
    return heatmap_pil, overlay


def generate_pipeline_walkthrough(model, img_path, class_names, out_path):
    """
    Figure 1: Full Step-by-Step Intermediate Pipeline Walkthrough
    Shows:
    Stage 1: Raw Input Leaf Image
    Stage 2: Shallow Features (Conv 32ch @ 112×112: Edges & Veins)
    Stage 3: Mid-Level Features (Bottleneck @ 28×28: Lesion Boundaries)
    Stage 4: Penultimate Semantics (Conv 1280ch @ 7×7: Disease Pathology)
    Stage 5: Grad-CAM Saliency Heatmap (Activation Attention)
    Stage 6: Attention Overlay (Model Focus on Leaf Symptoms)
    Stage 7: 8-Qubit Quantum Observables (Pauli-Z Expectation Values)
    Stage 8: Final Diagnostic Decision & Confidence Distribution
    """
    raw_img = Image.open(img_path).convert('RGB').resize((224, 224))
    input_tensor = PREPROCESS(raw_img).unsqueeze(0).to(cfg.DEVICE)

    # 1. Grad-CAM & Quantum Measurements
    target_layer = model.backbone[18]
    cam_generator = GradCAM(model, target_layer)
    cam, logits, q_out, z_out = cam_generator.generate(input_tensor)
    cam_generator.remove_hooks()

    # 2. Intermediate Feature Maps
    feat_maps = extract_intermediate_feature_maps(model, input_tensor)
    early_map = feat_maps['early'][0].mean(dim=0).numpy() # (112, 112)
    mid_map   = feat_maps['mid'][0].mean(dim=0).numpy()   # (28, 28)
    deep_map  = feat_maps['deep'][0].mean(dim=0).numpy()  # (7, 7)

    def norm_map(m):
        return (m - m.min()) / (m.max() - m.min() + 1e-8)

    early_map = norm_map(early_map)
    mid_map = norm_map(mid_map)
    deep_map = norm_map(deep_map)

    heatmap_pil, overlay_pil = overlay_cam(raw_img, cam, alpha=0.55)

    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    pred_label = class_names[pred_idx]
    q_vals = q_out.squeeze(0).cpu().numpy()

    # Create publication figure
    fig = plt.figure(figsize=(19, 10), dpi=300)
    fig.patch.set_facecolor('#0f172a') # Dark slate background

    # Title & Subtitle banner with ample spacing
    fig.text(0.5, 0.965, "QAgroNet — Intermediate Representations & Diagnostic Output Pipeline",
             fontsize=17, fontweight='bold', color='white', ha='center')
    fig.text(0.5, 0.932, f"Input Sample: {os.path.basename(img_path)}  |  Diagnosis: {pred_label.upper()} ({probs[pred_idx]*100:.2f}% Confidence)",
             fontsize=12, color='#38bdf8', ha='center')

    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25,
                           top=0.88, bottom=0.08, left=0.06, right=0.96)

    # Subplot 1: Raw Input
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(raw_img)
    ax1.set_title("Stage 1: Input Leaf Image\n(RGB 224 × 224)", color='white', fontsize=11, fontweight='bold', pad=8)
    ax1.axis('off')

    # Subplot 2: Early Feature Map
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(early_map, cmap='viridis')
    ax2.set_title("Stage 2: Shallow Features\n(Conv 32ch @ 112×112: Edges & Veins)", color='white', fontsize=11, fontweight='bold', pad=8)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors='white', labelsize=8)

    # Subplot 3: Mid Feature Map
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(mid_map, cmap='magma')
    ax3.set_title("Stage 3: Mid-Level Features\n(Bottleneck @ 28×28: Lesion Boundaries)", color='white', fontsize=11, fontweight='bold', pad=8)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).ax.tick_params(colors='white', labelsize=8)

    # Subplot 4: Deep Feature Map
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(deep_map, cmap='inferno')
    ax4.set_title("Stage 4: Penultimate Semantics\n(Conv 1280ch @ 7×7: Disease Pathology)", color='white', fontsize=11, fontweight='bold', pad=8)
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).ax.tick_params(colors='white', labelsize=8)

    # Subplot 5: Grad-CAM Saliency Heatmap
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(cam, cmap='jet')
    ax5.set_title("Stage 5: Grad-CAM Heatmap\n(Activation Saliency on Lesions)", color='white', fontsize=11, fontweight='bold', pad=8)
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04).ax.tick_params(colors='white', labelsize=8)

    # Subplot 6: Grad-CAM Overlay
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(overlay_pil)
    ax6.set_title("Stage 6: Attention Overlay\n(Model Focus on Leaf Symptoms)", color='white', fontsize=11, fontweight='bold', pad=8)
    ax6.axis('off')

    # Subplot 7: 8-Qubit Quantum Observables
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.set_facecolor('#1e293b')
    q_x = np.arange(cfg.N_QUBITS)
    q_colors = ['#10b981' if v >= 0 else '#f43f5e' for v in q_vals]
    bars = ax7.bar(q_x, q_vals, color=q_colors, edgecolor='white', lw=0.6, width=0.65)
    ax7.axhline(0, color='white', lw=0.8, linestyle='--')
    ax7.set_xticks(q_x)
    ax7.set_xticklabels([f"q[{i}]" for i in q_x], color='white', fontsize=8.5)
    ax7.set_ylim(-1.18, 1.18)
    ax7.tick_params(colors='white', labelsize=8.5)
    ax7.set_ylabel(r"Pauli-Z $\langle \sigma_z \rangle$", color='white', fontsize=9.5, fontweight='bold')
    ax7.set_title("Stage 7: 8-Qubit Quantum Observables\n(VQC Hilbert Space Expectation Values)", color='white', fontsize=11, fontweight='bold', pad=8)
    for b in bars:
        h = b.get_height()
        va = 'bottom' if h >= 0 else 'top'
        ax7.text(b.get_x() + b.get_width()/2., h + (0.05 if h>=0 else -0.15), f"{h:.2f}",
                 ha='center', va=va, color='white', fontsize=7.5, fontweight='bold')
    for spine in ax7.spines.values(): spine.set_color('#334155')

    # Subplot 8: Diagnostic Output Probability Distribution
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.set_facecolor('#1e293b')
    display_names = [c.replace('_', ' ').title() for c in class_names]
    y_pos = np.arange(len(class_names))
    prob_colors = ['#38bdf8' if i == pred_idx else '#64748b' for i in range(len(class_names))]
    bars8 = ax8.barh(y_pos, probs * 100, color=prob_colors, edgecolor='white', lw=0.5, height=0.65)
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(display_names, color='white', fontsize=9, fontweight='medium')
    ax8.set_xlim(0, 115)
    ax8.tick_params(colors='white', labelsize=8.5)
    ax8.set_xlabel("Confidence (%)", color='white', fontsize=9.5, fontweight='bold')
    ax8.set_title(f"Stage 8: Final Diagnosis\n★ {pred_label.replace('_', ' ').title()} ({probs[pred_idx]*100:.1f}%)",
                  color='#38bdf8', fontsize=11, fontweight='bold', pad=8)
    for b in bars8:
        w = b.get_width()
        ax8.text(w + 2.5, b.get_y() + b.get_height()/2., f"{w:.1f}%",
                 ha='left', va='center', color='white', fontsize=8, fontweight='bold')
    for spine in ax8.spines.values(): spine.set_color('#334155')

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[Generated Figure 1] {out_path}")


def generate_multiclass_intermediate_grid(model, class_samples, class_names, out_path):
    """
    Figure 2: Multi-Class Intermediate Progression Comparison Grid
    5 Rows (One per disease class):
    Anthracnose, Healthy, Leaf Crinkle, Powdery Mildew, Yellow Mosaic.
    6 Columns:
    1. Input Leaf Image
    2. Shallow Feature Map (Edges & Surface)
    3. Deep Penultimate Map (Lesion Semantics)
    4. Grad-CAM Activation Heatmap
    5. Attention Saliency Overlay
    6. 8-Qubit Quantum Observable Signature & Diagnosis
    """
    n_classes = len(class_samples)
    fig, axes = plt.subplots(n_classes, 6, figsize=(20, 3.4 * n_classes), dpi=300)
    fig.patch.set_facecolor('#0f172a')

    fig.text(0.5, 0.985, "QAgroNet — Multi-Class Intermediate Representations & Disease Diagnostic Grid",
             fontsize=17, fontweight='bold', color='white', ha='center')

    col_titles = [
        "1. Input Leaf Image",
        "2. Shallow Features (Edges)",
        "3. Deep Features (Pathology)",
        "4. Grad-CAM Heatmap",
        "5. Attention Overlay",
        "6. Quantum Signature & Output"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].annotate(title, xy=(0.5, 1.15), xycoords='axes fraction',
                                  ha='center', va='bottom', fontsize=12, fontweight='bold',
                                  color='#38bdf8')

    target_layer = model.backbone[18]
    cam_generator = GradCAM(model, target_layer)

    for row_idx, (cname, img_path) in enumerate(class_samples.items()):
        raw_img = Image.open(img_path).convert('RGB').resize((224, 224))
        input_tensor = PREPROCESS(raw_img).unsqueeze(0).to(cfg.DEVICE)

        cam, logits, q_out, _ = cam_generator.generate(input_tensor)
        feat_maps = extract_intermediate_feature_maps(model, input_tensor)

        early_map = feat_maps['early'][0].mean(dim=0).numpy()
        deep_map = feat_maps['deep'][0].mean(dim=0).numpy()
        early_norm = (early_map - early_map.min()) / (early_map.max() - early_map.min() + 1e-8)
        deep_norm = (deep_map - deep_map.min()) / (deep_map.max() - deep_map.min() + 1e-8)

        _, overlay_pil = overlay_cam(raw_img, cam, alpha=0.55)

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_name = class_names[pred_idx]
        conf = probs[pred_idx] * 100
        q_vals = q_out.squeeze(0).cpu().numpy()

        display_true = cname.replace('_', ' ').title()
        display_pred = pred_name.replace('_', ' ').title()

        # Col 1: Raw Image
        ax1 = axes[row_idx, 0]
        ax1.imshow(raw_img)
        ax1.set_ylabel(f"{display_true}\n(Ground Truth)", color='#f8fafc', fontsize=11, fontweight='bold', labelpad=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        # Col 2: Shallow
        ax2 = axes[row_idx, 1]
        ax2.imshow(early_norm, cmap='viridis')
        ax2.set_xticks([]); ax2.set_yticks([])

        # Col 3: Deep
        ax3 = axes[row_idx, 2]
        ax3.imshow(deep_norm, cmap='inferno')
        ax3.set_xticks([]); ax3.set_yticks([])

        # Col 4: Heatmap
        ax4 = axes[row_idx, 3]
        ax4.imshow(cam, cmap='jet')
        ax4.set_xticks([]); ax4.set_yticks([])

        # Col 5: Overlay
        ax5 = axes[row_idx, 4]
        ax5.imshow(overlay_pil)
        ax5.set_xticks([]); ax5.set_yticks([])

        # Col 6: Quantum Bar + Prediction text
        ax6 = axes[row_idx, 5]
        ax6.set_facecolor('#1e293b')
        q_x = np.arange(cfg.N_QUBITS)
        q_colors = ['#10b981' if v >= 0 else '#f43f5e' for v in q_vals]
        ax6.bar(q_x, q_vals, color=q_colors, width=0.65, edgecolor='white', lw=0.4)
        ax6.axhline(0, color='white', lw=0.6, linestyle='--')
        ax6.set_ylim(-1.18, 1.18)
        ax6.set_xticks(q_x)
        ax6.set_xticklabels([f"q{i}" for i in q_x], color='white', fontsize=7.5)
        ax6.tick_params(colors='white', labelsize=7.5)
        ax6.set_ylabel(r"$\langle \sigma_z \rangle$", color='white', fontsize=8)

        is_match = (pred_name.lower().replace('_', ' ') == cname.lower().replace('_', ' '))
        status_color = '#4ade80' if is_match else '#f87171'
        ax6.set_title(f"Pred: {display_pred} ({conf:.1f}%)", color=status_color, fontsize=10, fontweight='bold', pad=4)
        for spine in ax6.spines.values(): spine.set_color('#334155')

    cam_generator.remove_hooks()
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[Generated Figure 2] {out_path}")


def generate_feature_map_channels_grid(model, img_path, out_path, num_channels=16):
    """
    Figure 3: Deep Dive into Multi-Channel Intermediate Feature Activations
    Displays 16 individual filter activation maps across:
    - Shallow layer (Low-level edge / texture detectors)
    - Mid layer (Inverted Residual lesion contour detectors)
    - Penultimate layer (High-level semantic disease symptom detectors)
    """
    raw_img = Image.open(img_path).convert('RGB').resize((224, 224))
    input_tensor = PREPROCESS(raw_img).unsqueeze(0).to(cfg.DEVICE)
    feat_maps = extract_intermediate_feature_maps(model, input_tensor)

    fig = plt.figure(figsize=(18, 12), dpi=300)
    fig.patch.set_facecolor('#0f172a')

    fig.text(0.5, 0.97, "QAgroNet — Filter Activation Channels Across Intermediate Neural Layers",
             fontsize=17, fontweight='bold', color='white', ha='center')
    fig.text(0.5, 0.94, f"Image: {os.path.basename(img_path)} | Visualizing Top 16 Filter Channels per Depth Stage",
             fontsize=11.5, color='#38bdf8', ha='center')

    stages = [
        ('Shallow: ConvBNActivation Layer 0 (32 channels, 112×112)', feat_maps['early'][0], 'viridis'),
        ('Mid: Inverted Residual Block 4 (32 channels, 28×28)', feat_maps['mid'][0], 'magma'),
        ('Deep: Penultimate Conv Layer 18 (1280 channels, 7×7)', feat_maps['deep'][0], 'inferno')
    ]

    gs = fig.add_gridspec(3, num_channels, hspace=0.38, wspace=0.12,
                           top=0.91, bottom=0.05, left=0.04, right=0.98)

    for row_idx, (stage_name, tensor_maps, colormap) in enumerate(stages):
        fig.text(0.04, 0.90 - row_idx * 0.30, stage_name,
                 fontsize=12, fontweight='bold', color='white', ha='left')

        variances = tensor_maps.var(dim=(1, 2)).numpy()
        top_channel_indices = np.argsort(variances)[::-1][:num_channels]

        for col_idx, ch in enumerate(top_channel_indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ch_map = tensor_maps[ch].numpy()
            ch_norm = (ch_map - ch_map.min()) / (ch_map.max() - ch_map.min() + 1e-8)
            ax.imshow(ch_norm, cmap=colormap)
            ax.set_title(f"Ch {ch}", color='#94a3b8', fontsize=7.5, pad=3)
            ax.axis('off')

    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"[Generated Figure 3] {out_path}")


def main():
    out_dir = './results'
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = './checkpoints/qagronet_bpld_transfer.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = './checkpoints/qagronet_bpld.pth'

    model, class_names = load_trained_model(ckpt_path)
    print(f"[Main] Model classes ({len(class_names)}): {class_names}")

    # Map dataset folders in exact class list order
    sample_images = {}
    bpld_root = cfg.BPLD_ROOT
    for c in class_names:
        c_dir = os.path.join(bpld_root, c)
        if os.path.isdir(c_dir):
            imgs = [os.path.join(c_dir, f) for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.png'))]
            if imgs:
                sample_images[c] = imgs[0]

    print("\nSelected representative sample images:")
    for c, p in sample_images.items():
        print(f"  {c:<18}: {p}")

    # 1. Primary Disease Walkthrough (Anthracnose sample)
    primary_img = sample_images.get('anthracnose', list(sample_images.values())[0])
    fig1_path = os.path.join(out_dir, 'intermediate_pipeline_walkthrough.png')
    generate_pipeline_walkthrough(model, primary_img, class_names, fig1_path)

    # 2. Yellow Mosaic Walkthrough
    if 'yellow mosaic' in sample_images:
        fig1_ym_path = os.path.join(out_dir, 'intermediate_pipeline_yellow_mosaic.png')
        generate_pipeline_walkthrough(model, sample_images['yellow mosaic'], class_names, fig1_ym_path)

    # 3. Multi-Class Comparison Grid
    fig2_path = os.path.join(out_dir, 'intermediate_multiclass_comparison.png')
    generate_multiclass_intermediate_grid(model, sample_images, class_names, fig2_path)

    # 4. Multi-Channel Feature Maps Deep Dive
    fig3_path = os.path.join(out_dir, 'intermediate_feature_channels_grid.png')
    generate_feature_map_channels_grid(model, primary_img, fig3_path)

    print("\n" + "="*62)
    print(" ALL INTERMEDIATE OUTPUT IMAGES RE-GENERATED SUCCESSFULLY!")
    print("="*62)

if __name__ == '__main__':
    main()
