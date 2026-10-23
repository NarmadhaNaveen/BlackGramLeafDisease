"""
QAgroNet Interactive Streamlit Dashboard
Visualizes Hybrid Quantum-Classical Deep Learning for Blackgram Leaf Disease Detection.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import streamlit as st
import pandas as pd

from config import cfg
from model import QAgroNet
from dataset import BPLD_CLASSES

st.set_page_config(
    page_title="QAgroNet — Quantum-Classical Leaf Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10b981, #06b6d4, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .highlight-badge {
        background-color: #065f46;
        color: #6ee7b7;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    st.markdown("**QAgroNet Specs:**")
    st.markdown("- **Backbone:** MobileNetV2 (1280 features)")
    st.markdown("- **Classical Compression:** 1280 → 8 (Tanh × π/2)")
    st.markdown("- **Qubits:** 8 qubits (`default.qubit`)")
    st.markdown("- **VQC Depth:** 4 layers (RY + RZ gates)")
    st.markdown("- **Entanglement:** CNOT Circular Ring")
    st.markdown("- **Output Head:** Softmax Classifier")
    st.markdown("---")

    device_str = "NVIDIA CUDA GPU" if cfg.DEVICE.type == 'cuda' else "CPU"
    st.info(f"⚡ Hardware Accelerator: **{device_str}**")

    ckpt_options = ["None (Initial Pretrained MobileNetV2 + Untrained VQC)"]
    if os.path.exists(cfg.CKPT_DIR):
        files = [f for f in os.listdir(cfg.CKPT_DIR) if f.endswith('.pth')]
        ckpt_options.extend(files)

    selected_ckpt = st.selectbox("Select Checkpoint", ckpt_options)

# Header
st.markdown('<div class="main-title">🌿 QAgroNet: Quantum Deep Learning for Agriculture</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid MobileNetV2 + Variational Quantum Circuit (VQC) for Blackgram Disease Recognition</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Leaf Diagnostics & Inference", "⚛️ Quantum Circuit Architecture", "📊 Benchmark Performance"])

@st.cache_resource
def load_model(ckpt_name):
    model = QAgroNet(num_classes=len(BPLD_CLASSES), config=cfg).to(cfg.DEVICE)
    classes = BPLD_CLASSES
    if ckpt_name and ckpt_name != "None (Initial Pretrained MobileNetV2 + Untrained VQC)":
        full_path = os.path.join(cfg.CKPT_DIR, ckpt_name)
        if os.path.exists(full_path):
            ckpt = torch.load(full_path, map_location=cfg.DEVICE, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            if 'class_names' in ckpt:
                classes = ckpt['class_names']
                model = QAgroNet(num_classes=len(classes), config=cfg).to(cfg.DEVICE)
            model.load_state_dict(state_dict)
    model.eval()
    return model, classes

model, active_classes = load_model(selected_ckpt)

with tab1:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("### 📤 Upload Blackgram Leaf Image")
        uploaded_file = st.file_uploader("Choose an image (JPG / PNG)", type=["jpg", "jpeg", "png"])

        sample_choice = st.selectbox(
            "Or pick an existing sample:",
            ["None", "Sample from BPLD Dataset"]
        )

        test_img_path = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
        elif sample_choice == "Sample from BPLD Dataset":
            sample_dir = os.path.join(cfg.BPLD_ROOT, "Healthy")
            if not os.path.exists(sample_dir):
                # pick any available directory
                if os.path.exists(cfg.BPLD_ROOT) and len(os.listdir(cfg.BPLD_ROOT)) > 0:
                    first_sub = os.listdir(cfg.BPLD_ROOT)[0]
                    sample_dir = os.path.join(cfg.BPLD_ROOT, first_sub)
            if os.path.exists(sample_dir) and len(os.listdir(sample_dir)) > 0:
                test_img_path = os.path.join(sample_dir, os.listdir(sample_dir)[0])
                image = Image.open(test_img_path).convert("RGB")
                st.image(image, caption="Sample Leaf Image", use_container_width=True)
            else:
                image = None
                st.warning("No sample images found in dataset directory.")
        else:
            image = None
            st.info("Please upload a leaf image to run disease classification.")

    with col2:
        if image is not None:
            st.markdown("### 🔬 Diagnostic Report")

            tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = tf(image).unsqueeze(0).to(cfg.DEVICE)

            with torch.no_grad():
                logits, q_out = model(tensor, return_quantum_state=True)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                q_vals = q_out.squeeze(0).cpu().numpy()

            pred_idx = int(probs.argmax())
            pred_class = active_classes[pred_idx]
            confidence = float(probs[pred_idx])

            st.markdown(f"""
            <div class="metric-card">
                <span class="highlight-badge">Diagnosis</span>
                <h2 style="color: #38bdf8; margin: 0.5rem 0 0.2rem 0;">{pred_class}</h2>
                <p style="color: #94a3b8; font-size: 1.1rem;">Model Confidence: <strong>{confidence*100:.2f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Disease Probability Distribution")
            df_probs = pd.DataFrame({
                "Disease Category": active_classes,
                "Probability": probs
            }).sort_values(by="Probability", ascending=True)

            st.bar_chart(df_probs.set_index("Disease Category"), color="#38bdf8")

            st.markdown("#### ⚛️ 8-Qubit Pauli-Z Expectation Values $\\langle \\sigma_z \\rangle$")
            st.caption("Expectation values measured from the 8-wire Variational Quantum Circuit:")

            df_q = pd.DataFrame({
                "Qubit": [f"Q{i}" for i in range(cfg.N_QUBITS)],
                "Expectation Value": q_vals
            })
            st.bar_chart(df_q.set_index("Qubit"), color="#a855f7")

with tab2:
    st.markdown("### 🧬 QAgroNet Mathematical Pipeline")
    st.markdown("""
    1. **Classical Feature Extraction**:
       $$f = \\text{AdaptiveAvgPool}(\\text{MobileNetV2}(x)) \\in \\mathbb{R}^{1280}$$
    2. **Dimension Reduction & Quantum Phase Scaling**:
       $$z = \\tanh(\\text{Linear}_{128 \\to 8}(\\text{GELU}(\\text{Linear}_{1280 \\to 128}(f)))) \\times \\frac{\\pi}{2} \\in \\left[-\\frac{\\pi}{2}, \\frac{\\pi}{2}\\right]^8$$
    3. **Quantum Angle Embedding**:
       $$|\\psi_0\\rangle = \\bigotimes_{i=0}^7 R_Y(z_i) |0\\rangle$$
    4. **Parameterized Quantum Layers (4 Layers with CNOT Ring Entanglement)**:
       $$U(\\theta, \\phi) = \\prod_{l=1}^4 \\left[ \\left( \\prod_{i=0}^7 \\text{CNOT}_{i, (i+1)\\%8} \\right) \\left( \\bigotimes_{i=0}^7 R_Z(\\phi_{l, i}) R_Y(\\theta_{l, i}) \\right) \\right]$$
    5. **Quantum Observables Measurement**:
       $$q_i = \\langle \\psi(\\theta, \\phi) | \\sigma_z^{(i)} | \\psi(\\theta, \\phi) \\rangle \\in [-1.0, 1.0], \\quad i \\in \\{0, \\dots, 7\\}$$
    6. **Softmax Output Head**:
       $$\\hat{y} = \\text{Softmax}(\\text{Linear}_{64 \\to C}(\\text{GELU}(\\text{Linear}_{8 \\to 64}(q))))$$
    """)

    st.info("💡 **Key Advantage**: The non-linear Hilbert space mapping generated by entangled qubits refines class boundaries between visually subtle symptoms (e.g. Yellow Mosaic vs Leaf Crinkle) that conventional Softmax classifiers confuse.")

with tab3:
    st.markdown("### 🏆 Benchmark Comparison (From Research Abstract)")
    st.markdown("Comparison between QAgroNet and standard vision transformer baselines on Blackgram and PlantVillage:")

    metrics_data = {
        "Model": ["Swin Transformer-Tiny", "QAgroNet (Proposed Hybrid DL-Quantum)"],
        "BPLD (1,000 imgs, 5 classes)": ["0.992", "0.994 (+0.2%)"],
        "PlantVillage (54,303 imgs, 38 classes)": ["0.994", "0.996 (+0.2%)"],
        "Cross-Dataset Transfer (PV → BBLD)": ["-", "0.968"]
    }
    st.table(pd.DataFrame(metrics_data))
