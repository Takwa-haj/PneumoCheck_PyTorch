"""
app.py — PneumoCheck Streamlit (PyTorch)
Détection de pneumonie avec visualisation Grad-CAM interactive.

Lancement : streamlit run app/app.py
"""
import streamlit as st
import numpy as np
import torch
from PIL import Image
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from gradcam import gradcam_for_image

# ── Configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title  = "PneumoCheck 🫁",
    page_icon   = "🫁",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

CLASSES      = ['NORMAL', 'BACTERIA', 'VIRUS']
CLASS_ICONS  = {'NORMAL': '✅', 'BACTERIA': '🦠', 'VIRUS': '🦠'}
CLASS_COLORS = {'NORMAL': '#2ecc71', 'BACTERIA': '#e74c3c', 'VIRUS': '#e74c3c'}
MODEL_PATH   = 'models/resnet50_best.pt'
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
.big-pred { font-size:2.4rem; font-weight:bold; text-align:center; padding:16px 0; }
.card     { background:#f8f9fa; border-radius:12px; padding:20px; margin:8px 0; }
.warning  { background:#fff3cd; border-left:4px solid #f39c12;
            padding:12px; border-radius:6px; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("🫁 PneumoCheck")
    st.markdown("**Détection de pneumonie par Deep Learning**")
    st.markdown("---")
    st.markdown("### Framework")
    st.markdown("🔥 **PyTorch** + torchvision")
    st.markdown("### Modèle")
    st.markdown("ResNet50 — Transfer Learning (ImageNet)")
    st.markdown("**Test accuracy** : 95.2%")
    st.markdown("**AUC-ROC moyen** : 0.986")
    st.markdown("---")
    st.markdown("### Classes")
    for cls, icon in CLASS_ICONS.items():
        st.markdown(f"{icon} **{cls}**")
    st.markdown("---")
    threshold    = st.slider("Seuil de confiance (%)", 50, 95, 70, 5)
    show_gradcam = st.checkbox("Afficher Grad-CAM", value=True)
    st.markdown("---")
    st.markdown(
        '<div class="warning">⚠️ Cet outil est à but éducatif uniquement. '
        'Il ne remplace pas un diagnostic médical professionnel.</div>',
        unsafe_allow_html=True)

# ── Chargement du modèle ─────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    from models import ResNet50Transfer
    model = ResNet50Transfer(unfreeze_top=3).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model

model = load_model()

# ── Interface principale ──────────────────────────────────────────
st.title("🫁 PneumoCheck — Détection de Pneumonie (PyTorch)")
st.markdown("Uploadez une radiographie thoracique pour obtenir une prédiction instantanée.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded = st.file_uploader(
        "📤 Radiographie thoracique (JPG / PNG)",
        type=['jpg', 'jpeg', 'png'])
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption="Radiographie uploadée", use_column_width=True)

with col2:
    if uploaded and model:
        with st.spinner("Analyse PyTorch en cours..."):
            tmp_path = "/tmp/pneumocheck_upload.jpg"
            image.save(tmp_path)
            result = gradcam_for_image(tmp_path, model, device=str(DEVICE))

        pred  = result['prediction']
        conf  = result['confidence']
        probs = result['probabilities']
        color = CLASS_COLORS.get(pred, '#333')
        icon  = CLASS_ICONS.get(pred, '❓')

        st.markdown(
            f'<div class="big-pred" style="color:{color}">'
            f'{icon} {pred}<br>'
            f'<span style="font-size:1.2rem">Confiance : {conf:.1%}</span></div>',
            unsafe_allow_html=True)
        st.progress(conf)

        if conf < threshold / 100:
            st.warning(f"⚠️ Confiance ({conf:.1%}) < seuil ({threshold}%). "
                       "Consultez un professionnel de santé.")

        st.markdown("#### Probabilités par classe")
        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{CLASS_ICONS[cls]} {cls}**")
            st.progress(prob)
            st.caption(f"{prob:.2%}")

        if show_gradcam:
            st.markdown("#### 🗺️ Grad-CAM — Zones d'attention (PyTorch hooks)")
            st.image(result['overlay'],
                     caption="Zones influençant la décision (rouge = forte activation)",
                     use_column_width=True)
            st.caption("Implémenté via register_forward_hook + register_full_backward_hook")

    elif uploaded and not model:
        st.error("❌ Modèle non trouvé (`models/resnet50_best.pt`).\n"
                 "Lancez d'abord : `python src/train.py --model resnet50`")
    else:
        st.info("📤 Uploadez une radiographie pour commencer l'analyse.")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption(f"PneumoCheck (PyTorch) — Deep Learning 2024–2025 · M. Abdallah Khemais · "
           f"Device: {DEVICE}")
