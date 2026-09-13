#!/usr/bin/env python3
"""Streamlit deployment for code-switched intent classification."""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from safetensors.torch import load_file
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.base_model import XLMRobertaForIntentClassification

st.set_page_config(page_title="Code-Switched Intent Classifier", layout="wide")

INTENT_LABELS = {
    0: "complaint",
    1: "recommendation",
    2: "informational",
}

@st.cache_resource
def load_models():
    """Load both models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    # Contrastive model
    checkpoint_dir = Path("models/checkpoints/contrastive")
    latest = sorted(checkpoint_dir.glob("checkpoint-*"))[-1]
    config = AutoConfig.from_pretrained("xlm-roberta-large")
    config.num_labels = 3
    contrastive_model = XLMRobertaForIntentClassification(config)
    state_dict = load_file(latest / "model.safetensors")
    contrastive_model.load_state_dict(state_dict)
    contrastive_model.to(device).eval()

    # Baseline model
    checkpoint_dir = Path("models/checkpoints/baseline")
    latest = sorted(checkpoint_dir.glob("checkpoint-*"))[-1]
    baseline_model = AutoModelForSequenceClassification.from_config(config)
    state_dict = load_file(latest / "model.safetensors")
    baseline_model.load_state_dict(state_dict)
    baseline_model.to(device).eval()

    return tokenizer, contrastive_model, baseline_model, device

def predict(text, model, tokenizer, device):
    """Get model prediction."""
    inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = logits.argmax(-1).item()
    return pred, probs.cpu().numpy()

# Title
st.title("🌍 Code-Switched Intent Classification")
st.markdown("**Hindi-English & Spanish-English Code-Switched Text**")
st.markdown("---")

# Load models
with st.spinner("Loading models..."):
    tokenizer, contrastive_model, baseline_model, device = load_models()

# Input
st.subheader("📝 Enter Code-Switched Text")
example_texts = [
    "Bhai, tumhe pata hai ki naya iPhone release ho gaya?",
    "Isse ziada disappointing update kabhi nahi dekha.",
    "You should totally try this new chai recipe!",
]
text = st.text_area(
    "Text",
    value=example_texts[0],
    height=100,
    placeholder="Enter Hindi-English or Spanish-English mixed text..."
)

if text.strip():
    # Get predictions
    contrastive_pred, contrastive_probs = predict(text, contrastive_model, tokenizer, device)
    baseline_pred, baseline_probs = predict(text, baseline_model, tokenizer, device)

    # Display results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Contrastive Model (91.6% Accuracy)")
        st.metric("Prediction", INTENT_LABELS[contrastive_pred].capitalize())
        st.bar_chart({
            "Complaint": contrastive_probs[0],
            "Recommendation": contrastive_probs[1],
            "Informational": contrastive_probs[2],
        })

    with col2:
        st.subheader("📊 Baseline Model (75.9% Accuracy)")
        st.metric("Prediction", INTENT_LABELS[baseline_pred].capitalize())
        st.bar_chart({
            "Complaint": baseline_probs[0],
            "Recommendation": baseline_probs[1],
            "Informational": baseline_probs[2],
        })

    # Agreement
    st.markdown("---")
    if contrastive_pred == baseline_pred:
        st.success(f"✅ **Models Agree**: {INTENT_LABELS[contrastive_pred].upper()}")
    else:
        st.warning(f"⚠️ **Models Disagree** - Contrastive: {INTENT_LABELS[contrastive_pred]}, Baseline: {INTENT_LABELS[baseline_pred]}")

# Examples
st.markdown("---")
st.subheader("📚 Try These Examples")
for i, example in enumerate(example_texts, 1):
    if st.button(f"Example {i}", key=f"ex_{i}"):
        st.rerun()

# Info
st.markdown("---")
st.info("""
**Model Details:**
- **Contrastive Model**: XLM-RoBERTa-large with supervised contrastive loss (T=0.07)
- **Baseline Model**: XLM-RoBERTa-large with standard cross-entropy loss
- **Dataset**: SemEval 2020 Task 9 (11.2k code-switched samples)
- **Classes**: Complaint, Recommendation, Informational
""")
