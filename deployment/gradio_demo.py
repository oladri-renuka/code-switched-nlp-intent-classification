#!/usr/bin/env python3
"""
Gradio demo for code-switched intent classification.
Shows side-by-side comparison of base and fine-tuned models.
"""

import logging
from typing import Dict, Tuple

import gradio as gr
import requests
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Model cache
models_cache = {}
tokenizer_cache = None


class CodeSwitchedClassifier:
    """Local classifier for Gradio demo."""

    INTENT_LABELS = {
        0: "question",
        1: "complaint",
        2: "recommendation",
        3: "humor",
        4: "emotional",
        5: "informational",
    }

    INTENT_DESCRIPTIONS = {
        "question": "❓ Asking for information or clarification",
        "complaint": "😞 Expressing dissatisfaction or problems",
        "recommendation": "💡 Suggesting or recommending something",
        "humor": "😄 Making a joke or being funny",
        "emotional": "💭 Expressing feelings or emotions",
        "informational": "ℹ️ Providing information or facts",
    }

    def __init__(self, config: Dict):
        """Initialize classifier."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_models(self) -> None:
        """Load models."""
        global tokenizer_cache

        base_model_name = self.config["model"]["base_model"]

        # Load tokenizer
        logger.info(f"Loading tokenizer...")
        tokenizer_cache = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )

        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        try:
            models_cache["base"] = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=6,
            ).to(self.device)
            models_cache["base"].eval()
            logger.info("✓ Base model loaded")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")

        # Try to load fine-tuned model
        finetuned_path = f"{self.config['paths']['models']}/contrastive"
        try:
            logger.info(f"Loading fine-tuned model from: {finetuned_path}")
            models_cache["finetuned"] = AutoModelForSequenceClassification.from_pretrained(
                finetuned_path,
                num_labels=6,
            ).to(self.device)
            models_cache["finetuned"].eval()
            logger.info("✓ Fine-tuned model loaded")
        except Exception as e:
            logger.warning(f"Fine-tuned model not available: {e}")

    def classify(self, text: str, model_name: str = "base") -> Dict:
        """Classify intent."""
        if model_name not in models_cache:
            return None

        model = models_cache[model_name]
        tokenizer = tokenizer_cache

        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get predictions
        probs = torch.softmax(logits, dim=1)[0]
        pred_label_id = probs.argmax().item()
        confidence = probs[pred_label_id].item()

        return {
            "label_id": pred_label_id,
            "intent": self.INTENT_LABELS[pred_label_id],
            "confidence": float(confidence),
            "all_probs": {
                self.INTENT_LABELS[i]: float(p) for i, p in enumerate(probs)
            },
        }

    def detect_languages(self, text: str) -> Dict[str, float]:
        """Detect languages in text."""
        from langdetect import detect_langs

        try:
            detections = detect_langs(text)
            return {str(d).split(":")[0]: float(str(d).split(":")[1]) for d in detections}
        except Exception:
            return {"unknown": 0.0}


# Initialize classifier
classifier = CodeSwitchedClassifier(config)


def classify_text(text: str) -> Tuple[str, str, Dict, Dict, str]:
    """
    Classify text and return results for both models.

    Returns:
    - Base model prediction
    - Fine-tuned model prediction
    - Language detection
    - Confidence comparison
    - Analysis
    """
    if not text or len(text.strip()) == 0:
        return "Please enter text", "", {}, {}, "Waiting for input..."

    try:
        # Detect languages
        languages = classifier.detect_languages(text)

        # Classify with base model
        base_result = classifier.classify(text, model_name="base")

        # Classify with fine-tuned model if available
        finetuned_result = None
        if "finetuned" in models_cache:
            finetuned_result = classifier.classify(text, model_name="finetuned")

        # Format results for display
        base_pred = f"**{base_result['intent'].upper()}** (confidence: {base_result['confidence']:.2%})"

        finetuned_pred = "Model not available"
        if finetuned_result:
            finetuned_pred = f"**{finetuned_result['intent'].upper()}** (confidence: {finetuned_result['confidence']:.2%})"

        # Create analysis
        lang_str = ", ".join(
            [f"{lang} ({conf:.1%})" for lang, conf in sorted(languages.items(), key=lambda x: x[1], reverse=True)]
        )

        analysis = f"""
**Language Detection:**
- {lang_str}

**Intent Interpretations:**
- Base: {classifier.INTENT_DESCRIPTIONS.get(base_result['intent'], '')}
"""

        if finetuned_result:
            same_intent = base_result["intent"] == finetuned_result["intent"]
            agreement = "✓ Agree" if same_intent else "✗ Disagree"
            analysis += f"- Fine-tuned: {classifier.INTENT_DESCRIPTIONS.get(finetuned_result['intent'], '')}\n"
            analysis += f"\n**Model Comparison:** {agreement}"

        return base_pred, finetuned_pred, base_result["all_probs"], finetuned_result[
            "all_probs"
        ] if finetuned_result else {}, analysis

    except Exception as e:
        logger.error(f"Error: {e}")
        return f"Error: {str(e)}", "", {}, {}, ""


def create_demo() -> gr.Blocks:
    """Create Gradio interface."""
    with gr.Blocks(title="Code-Switched Intent Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌍 Code-Switched Intent Classification")
        gr.Markdown(
            """
        Classify intents in code-switched text (Hindi-English or Spanish-English).
        Compare predictions from the base XLM-RoBERTa model and our fine-tuned model with contrastive loss.
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                text_input = gr.Textbox(
                    label="Enter code-switched text",
                    placeholder="E.g., 'Bhai, naya iPhone lelu kya? Bahut mehnga ho gaya na!'",
                    lines=3,
                    interactive=True,
                )

                example_button = gr.Button("Load Example")

                gr.Markdown("### Examples")
                gr.Examples(
                    examples=[
                        "Bhai, tumhe pata hai ki naya iPhone release ho gaya? Mujhe lena hai but itna mehnga hai!",
                        "Isse ziada disappointing update kabhi nahi dekha. Devs ne kya kiya? Complete disaster!",
                        "Hey, you should totally try this new chai recipe I found. Es muy delicioso and so easy to make!",
                        "Mein sach kahun toh yeh sab bakwas hai. 😂 Indians in USA dealing with this everyday!",
                        "I miss home so much, yaar. The weather here no se compara with back home.",
                    ],
                    inputs=[text_input],
                )

            with gr.Column(scale=1):
                gr.Markdown("### Results")

                with gr.Tabs():
                    with gr.TabItem("Base Model"):
                        base_pred = gr.Textbox(label="Prediction", interactive=False)
                        base_chart = gr.BarPlot(
                            x="intent",
                            y="confidence",
                            title="Base Model Confidence",
                            interactive=False,
                        )

                    with gr.TabItem("Fine-Tuned Model"):
                        finetuned_pred = gr.Textbox(label="Prediction", interactive=False)
                        finetuned_chart = gr.BarPlot(
                            x="intent",
                            y="confidence",
                            title="Fine-Tuned Model Confidence",
                            interactive=False,
                        )

                    with gr.TabItem("Analysis"):
                        analysis = gr.Markdown(label="Analysis")

        # Event handlers
        def on_text_change(text):
            """Update predictions when text changes."""
            base_p, ft_p, base_probs, ft_probs, analysis_text = classify_text(text)

            # Convert probs to dataframe for plotting
            import pandas as pd

            base_df = pd.DataFrame(list(base_probs.items()), columns=["intent", "confidence"])
            ft_df = pd.DataFrame(list(ft_probs.items()), columns=["intent", "confidence"]) if ft_probs else pd.DataFrame()

            return {
                base_pred: base_p,
                base_chart: base_df if not base_df.empty else None,
                finetuned_pred: ft_p,
                finetuned_chart: ft_df if not ft_df.empty else None,
                analysis: analysis_text,
            }

        text_input.change(
            fn=on_text_change,
            inputs=[text_input],
            outputs=[base_pred, base_chart, finetuned_pred, finetuned_chart, analysis],
        )

        gr.Markdown(
            """
        ---
        **About this demo:**
        - **Base Model**: XLM-RoBERTa-large (no fine-tuning)
        - **Fine-Tuned Model**: XLM-RoBERTa-large + Supervised Contrastive Loss
        - **Task**: Intent classification (question, complaint, recommendation, humor, emotional, informational)
        - **Languages**: Hindi-English (en-hi) and Spanish-English (en-es)
        """
        )

    return demo


if __name__ == "__main__":
    logger.info("Loading classifier models...")
    classifier.load_models()

    logger.info("Creating Gradio interface...")
    demo = create_demo()

    logger.info(
        f"Launching Gradio at http://{config['deployment']['gradio']['host']}:{config['deployment']['gradio']['port']}"
    )
    demo.launch(
        server_name=config["deployment"]["gradio"]["host"],
        server_port=config["deployment"]["gradio"]["port"],
        share=config["deployment"]["gradio"]["share"],
    )
