#!/usr/bin/env python3
"""
FastAPI server for code-switched intent classification.
Supports both base and fine-tuned models for comparison.
"""

import logging
from typing import Dict, Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException
from langdetect import detect_langs
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml") as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="Code-Switched Intent Classifier",
    description="Intent classification for code-switched (Hindi-English, Spanish-English) text",
    version="1.0.0",
)

# Model storage
models = {}
tokenizer = None


class ClassificationRequest(BaseModel):
    """Request model for classification."""

    text: str = Field(..., min_length=1, max_length=500)
    compare_models: bool = Field(default=False, description="Compare base vs fine-tuned model")


class LanguageDetectionResult(BaseModel):
    """Language detection result."""

    language: str
    confidence: float


class IntentPrediction(BaseModel):
    """Intent prediction result."""

    intent: str
    confidence: float
    label_id: int


class ClassificationResponse(BaseModel):
    """Response model for classification."""

    text: str
    languages: Dict[str, float]
    base_model_prediction: Optional[IntentPrediction] = None
    finetuned_model_prediction: Optional[IntentPrediction] = None
    processing_time_ms: float


class ModelManager:
    """Manager for loading and using models."""

    INTENT_LABELS = {
        0: "question",
        1: "complaint",
        2: "recommendation",
        3: "humor",
        4: "emotional",
        5: "informational",
    }

    def __init__(self, config: Dict):
        """Initialize model manager."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def load_models(self) -> None:
        """Load base and fine-tuned models."""
        global tokenizer

        base_model_name = self.config["model"]["base_model"]

        # Load tokenizer (same for both)
        logger.info(f"Loading tokenizer: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
        )

        # Load base model
        logger.info(f"Loading base model: {base_model_name}")
        try:
            models["base"] = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                num_labels=6,  # intent classes
            ).to(self.device)
            models["base"].eval()
            logger.info("✓ Base model loaded")
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")

        # Load fine-tuned model if available
        finetuned_path = self.config["paths"]["models"] + "/contrastive/checkpoint-best"
        try:
            logger.info(f"Loading fine-tuned model from: {finetuned_path}")
            models["finetuned"] = AutoModelForSequenceClassification.from_pretrained(
                finetuned_path,
                num_labels=6,
            ).to(self.device)
            models["finetuned"].eval()
            logger.info("✓ Fine-tuned model loaded")
        except Exception as e:
            logger.warning(f"Fine-tuned model not available: {e}")
            logger.warning("Will use base model predictions only")

    def _detect_languages(self, text: str) -> Dict[str, float]:
        """Detect languages in text."""
        try:
            detections = detect_langs(text)
            return {str(d).split(":")[0]: float(str(d).split(":")[1]) for d in detections}
        except Exception:
            return {"unknown": 0.0}

    def classify(self, text: str, model_name: str = "base") -> Dict:
        """Classify intent for given text."""
        if model_name not in models:
            raise ValueError(f"Model {model_name} not available")

        model = models[model_name]

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


# Initialize manager
manager = ModelManager(config)


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Loading models...")
    manager.load_models()
    logger.info("Models loaded and ready!")


@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ClassificationRequest):
    """
    Classify intent for code-switched text.

    Returns:
    - Detected languages
    - Intent prediction(s) from model(s)
    - Confidence scores
    """
    import time

    start_time = time.time()

    try:
        # Detect languages
        languages = manager._detect_languages(request.text)

        # Classify with base model
        base_pred = manager.classify(request.text, model_name="base")

        # Optionally classify with fine-tuned model
        finetuned_pred = None
        if request.compare_models and "finetuned" in models:
            finetuned_pred = manager.classify(request.text, model_name="finetuned")

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return ClassificationResponse(
            text=request.text,
            languages=languages,
            base_model_prediction=IntentPrediction(
                intent=base_pred["intent"],
                confidence=base_pred["confidence"],
                label_id=base_pred["label_id"],
            ),
            finetuned_model_prediction=IntentPrediction(
                intent=finetuned_pred["intent"],
                confidence=finetuned_pred["confidence"],
                label_id=finetuned_pred["label_id"],
            ) if finetuned_pred else None,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
    }


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "available_models": list(models.keys()),
        "base_model": config["model"]["base_model"],
        "intent_labels": {v: k for k, v in manager.INTENT_LABELS.items()},
    }


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Code-Switched Intent Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "POST /classify": "Classify intent for code-switched text",
            "GET /health": "Health check",
            "GET /models": "List available models",
            "GET /docs": "Swagger documentation",
            "GET /redoc": "ReDoc documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config["deployment"]["fastapi"]["host"],
        port=config["deployment"]["fastapi"]["port"],
        reload=config["deployment"]["fastapi"]["reload"],
    )
