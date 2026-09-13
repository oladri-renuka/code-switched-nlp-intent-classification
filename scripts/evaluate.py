#!/usr/bin/env python3
"""Evaluate both models on test set - simple & robust."""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import json
import logging
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from safetensors.torch import load_file

from scripts.train import CodeSwitchedDataset
from models.base_model import XLMRobertaForIntentClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate trained models."""

    INTENT_LABELS = {
        "question": 0, "complaint": 1, "recommendation": 2,
        "humor": 3, "emotional": 4, "informational": 5,
    }
    ID_TO_INTENT = {v: k for k, v in INTENT_LABELS.items()}

    def __init__(self, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    def load_contrastive_model(self):
        """Load best contrastive model (last checkpoint)."""
        logger.info(f"Loading contrastive model...")
        checkpoint_dir = Path("models/checkpoints/contrastive")
        latest = sorted(checkpoint_dir.glob("checkpoint-*"))[-1]

        config = AutoConfig.from_pretrained("xlm-roberta-large")
        config.num_labels = 6
        model = XLMRobertaForIntentClassification(config)
        state_dict = load_file(latest / "model.safetensors")
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def load_baseline_model(self):
        """Load best baseline model (last checkpoint)."""
        logger.info(f"Loading baseline model...")
        checkpoint_dir = Path("models/checkpoints/baseline")
        latest = sorted(checkpoint_dir.glob("checkpoint-*"))[-1]

        config = AutoConfig.from_pretrained("xlm-roberta-large")
        config.num_labels = 6
        model = AutoModelForSequenceClassification.from_config(config)
        state_dict = load_file(latest / "model.safetensors")
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        return model

    def predict_batch(self, texts, model, model_type="contrastive"):
        """Get predictions for texts."""
        predictions = []
        for text in texts:
            inputs = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            pred = logits.argmax(-1).item()
            predictions.append(pred)
        return predictions

    def evaluate(self, test_items, contrastive_model, baseline_model):
        """Evaluate both models."""
        texts = [item["text"] for item in test_items]
        true_labels = [self.INTENT_LABELS[item["intent"]] for item in test_items]

        logger.info("\n" + "="*60)
        logger.info("EVALUATING MODELS")
        logger.info("="*60)

        logger.info("\nContrastive model predictions...")
        contrastive_preds = self.predict_batch(texts, contrastive_model, "contrastive")

        logger.info("Baseline model predictions...")
        baseline_preds = self.predict_batch(texts, baseline_model, "baseline")

        logger.info("\n" + "="*60)
        logger.info("CONTRASTIVE MODEL RESULTS")
        logger.info("="*60)
        self._print_metrics(true_labels, contrastive_preds)

        logger.info("\n" + "="*60)
        logger.info("BASELINE MODEL RESULTS")
        logger.info("="*60)
        self._print_metrics(true_labels, baseline_preds)

        logger.info("\n" + "="*60)
        logger.info("COMPARISON")
        logger.info("="*60)
        agreement = sum(1 for c, b in zip(contrastive_preds, baseline_preds) if c == b)
        logger.info(f"Model Agreement: {agreement}/{len(contrastive_preds)} ({agreement/len(contrastive_preds)*100:.1f}%)")

    def _print_metrics(self, true_labels, preds):
        """Print metrics."""
        acc = accuracy_score(true_labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(true_labels, preds, average="macro", zero_division=0)

        logger.info(f"Accuracy:  {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.info(f"F1 (macro): {f1:.4f}")

        logger.info("\nPer-class:")
        report = classification_report(true_labels, preds, labels=range(6), target_names=[self.ID_TO_INTENT[i] for i in range(6)], zero_division=0)
        logger.info("\n" + report)

def main():
    logger.info("="*60)
    logger.info("Model Evaluation")
    logger.info("="*60)

    dataset_handler = CodeSwitchedDataset()
    items = dataset_handler.load_annotations("data/processed/cleaned_codeswitched.jsonl")

    np.random.shuffle(items)
    train_size = int(0.8 * len(items))
    val_size = int(0.1 * len(items))
    test_items = items[train_size + val_size:]

    logger.info(f"Test set: {len(test_items)} samples")

    evaluator = ModelEvaluator()
    contrastive_model = evaluator.load_contrastive_model()
    baseline_model = evaluator.load_baseline_model()

    evaluator.evaluate(test_items, contrastive_model, baseline_model)
    logger.info("\n✓ Done!")

if __name__ == "__main__":
    main()
