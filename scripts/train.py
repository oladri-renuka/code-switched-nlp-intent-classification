#!/usr/bin/env python3
"""
Train XLM-RoBERTa with contrastive loss for code-switched intent classification.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.base_model import load_model_for_training, load_base_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CodeSwitchedDataset:
    """Dataset for code-switched intent classification."""

    INTENT_LABELS = {
        "question": 0,
        "complaint": 1,
        "recommendation": 2,
        "humor": 3,
        "emotional": 4,
        "informational": 5,
    }

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize dataset handler."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["base_model"],
            trust_remote_code=True,
        )

    def load_annotations(self, filepath: str) -> List[Dict]:
        """Load annotated data from JSONL file."""
        items = []
        with open(filepath) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(items)} items from {filepath}")
        return items

    def prepare_dataset(
        self,
        items: List[Dict],
        use_llm_annotation: bool = False,
    ) -> Dataset:
        """Prepare HuggingFace Dataset from annotated items."""
        texts = []
        labels = []

        for item in items:
            text = item.get("text", "")
            if not text:
                continue

            intent = None

            # Get annotation (direct intent, manual annotation, or LLM)
            if "intent" in item:
                intent = item.get("intent", "").lower()
            elif use_llm_annotation and "llm_annotation" in item:
                annotation = item["llm_annotation"]
                intent = annotation.get("intent", "").lower()
            elif "annotation" in item:
                annotation = item["annotation"]
                intent = annotation.get("intent", "").lower()

            if not intent or intent not in self.INTENT_LABELS:
                continue

            texts.append(text)
            labels.append(self.INTENT_LABELS[intent])

        logger.info(f"Prepared {len(texts)} examples for training")

        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.config["model"]["data"]["max_seq_length"],
            padding=True,
            return_tensors="pt",
        )

        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        })

        return dataset

    def create_data_collator(self):
        """Create data collator for batching."""

        def collate_fn(batch):
            return {
                "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in batch]),
                "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in batch]),
                "labels": torch.tensor([item["labels"] for item in batch]),
            }

        return collate_fn


class ModelTrainer:
    """Train and evaluate code-switched intent classification models."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model_dir = Path(self.config["paths"]["models"])
        self.model_dir.mkdir(exist_ok=True)

    def compute_metrics(self, pred):
        """Compute classification metrics."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train_with_contrastive_loss(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = "models/checkpoints/contrastive",
    ) -> Dict:
        """Train model with contrastive loss."""
        logger.info("="*60)
        logger.info("Training with Contrastive Loss")
        logger.info("="*60)

        model = load_model_for_training(
            model_name=self.config["model"]["base_model"],
            num_intent_labels=len(CodeSwitchedDataset.INTENT_LABELS),
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(self.config["model"]["training"]["epochs"]),
            per_device_train_batch_size=int(self.config["model"]["training"]["batch_size"]),
            per_device_eval_batch_size=int(self.config["model"]["training"]["batch_size"]),
            learning_rate=float(self.config["model"]["training"]["learning_rate"]),
            weight_decay=float(self.config["model"]["training"]["weight_decay"]),
            warmup_steps=int(self.config["model"]["training"]["warmup_steps"]),
            max_grad_norm=float(self.config["model"]["training"]["max_grad_norm"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=self.config["model"]["training"]["early_stopping"]["patience"],
            )],
        )

        result = trainer.train()

        logger.info(f"Training complete!")
        logger.info(f"Final loss: {result.training_loss:.4f}")

        return {"trainer": trainer, "result": result}

    def train_baseline(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = "models/checkpoints/baseline",
    ) -> Dict:
        """Train baseline model (no contrastive loss)."""
        logger.info("="*60)
        logger.info("Training Baseline Model (No Contrastive Loss)")
        logger.info("="*60)

        model = load_base_model(
            model_name=self.config["model"]["base_model"],
            num_intent_labels=len(CodeSwitchedDataset.INTENT_LABELS),
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=int(self.config["model"]["training"]["epochs"]),
            per_device_train_batch_size=int(self.config["model"]["training"]["batch_size"]),
            per_device_eval_batch_size=int(self.config["model"]["training"]["batch_size"]),
            learning_rate=float(self.config["model"]["training"]["learning_rate"]),
            weight_decay=float(self.config["model"]["training"]["weight_decay"]),
            warmup_steps=int(self.config["model"]["training"]["warmup_steps"]),
            max_grad_norm=float(self.config["model"]["training"]["max_grad_norm"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir="logs",
            logging_steps=100,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        result = trainer.train()

        logger.info(f"Training complete!")
        logger.info(f"Final loss: {result.training_loss:.4f}")

        return {"trainer": trainer, "result": result}


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Code-Switched Intent Classification Training")
    logger.info("="*60)

    # Setup
    config_path = "config.yaml"
    dataset_handler = CodeSwitchedDataset(config_path)
    trainer = ModelTrainer(config_path)

    # Load cleaned data
    annotated_file = "data/processed/cleaned_codeswitched.jsonl"

    if not Path(annotated_file).exists():
        logger.error(f"Cleaned data file not found: {annotated_file}")
        logger.info("Please run data preparation scripts first:")
        logger.info("  1. python scripts/01_load_semeval_data.py SemEval2020-Task9")
        logger.info("  2. python scripts/02_verify_codeswitching.py data/raw/semeval_codeswitched_data.jsonl")
        logger.info("  3. python scripts/06_deduplicate_and_clean.py")
        return

    # Load and split data
    items = dataset_handler.load_annotations(annotated_file)

    # Split into train/val/test (80/10/10)
    np.random.shuffle(items)
    train_size = int(0.8 * len(items))
    val_size = int(0.1 * len(items))

    train_items = items[:train_size]
    val_items = items[train_size : train_size + val_size]
    test_items = items[train_size + val_size :]

    logger.info(f"Dataset split: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

    # Prepare datasets
    train_dataset = dataset_handler.prepare_dataset(train_items)
    val_dataset = dataset_handler.prepare_dataset(val_items)
    test_dataset = dataset_handler.prepare_dataset(test_items)

    # Train with contrastive loss
    contrastive_result = trainer.train_with_contrastive_loss(train_dataset, val_dataset)

    # Train baseline
    baseline_result = trainer.train_baseline(train_dataset, val_dataset)

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Models saved to: {trainer.model_dir}")


if __name__ == "__main__":
    main()
