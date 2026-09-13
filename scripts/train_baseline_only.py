#!/usr/bin/env python3
"""Train only baseline model (skip contrastive). Use this after contrastive training is done."""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from scripts.train import CodeSwitchedDataset, ModelTrainer
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("Baseline Model Training Only")
    logger.info("="*60)

    config_path = "config.yaml"
    dataset_handler = CodeSwitchedDataset(config_path)
    trainer = ModelTrainer(config_path)

    annotated_file = "data/processed/cleaned_codeswitched.jsonl"

    if not Path(annotated_file).exists():
        logger.error(f"Data file not found: {annotated_file}")
        return

    items = dataset_handler.load_annotations(annotated_file)

    np.random.shuffle(items)
    train_size = int(0.8 * len(items))
    val_size = int(0.1 * len(items))

    train_items = items[:train_size]
    val_items = items[train_size : train_size + val_size]

    logger.info(f"Dataset split: train={len(train_items)}, val={len(val_items)}")

    train_dataset = dataset_handler.prepare_dataset(train_items)
    val_dataset = dataset_handler.prepare_dataset(val_items)

    logger.info("\n" + "="*60)
    logger.info("Training Baseline Model (No Contrastive Loss)")
    logger.info("="*60)

    baseline_result = trainer.train_baseline(train_dataset, val_dataset)

    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Models saved to: {trainer.model_dir}")

if __name__ == "__main__":
    main()
