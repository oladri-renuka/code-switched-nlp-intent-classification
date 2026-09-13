#!/usr/bin/env python3
"""
Complete data preparation pipeline: load → verify → clean → ready for training.
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml
from langdetect import detect_langs
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """Complete data preparation for training."""

    INTENT_LABELS = {
        "negative": "complaint",
        "neutral": "informational",
        "positive": "recommendation",
    }

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.raw_dir = Path(self.config["paths"]["raw_data"])
        self.processed_dir = Path(self.config["paths"]["processed_data"])
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

    def load_semeval(self, semeval_dir: str) -> List[Dict]:
        """Load SemEval TSV files."""
        logger.info(f"Loading SemEval data from {semeval_dir}...")

        items = []
        processed_dir = Path(semeval_dir) / "Fully Processed Datasets"

        splits = {
            "train": processed_dir / "FinalTrainingOnly.tsv",
            "validation": processed_dir / "ValidationOnly.tsv",
            "test": processed_dir / "FinalTest.tsv",
        }

        for split_name, file_path in splits.items():
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue

                    item_id, text, sentiment_label = parts[0], parts[1], parts[2].strip()

                    if sentiment_label not in self.INTENT_LABELS:
                        continue

                    items.append({
                        "id": f"semeval_{split_name}_{item_id}",
                        "text": text,
                        "intent": self.INTENT_LABELS[sentiment_label],
                        "sentiment": sentiment_label,
                        "source": "SemEval2020-Task9",
                        "split": split_name,
                    })

        logger.info(f"✓ Loaded {len(items)} items")
        return items

    def verify_codeswitching(self, items: List[Dict]) -> List[Dict]:
        """Keep only multilingual text (code-switched)."""
        logger.info("Verifying code-switching...")

        verified = []
        for item in tqdm(items, desc="Verifying"):
            text = item.get("text", "")
            try:
                langs = {str(lang).split("-")[0] for lang in detect_langs(text)}
                if len(langs) >= 2:
                    verified.append(item)
            except:
                continue

        logger.info(f"✓ {len(verified)} verified ({len(verified)/len(items)*100:.1f}%)")
        return verified

    def deduplicate_and_clean(self, items: List[Dict], min_words: int = 3) -> List[Dict]:
        """Remove duplicates and short texts."""
        logger.info("Cleaning & deduplicating...")

        cleaned = []
        seen_texts = set()
        duplicates = 0
        short_texts = 0

        for item in items:
            text = item.get("text", "").strip()

            if text in seen_texts:
                duplicates += 1
                continue
            seen_texts.add(text)

            if len(text.split()) < min_words:
                short_texts += 1
                continue

            cleaned.append(item)

        logger.info(f"✓ Kept: {len(cleaned)}")
        logger.info(f"  Removed {duplicates} duplicates, {short_texts} short texts")
        return cleaned

    def print_quality_report(self, items: List[Dict]):
        """Print data quality summary."""
        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY REPORT")
        logger.info("="*60)

        logger.info(f"Total samples: {len(items)}")

        intents = Counter(item.get("intent") for item in items)
        logger.info("\nIntent distribution:")
        for intent, count in intents.most_common():
            pct = (count / len(items)) * 100
            logger.info(f"  {intent}: {count} ({pct:.1f}%)")

        lengths = [len(item.get("text", "").split()) for item in items]
        logger.info(f"\nText length (words):")
        logger.info(f"  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)/len(lengths):.1f}")
        logger.info("="*60 + "\n")

    def save_data(self, items: List[Dict], filename: str = "cleaned_codeswitched.jsonl"):
        """Save prepared data."""
        output_path = self.processed_dir / filename
        with open(output_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        logger.info(f"✓ Saved to: {output_path}")
        return output_path


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python prepare_data.py <semeval_dir>")
        logger.error("Example: python prepare_data.py SemEval2020-Task9")
        sys.exit(1)

    semeval_dir = sys.argv[1]

    pipeline = DataPreparationPipeline()

    logger.info("="*60)
    logger.info("DATA PREPARATION PIPELINE")
    logger.info("="*60 + "\n")

    # Step 1: Load
    items = pipeline.load_semeval(semeval_dir)

    # Step 2: Verify code-switching
    items = pipeline.verify_codeswitching(items)

    # Step 3: Clean & deduplicate
    items = pipeline.deduplicate_and_clean(items)

    # Step 4: Quality report
    pipeline.print_quality_report(items)

    # Step 5: Save
    pipeline.save_data(items)

    logger.info("✓ Data preparation complete! Ready for training.")


if __name__ == "__main__":
    main()
