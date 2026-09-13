#!/usr/bin/env python3
"""
Load SemEval 2020 Task 9 code-switched sentiment data.

This dataset contains Hindi-English code-switched text with sentiment labels.
Source: https://github.com/singhnivedita/SemEval2020-Task9

Dataset: 20,594 examples total
- Training: 14,594
- Validation: 3,000
- Test: 3,000
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SemEvalDataLoader:
    """Load SemEval 2020 Task 9 code-switched sentiment data."""

    SENTIMENT_LABELS = {
        "0": "negative",
        "1": "neutral",
        "2": "positive",
    }

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize SemEval data loader."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_data_dir = Path(self.config["paths"]["raw_data"])
        self.raw_data_dir.mkdir(exist_ok=True)

    def load_tsv_file(self, tsv_path: str, split: str = "train") -> List[Dict]:
        """
        Load TSV file from SemEval dataset.

        Format: ID \t Text \t Sentiment_Label
        """
        logger.info(f"Loading {split} data from {Path(tsv_path).name}...")

        examples = []

        try:
            with open(tsv_path, encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t")
                    if len(parts) < 3:
                        logger.warning(f"Skipping malformed line {line_idx}: {line}")
                        continue

                    item_id = parts[0]
                    text = parts[1]
                    sentiment_label = parts[2].strip()

                    # Map sentiment to label
                    if sentiment_label not in self.SENTIMENT_LABELS:
                        logger.warning(f"Unknown sentiment label: {sentiment_label}")
                        continue

                    example = {
                        "id": f"semeval_{split}_{item_id}",
                        "text": text,
                        "type": "semeval_sentiment",
                        "source": "SemEval2020-Task9",
                        "split": split,
                        "sentiment": self.SENTIMENT_LABELS[sentiment_label],
                        "sentiment_label": int(sentiment_label),
                        "language_pair": "en-hi",  # Hindi-English
                        "subreddit": "r/hinglish",
                    }

                    examples.append(example)

            logger.info(f"  ✓ Loaded {len(examples)} examples from {split} split")
            return examples

        except Exception as e:
            logger.error(f"Error loading {tsv_path}: {e}")
            return []

    def load_all_splits(self, semeval_dir: str) -> List[Dict]:
        """Load all dataset splits."""
        logger.info(f"\nLoading SemEval 2020 Task 9 data from {semeval_dir}...")

        semeval_path = Path(semeval_dir)
        processed_dir = semeval_path / "Fully Processed Datasets"

        if not processed_dir.exists():
            logger.error(f"Processed datasets directory not found: {processed_dir}")
            return []

        all_examples = []

        # Load all splits
        splits = {
            "train": processed_dir / "FinalTrainingOnly.tsv",
            "validation": processed_dir / "ValidationOnly.tsv",
            "test": processed_dir / "FinalTest.tsv",
        }

        for split_name, file_path in splits.items():
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            examples = self.load_tsv_file(str(file_path), split=split_name)
            all_examples.extend(examples)

        return all_examples

    def convert_to_our_format(self, examples: List[Dict]) -> List[Dict]:
        """
        Convert sentiment labels to intent labels.

        Mapping sentiment → intent for our classification task:
        - negative: complaint (expressing dissatisfaction)
        - neutral: informational (neutral statement)
        - positive: recommendation (positive endorsement)
        """
        sentiment_to_intent = {
            "negative": "complaint",
            "neutral": "informational",
            "positive": "recommendation",
        }

        converted = []

        for example in examples:
            sentiment = example.get("sentiment", "neutral")
            intent = sentiment_to_intent.get(sentiment, "informational")

            converted_example = {
                "id": example.get("id", ""),
                "text": example.get("text", ""),
                "type": "sentiment_to_intent",
                "subreddit": example.get("subreddit", "r/hinglish"),
                "language_pair": example.get("language_pair", "en-hi"),
                "source": "SemEval2020-Task9",
                "split": example.get("split", "train"),
                # Keep sentiment for reference
                "sentiment": sentiment,
                # Add intent derived from sentiment
                "intent": intent,
            }

            converted.append(converted_example)

        return converted

    def save_examples(
        self,
        examples: List[Dict],
        output_filename: str = "semeval_codeswitched_data.jsonl",
    ) -> str:
        """Save examples to JSONL file."""
        output_path = self.raw_data_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"✓ Saved {len(examples)} examples to: {output_path}")
        return str(output_path)

    def get_statistics(self, examples: List[Dict]) -> Dict:
        """Get dataset statistics."""
        if not examples:
            return {}

        splits = {}
        sentiments = {}

        for example in examples:
            split = example.get("split", "unknown")
            splits[split] = splits.get(split, 0) + 1

            sentiment = example.get("sentiment", "unknown")
            sentiments[sentiment] = sentiments.get(sentiment, 0) + 1

        avg_length = sum(len(ex.get("text", "").split()) for ex in examples) / len(
            examples
        )

        return {
            "total_examples": len(examples),
            "by_split": splits,
            "by_sentiment": sentiments,
            "avg_words_per_sentence": avg_length,
        }


def main():
    """Main entry point."""
    import sys

    logger.info("="*60)
    logger.info("SemEval 2020 Task 9 Data Loader")
    logger.info("="*60)

    if len(sys.argv) > 1:
        semeval_dir = sys.argv[1]
    else:
        semeval_dir = "SemEval2020-Task9"
        logger.info(f"\nUsing default directory: {semeval_dir}")

    semeval_path = Path(semeval_dir)

    if not semeval_path.exists():
        logger.error(f"\n✗ Directory not found: {semeval_dir}")
        logger.info("\nTo clone SemEval data:")
        logger.info("  git clone https://github.com/singhnivedita/SemEval2020-Task9.git")
        sys.exit(1)

    loader = SemEvalDataLoader()

    # Load all splits
    all_examples = loader.load_all_splits(str(semeval_path))

    if not all_examples:
        logger.error("\n✗ No data loaded!")
        sys.exit(1)

    # Convert sentiment to intent
    logger.info("\n" + "="*60)
    logger.info("Converting sentiment labels to intent labels...")
    logger.info("="*60)

    converted = loader.convert_to_our_format(all_examples)

    # Save
    output_file = loader.save_examples(converted)

    # Statistics
    stats = loader.get_statistics(converted)
    logger.info("\n" + "="*60)
    logger.info("Dataset Statistics")
    logger.info("="*60)
    logger.info(f"Total examples: {stats.get('total_examples', 0)}")
    logger.info(f"By split: {stats.get('by_split', {})}")
    logger.info(f"By sentiment: {stats.get('by_sentiment', {})}")
    logger.info(f"Avg words per sentence: {stats.get('avg_words_per_sentence', 0):.1f}")

    logger.info(f"\n✓ Data ready: {output_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify code-switching:")
    logger.info(f"     python scripts/02_verify_codeswitching.py {output_file}")
    logger.info("  2. Prepare for annotation:")
    logger.info(f"     python scripts/03_prepare_label_studio.py {output_file}")


if __name__ == "__main__":
    main()
