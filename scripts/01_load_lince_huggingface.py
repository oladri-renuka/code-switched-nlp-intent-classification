#!/usr/bin/env python3
"""
Load LinCE benchmark data directly from HuggingFace Datasets.

This is cleaner than CSV files and handles data properly.
Available datasets:
- lid_hineng (Hindi-English)
- lid_spaeng (Spanish-English)
- ner_hineng (Hindi-English NER)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LinCEHuggingFaceLoader:
    """Load LinCE data from HuggingFace Datasets Hub."""

    # Available datasets on HuggingFace
    DATASETS = {
        "lid_hineng": {
            "name": "Hindi-English Language ID",
            "total_size": "~7400 examples",
        },
        "lid_spaeng": {
            "name": "Spanish-English Language ID",
            "total_size": "~32600 examples",
        },
        "lid_nepeng": {
            "name": "Nepali-English Language ID",
            "total_size": "~13000 examples",
        },
        "ner_hineng": {
            "name": "Hindi-English NER",
            "total_size": "~2000 examples",
        },
    }

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize HuggingFace loader."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_data_dir = Path(self.config["paths"]["raw_data"])
        self.raw_data_dir.mkdir(exist_ok=True)

    def load_dataset_from_huggingface(self, dataset_name: str) -> List[Dict]:
        """
        Load dataset from HuggingFace.

        Args:
            dataset_name: "lid_hineng", "lid_spaeng", "ner_hineng", etc.

        Returns:
            List of examples
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info(f"Loading {self.DATASETS[dataset_name]['name']} from HuggingFace...")

        try:
            # Load from HuggingFace
            dataset = load_dataset("lince-benchmark/lince", dataset_name)
            logger.info(f"✓ Successfully loaded from HuggingFace")

            examples = []

            # Process all splits (train, validation, test)
            for split in dataset.keys():
                logger.info(f"  Processing {split} split ({len(dataset[split])} examples)...")

                for idx, item in enumerate(dataset[split]):
                    # Reconstruct sentence from tokens
                    text = " ".join(item["words"])

                    example = {
                        "id": f"lince_{dataset_name}_{split}_{idx:05d}",
                        "text": text,
                        "words": item["words"],
                        "source": "LinCE-HuggingFace",
                        "dataset": dataset_name,
                        "split": split,
                    }

                    # Add language labels if present
                    if "lid" in item:
                        example["lid_labels"] = item["lid"]

                    # Add NER labels if present
                    if "ner" in item:
                        example["ner_labels"] = item["ner"]

                    examples.append(example)

            logger.info(f"  ✓ Loaded {len(examples)} examples total from {dataset_name}")
            return examples

        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {e}")
            logger.info(
                "Make sure you have the datasets library: pip install datasets"
            )
            return []

    def load_multiple_datasets(self, dataset_names: List[str]) -> List[Dict]:
        """Load multiple datasets and combine them."""
        logger.info(f"\nLoading {len(dataset_names)} datasets from HuggingFace...")

        all_examples = []

        for dataset_name in dataset_names:
            try:
                examples = self.load_dataset_from_huggingface(dataset_name)
                all_examples.extend(examples)
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {e}")

        return all_examples

    def infer_language_pair(self, dataset_name: str) -> str:
        """Infer language pair from dataset name."""
        if "hineng" in dataset_name:
            return "en-hi"
        elif "spaeng" in dataset_name:
            return "en-es"
        elif "nepeng" in dataset_name:
            return "en-ne"
        else:
            return "unknown"

    def convert_to_our_format(self, examples: List[Dict]) -> List[Dict]:
        """Convert to our standard format."""
        converted = []

        for example in examples:
            dataset_name = example.get("dataset", "")
            language_pair = self.infer_language_pair(dataset_name)

            converted_example = {
                "id": example.get("id", ""),
                "text": example.get("text", ""),
                "type": "lince_hf",
                "subreddit": f"r/{language_pair.replace('-', '')}",
                "language_pair": language_pair,
                "source": "LinCE-HuggingFace",
                "split": example.get("split", "train"),
            }

            converted.append(converted_example)

        return converted

    def save_examples(
        self,
        examples: List[Dict],
        output_filename: str = "lince_huggingface_data.jsonl",
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

        language_pairs = {}
        for example in examples:
            pair = example.get("language_pair", "unknown")
            language_pairs[pair] = language_pairs.get(pair, 0) + 1

        avg_length = sum(len(ex.get("text", "").split()) for ex in examples) / len(
            examples
        )

        return {
            "total_examples": len(examples),
            "by_language_pair": language_pairs,
            "avg_words_per_sentence": avg_length,
        }


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("LinCE HuggingFace Data Loader")
    logger.info("="*60)
    logger.info(
        "\nLoading from: https://huggingface.co/datasets/lince-benchmark/lince"
    )

    # Check if datasets library is installed
    try:
        import datasets
    except ImportError:
        logger.error("datasets library not found")
        logger.info("Install it with: pip install datasets")
        return

    loader = LinCEHuggingFaceLoader()

    # Load both Hindi-English and Spanish-English
    datasets_to_load = ["lid_hineng", "lid_spaeng"]

    logger.info(f"\nLoading {len(datasets_to_load)} datasets...")

    all_examples = loader.load_multiple_datasets(datasets_to_load)

    if not all_examples:
        logger.error("\n✗ No data loaded!")
        return

    # Convert to our format
    logger.info("\n" + "="*60)
    logger.info("Converting to standard format...")
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
    logger.info(f"By language pair: {stats.get('by_language_pair', {})}")
    logger.info(f"Avg words per sentence: {stats.get('avg_words_per_sentence', 0):.1f}")

    logger.info(f"\n✓ Data ready: {output_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify code-switching:")
    logger.info(f"     python scripts/02_verify_codeswitching.py {output_file}")
    logger.info("  2. Prepare for annotation:")
    logger.info(f"     python scripts/03_prepare_label_studio.py {output_file}")


if __name__ == "__main__":
    main()
