#!/usr/bin/env python3
"""
Load LinCE benchmark data from CSV format (as found on Kaggle/HuggingFace).

Sources:
1. Kaggle: https://www.kaggle.com/datasets/huggingface/lince
2. HuggingFace: https://huggingface.co/datasets/HiTZ/lince

CSV files include:
- lid_hineng_train.csv, lid_hineng_test.csv (Hindi-English Language ID)
- lid_spaeng_train.csv, lid_spaeng_test.csv (Spanish-English Language ID)
- pos_hineng_train.csv, pos_hineng_test.csv (Hindi-English POS)
- pos_spaeng_train.csv, pos_spaeng_test.csv (Spanish-English POS)
- ner_hineng_train.csv, ner_hineng_test.csv (Hindi-English NER)
- ner_spaeng_train.csv, ner_spaeng_test.csv (Spanish-English NER)
- sa_spaeng_train.csv, sa_spaeng_test.csv (Spanish-English Sentiment)
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LinCECSVLoader:
    """Load LinCE data from CSV files (Kaggle/HuggingFace format)."""

    # CSV file patterns
    CSV_FILES = {
        "hi_en": {
            "name": "Hindi-English",
            "files": [
                "lid_hineng_train.csv",
                "lid_hineng_test.csv",
                "lid_hineng_validation.csv",
            ],
        },
        "es_en": {
            "name": "Spanish-English",
            "files": [
                "lid_spaeng_train.csv",
                "lid_spaeng_test.csv",
                "lid_spaeng_validation.csv",
            ],
        },
    }

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize CSV loader."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_data_dir = Path(self.config["paths"]["raw_data"])
        self.raw_data_dir.mkdir(exist_ok=True)

    def load_csv_file(self, csv_path: str) -> List[Dict]:
        """
        Load single CSV file.

        Expected format:
        - Column 'words': space-separated tokens
        - Optional columns: 'lid', 'pos', 'ner', 'sa' (labels)
        """
        logger.info(f"Loading CSV: {csv_path}")

        examples = []
        sentences = defaultdict(lambda: {"tokens": [], "labels": {}})

        try:
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)

                if not reader.fieldnames:
                    logger.warning(f"Empty CSV file: {csv_path}")
                    return examples

                for row_idx, row in enumerate(reader):
                    # Get text/words
                    text = row.get("words", "").strip()

                    if not text:
                        continue

                    # Extract labels if present
                    label_dict = {}
                    if "lid" in row:
                        label_dict["lid"] = row["lid"]
                    if "pos" in row:
                        label_dict["pos"] = row["pos"]
                    if "ner" in row:
                        label_dict["ner"] = row["ner"]
                    if "sa" in row:
                        label_dict["sa"] = row["sa"]

                    example = {
                        "id": f"lince_{Path(csv_path).stem}_{row_idx:05d}",
                        "text": text,
                        "source": "LinCE",
                        "file": Path(csv_path).name,
                    }

                    # Add labels if present
                    if label_dict:
                        example["labels"] = label_dict

                    examples.append(example)

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return []

        logger.info(f"  ✓ Loaded {len(examples)} examples from {Path(csv_path).name}")
        return examples

    def load_language_pair(self, lince_dir: str, language_pair: str = "hi_en") -> List[Dict]:
        """
        Load all CSV files for a language pair.

        Args:
            lince_dir: Directory containing LinCE CSV files
            language_pair: "hi_en" or "es_en"

        Returns:
            List of examples
        """
        if language_pair not in self.CSV_FILES:
            raise ValueError(f"Unknown language pair: {language_pair}")

        pair_info = self.CSV_FILES[language_pair]
        lince_path = Path(lince_dir)

        logger.info(f"\nLoading {pair_info['name']} dataset...")

        all_examples = []

        for csv_file in pair_info["files"]:
            csv_path = lince_path / csv_file

            if not csv_path.exists():
                logger.warning(f"  ✗ File not found: {csv_path}")
                logger.info(f"    Make sure you have the LinCE CSV files in: {lince_path}")
                continue

            examples = self.load_csv_file(str(csv_path))
            all_examples.extend(examples)

        return all_examples

    def convert_to_our_format(self, examples: List[Dict]) -> List[Dict]:
        """
        Convert LinCE examples to our standard format.

        Our format includes intent labels which we'll add in annotation phase.
        """
        converted = []

        for example in examples:
            text = example.get("text", "")

            if not text:
                continue

            # Infer language pair from file name
            file_name = example.get("file", "")
            if "hineng" in file_name or "hin_eng" in file_name:
                language_pair = "en-hi"
                subreddit = "r/hinglish"
            elif "spaeng" in file_name or "spa_eng" in file_name:
                language_pair = "en-es"
                subreddit = "r/latinos"
            else:
                language_pair = "unknown"
                subreddit = "r/codeswitched"

            converted_example = {
                "id": example.get("id", ""),
                "text": text,
                "type": "lince_sentence",
                "subreddit": subreddit,
                "language_pair": language_pair,
                "source": "LinCE",
                "labels": example.get("labels", {}),  # Keep original labels for reference
            }

            converted.append(converted_example)

        return converted

    def save_examples(
        self,
        examples: List[Dict],
        output_filename: str = "lince_codeswitched_data.jsonl",
    ) -> str:
        """Save examples to JSONL file."""
        output_path = self.raw_data_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"\nSaved {len(examples)} examples to: {output_path}")
        return str(output_path)

    def get_statistics(self, examples: List[Dict]) -> Dict:
        """Get dataset statistics."""
        if not examples:
            return {}

        language_pairs = {}
        label_types = set()

        for example in examples:
            pair = example.get("language_pair", "unknown")
            language_pairs[pair] = language_pairs.get(pair, 0) + 1

            if "labels" in example:
                label_types.update(example["labels"].keys())

        avg_length = sum(len(ex.get("text", "").split()) for ex in examples) / len(examples)

        return {
            "total_examples": len(examples),
            "by_language_pair": language_pairs,
            "avg_tokens_per_sentence": avg_length,
            "available_label_types": sorted(list(label_types)),
        }


def main():
    """Main entry point."""
    import sys

    logger.info("="*60)
    logger.info("LinCE CSV Data Loader")
    logger.info("="*60)
    logger.info("\nSources:")
    logger.info("  - Kaggle: https://www.kaggle.com/datasets/huggingface/lince")
    logger.info("  - HuggingFace: https://huggingface.co/datasets/HiTZ/lince")

    # Get LinCE directory from command line or use default
    if len(sys.argv) > 1:
        lince_dir = sys.argv[1]
    else:
        lince_dir = "data/lince"
        logger.info(f"\nNo directory specified. Using default: {lince_dir}")
        logger.info("Usage: python scripts/01_load_lince_csv.py /path/to/lince/csv/files")

    lince_path = Path(lince_dir)

    if not lince_path.exists():
        logger.error(f"\n✗ Directory not found: {lince_dir}")
        logger.info("\nTo get LinCE data:")
        logger.info("1. Download from Kaggle: https://www.kaggle.com/datasets/huggingface/lince")
        logger.info("2. OR from HuggingFace: https://huggingface.co/datasets/HiTZ/lince")
        logger.info("3. Extract CSV files to: data/lince/")
        logger.info("4. Re-run: python scripts/01_load_lince_csv.py data/lince/")
        sys.exit(1)

    loader = LinCECSVLoader()

    # Load both language pairs
    all_examples = []

    for lang_pair in ["hi_en", "es_en"]:
        try:
            examples = loader.load_language_pair(str(lince_path), lang_pair)
            all_examples.extend(examples)
        except Exception as e:
            logger.error(f"Error loading {lang_pair}: {e}")

    if not all_examples:
        logger.error("\n✗ No data loaded!")
        logger.info("\nMake sure CSV files are in the directory:")
        logger.info("  - lid_hineng_train.csv")
        logger.info("  - lid_spaeng_train.csv")
        logger.info("  etc.")
        sys.exit(1)

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
    logger.info(f"Avg tokens per sentence: {stats.get('avg_tokens_per_sentence', 0):.1f}")
    logger.info(f"Available labels: {stats.get('available_label_types', [])}")

    logger.info(f"\n✓ Data ready: {output_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify code-switching:")
    logger.info(f"     python scripts/02_verify_codeswitching.py {output_file}")
    logger.info("  2. Prepare for annotation:")
    logger.info(f"     python scripts/03_prepare_label_studio.py {output_file}")


if __name__ == "__main__":
    main()
