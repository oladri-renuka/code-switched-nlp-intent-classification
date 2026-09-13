#!/usr/bin/env python3
"""
Load LinCE (Language-specific Code-switching Evaluation) benchmark data.

LinCE provides real code-switched data for Hindi-English and Spanish-English.
Download from: https://ritual.uh.edu/lince/
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlopen

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LinCEDataLoader:
    """Load LinCE benchmark code-switched data."""

    # LinCE datasets available
    LINCE_DATASETS = {
        "hi_en": {
            "name": "Hindi-English",
            "train": "https://raw.githubusercontent.com/code-switching-challenge/CS-SemEval2022/main/data/hi_en/hi_en_train.txt",
            "test": "https://raw.githubusercontent.com/code-switching-challenge/CS-SemEval2022/main/data/hi_en/hi_en_test.txt",
        },
        "es_en": {
            "name": "Spanish-English",
            "train": "https://raw.githubusercontent.com/code-switching-challenge/CS-SemEval2022/main/data/es_en/es_en_train.txt",
            "test": "https://raw.githubusercontent.com/code-switching-challenge/CS-SemEval2022/main/data/es_en/es_en_test.txt",
        },
    }

    # Alternative: Direct LinCE format if available locally
    LINCE_LOCAL_PATHS = {
        "hi_en": {
            "name": "Hindi-English",
            "train": "data/lince/hi_en_train.conll",
            "test": "data/lince/hi_en_test.conll",
        },
        "es_en": {
            "name": "Spanish-English",
            "train": "data/lince/es_en_train.conll",
            "test": "data/lince/es_en_test.conll",
        },
    }

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize LinCE loader."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_data_dir = Path(self.config["paths"]["raw_data"])
        self.raw_data_dir.mkdir(exist_ok=True)

    def download_dataset(self, language_pair: str = "hi_en") -> str:
        """
        Download LinCE dataset from GitHub.

        Args:
            language_pair: "hi_en" or "es_en"

        Returns:
            Path to downloaded file
        """
        if language_pair not in self.LINCE_DATASETS:
            raise ValueError(f"Unknown language pair: {language_pair}")

        dataset_info = self.LINCE_DATASETS[language_pair]
        logger.info(f"Downloading {dataset_info['name']} dataset from LinCE...")

        output_file = self.raw_data_dir / f"lince_{language_pair}_raw.jsonl"

        examples = []

        for split in ["train", "test"]:
            url = dataset_info[split]
            logger.info(f"  Downloading {split} split from {url}...")

            try:
                with urlopen(url) as response:
                    lines = response.read().decode("utf-8").split("\n")

                    for i, line in enumerate(lines):
                        if not line.strip():
                            continue

                        # Parse LinCE format (typically: text \t labels)
                        parts = line.strip().split("\t")
                        if len(parts) >= 1:
                            text = parts[0]

                            examples.append({
                                "id": f"lince_{language_pair}_{split}_{i:05d}",
                                "text": text,
                                "type": "lince_data",
                                "language_pair": language_pair,
                                "split": split,
                                "subreddit": f"r/{language_pair.replace('_', '-')}",
                                "source": "LinCE",
                            })

                logger.info(f"    ✓ Downloaded {len([x for x in examples if x['split'] == split])} examples")

            except Exception as e:
                logger.error(f"Failed to download {split}: {e}")
                logger.info("Alternative: Download manually from https://ritual.uh.edu/lince/")

        logger.info(f"Total examples loaded: {len(examples)}")

        # Save to JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"Saved to: {output_file}")
        return str(output_file)

    def load_local_conll(self, conll_path: str) -> List[Dict]:
        """
        Load LinCE data from local CoNLL format file.

        CoNLL format:
        - Each sentence on one line
        - Tokens and labels separated by space
        """
        logger.info(f"Loading LinCE data from {conll_path}")

        examples = []
        sentence_tokens = []
        sentence_id = 0

        with open(conll_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    # End of sentence
                    if sentence_tokens:
                        text = " ".join(sentence_tokens)
                        examples.append({
                            "id": f"lince_{sentence_id:05d}",
                            "text": text,
                            "type": "lince_conll",
                            "source": "LinCE",
                        })
                        sentence_id += 1
                        sentence_tokens = []
                else:
                    # Parse token (format: token label or just token)
                    parts = line.split()
                    if parts:
                        token = parts[0]
                        sentence_tokens.append(token)

        logger.info(f"Loaded {len(examples)} examples from CoNLL")
        return examples

    def load_from_json(self, json_path: str) -> List[Dict]:
        """Load LinCE data from JSON file if available."""
        logger.info(f"Loading LinCE data from {json_path}")

        examples = []

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                examples = data
            elif isinstance(data, dict):
                # Handle nested structure
                if "data" in data:
                    examples = data["data"]
                elif "examples" in data:
                    examples = data["examples"]

        logger.info(f"Loaded {len(examples)} examples")
        return examples

    def convert_to_our_format(self, examples: List[Dict]) -> List[Dict]:
        """Convert LinCE examples to our standard format."""
        converted = []

        for i, example in enumerate(examples):
            # Handle different LinCE formats
            text = example.get("text") or example.get("content") or ""

            if not text:
                continue

            converted_example = {
                "id": example.get("id", f"lince_{i:05d}"),
                "text": text,
                "type": "post",
                "subreddit": example.get("subreddit", "r/lince"),
                "language_pair": example.get("language_pair", "en-hi"),
                "source": "LinCE",
                "timestamp": int(__import__("time").time()),
            }

            converted.append(converted_example)

        return converted

    def save_examples(self, examples: List[Dict], output_filename: str = None) -> str:
        """Save examples to JSONL file."""
        if not output_filename:
            output_filename = "lince_codeswitched_data.jsonl"

        output_path = self.raw_data_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(examples)} examples to {output_path}")
        return str(output_path)

    def get_statistics(self, examples: List[Dict]) -> Dict:
        """Get dataset statistics."""
        if not examples:
            return {}

        language_pairs = {}
        for example in examples:
            pair = example.get("language_pair", "unknown")
            language_pairs[pair] = language_pairs.get(pair, 0) + 1

        return {
            "total_examples": len(examples),
            "by_language_pair": language_pairs,
            "avg_text_length": sum(len(ex.get("text", "")) for ex in examples) / len(examples),
        }


def main():
    """Main entry point."""
    import sys

    logger.info("="*60)
    logger.info("LinCE Code-Switched Data Loader")
    logger.info("="*60)

    loader = LinCEDataLoader()

    # Check if local files exist
    hi_en_path = Path("data/lince/hi_en_train.conll")
    es_en_path = Path("data/lince/es_en_train.conll")

    all_examples = []

    # Option 1: Load from local files
    if hi_en_path.exists():
        logger.info(f"\nLoading Hindi-English from {hi_en_path}")
        hi_en_examples = loader.load_local_conll(str(hi_en_path))
        all_examples.extend(hi_en_examples)
    else:
        logger.info(
            "\nNo local Hindi-English file found at data/lince/hi_en_train.conll"
        )
        logger.info("To use local files, download from https://ritual.uh.edu/lince/")

    if es_en_path.exists():
        logger.info(f"\nLoading Spanish-English from {es_en_path}")
        es_en_examples = loader.load_local_conll(str(es_en_path))
        all_examples.extend(es_en_examples)
    else:
        logger.info(
            "\nNo local Spanish-English file found at data/lince/es_en_train.conll"
        )

    # Option 2: Download from GitHub (if available)
    if not all_examples:
        logger.info("\n" + "="*60)
        logger.info("Attempting to download from GitHub...")
        logger.info("="*60)

        for lang_pair in ["hi_en", "es_en"]:
            try:
                output = loader.download_dataset(lang_pair)
                logger.info(f"✓ Downloaded {lang_pair}: {output}")

                # Load downloaded data
                examples = []
                with open(output) as f:
                    for line in f:
                        examples.append(json.loads(line))
                all_examples.extend(examples)

            except Exception as e:
                logger.warning(f"Could not download {lang_pair}: {e}")

    if not all_examples:
        logger.warning("\n" + "="*60)
        logger.warning("No data found!")
        logger.warning("="*60)
        logger.info("\nTo get LinCE data:")
        logger.info("1. Download from: https://ritual.uh.edu/lince/")
        logger.info("2. Place files in: data/lince/")
        logger.info("   - hi_en_train.conll")
        logger.info("   - es_en_train.conll")
        logger.info("   (or es_en_train.txt, hi_en_train.txt)")
        logger.info("3. Re-run this script")
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
    logger.info(f"Average text length: {stats.get('avg_text_length', 0):.0f} characters")

    logger.info(f"\n✓ Data ready: {output_file}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify code-switching:")
    logger.info(f"     python scripts/02_verify_codeswitching.py {output_file}")
    logger.info("  2. Prepare for annotation:")
    logger.info(f"     python scripts/03_prepare_label_studio.py {output_file}")


if __name__ == "__main__":
    main()
