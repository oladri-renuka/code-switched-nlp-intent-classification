#!/usr/bin/env python3
"""
Verify and analyze code-switching in collected Reddit data.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import yaml
from langdetect import detect_langs
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CodeSwitchingAnalyzer:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize analyzer with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_data_dir = Path(self.config["paths"]["raw_data"])
        self.processed_data_dir = Path(self.config["paths"]["processed_data"])
        self.processed_data_dir.mkdir(exist_ok=True)

    def _detect_languages(self, text: str) -> Dict[str, float]:
        """Detect languages in text."""
        try:
            detections = detect_langs(text)
            return {str(d).split(":")[0]: float(str(d).split(":")[1]) for d in detections}
        except Exception:
            return {}

    def analyze_file(self, filepath: str) -> Dict:
        """Analyze code-switching patterns in a JSONL file."""
        logger.info(f"Analyzing {filepath}")

        stats = {
            "total_items": 0,
            "valid_codeswitched": 0,
            "invalid_items": 0,
            "language_pairs": Counter(),
            "by_subreddit": {},
            "items": [],
        }

        with open(filepath) as f:
            for line in tqdm(f, desc="Analyzing"):
                try:
                    item = json.loads(line)
                    stats["total_items"] += 1

                    text = item.get("text", "") or item.get("title", "")
                    if not text:
                        stats["invalid_items"] += 1
                        continue

                    langs = self._detect_languages(text)
                    lang_list = sorted(langs.keys())
                    lang_pair = "-".join(lang_list)

                    # Check if valid code-switching
                    # For SemEval: romanized text won't detect Hindi, so accept multilingual
                    # For standard datasets: require English + (Hindi OR Spanish)
                    source = item.get("source", "")
                    if "SemEval" in source:
                        is_valid = len(langs) >= 2  # Any multilingual text
                    else:
                        is_valid = "en" in langs and ("hi" in langs or "es" in langs)

                    if is_valid:
                        stats["valid_codeswitched"] += 1
                        stats["language_pairs"][lang_pair] += 1

                        sub = item.get("subreddit", "unknown")
                        if sub not in stats["by_subreddit"]:
                            stats["by_subreddit"][sub] = 0
                        stats["by_subreddit"][sub] += 1

                        stats["items"].append({
                            "id": item.get("id"),
                            "text": text,
                            "languages": langs,
                            "language_pair": lang_pair,
                            "subreddit": sub,
                            "type": item.get("type"),
                        })
                    else:
                        stats["invalid_items"] += 1

                except json.JSONDecodeError:
                    stats["invalid_items"] += 1
                    continue

        return stats

    def print_statistics(self, stats: Dict) -> None:
        """Print analysis statistics."""
        logger.info("="*60)
        logger.info("Code-Switching Analysis Results")
        logger.info("="*60)
        logger.info(f"Total items processed: {stats['total_items']}")
        logger.info(f"Valid code-switched: {stats['valid_codeswitched']}")
        logger.info(f"Invalid/filtered: {stats['invalid_items']}")
        logger.info(f"Valid rate: {stats['valid_codeswitched']/max(1, stats['total_items'])*100:.2f}%")

        logger.info("\nLanguage Pair Distribution:")
        for lang_pair, count in stats["language_pairs"].most_common():
            logger.info(f"  {lang_pair}: {count}")

        logger.info("\nBy Subreddit:")
        for sub, count in sorted(stats["by_subreddit"].items()):
            logger.info(f"  {sub}: {count}")

    def save_verified_data(self, stats: Dict, output_filename: str = "verified_codeswitched.jsonl") -> str:
        """Save verified code-switched data to file."""
        output_path = self.processed_data_dir / output_filename

        with open(output_path, "w") as f:
            for item in stats["items"]:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Verified data saved to {output_path}")
        logger.info(f"Total verified items: {len(stats['items'])}")

        return str(output_path)

    def generate_sample_report(self, stats: Dict, num_samples: int = 10) -> None:
        """Generate a sample report of code-switched text."""
        logger.info("\n" + "="*60)
        logger.info("Sample Code-Switched Texts")
        logger.info("="*60)

        for i, item in enumerate(stats["items"][:num_samples]):
            logger.info(f"\n[{i+1}] {item['subreddit']} - {item['language_pair']}")
            logger.info(f"Text: {item['text'][:150]}...")
            logger.info(f"Languages: {item['languages']}")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python 02_verify_codeswitching.py <input_jsonl>")
        sys.exit(1)

    input_file = sys.argv[1]

    analyzer = CodeSwitchingAnalyzer()
    stats = analyzer.analyze_file(input_file)

    analyzer.print_statistics(stats)
    analyzer.generate_sample_report(stats)

    # Save verified data
    output_file = analyzer.save_verified_data(stats)

    logger.info(f"\n✓ Analysis complete. Verified data: {output_file}")


if __name__ == "__main__":
    main()
