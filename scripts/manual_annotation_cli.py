#!/usr/bin/env python3
"""
Simple CLI tool for manual annotation of code-switched text.
Allows quick manual annotation of 100-500 samples for Cohen's kappa calculation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INTENTS = ["question", "complaint", "recommendation", "humor", "emotional", "informational"]
SENTIMENTS = ["positive", "negative", "neutral"]


class ManualAnnotator:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize manual annotator."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.annotated_data_dir = Path(self.config["paths"]["annotated_data"])
        self.annotated_data_dir.mkdir(exist_ok=True)

    def display_annotation_guide(self):
        """Show annotation guidelines."""
        print("\n" + "="*70)
        print("INTENT CLASSIFICATION GUIDE")
        print("="*70)
        print("""
0 - question: Asking for information or clarification
1 - complaint: Expressing dissatisfaction or criticism
2 - recommendation: Suggesting or endorsing something
3 - humor: Making a joke or funny comment
4 - emotional: Expressing strong emotions (happiness, sadness, anger)
5 - informational: Providing factual information or news

SENTIMENT:
p - positive: Positive tone or opinion
n - negative: Negative tone or criticism
u - neutral: Objective or neutral tone
""")
        print("="*70 + "\n")

    def annotate_sample(self, item: Dict, sample_num: int, total: int) -> Dict:
        """Annotate a single sample interactively."""
        text = item.get("text", "")[:200]  # Truncate for readability
        print(f"\n[{sample_num}/{total}] Text: {text}...")
        print(f"ID: {item.get('id', 'unknown')}")

        # Get intent
        while True:
            intent_input = input(
                "Intent (0=question, 1=complaint, 2=recommendation, 3=humor, 4=emotional, 5=informational): "
            ).strip()
            if intent_input in ["0", "1", "2", "3", "4", "5"]:
                intent = INTENTS[int(intent_input)]
                break
            print("Invalid input. Please enter 0-5.")

        # Get sentiment
        while True:
            sentiment_input = input("Sentiment (p=positive, n=negative, u=neutral): ").strip().lower()
            if sentiment_input in ["p", "n", "u"]:
                sentiment_map = {"p": "positive", "n": "negative", "u": "neutral"}
                sentiment = sentiment_map[sentiment_input]
                break
            print("Invalid input. Please enter p, n, or u.")

        # Get language pair (optional)
        lang_pair = input("Language pair (en-hi/en-es, or press Enter to skip): ").strip()
        if not lang_pair:
            lang_pair = item.get("language_pair", "")

        item["annotation"] = {
            "intent": intent,
            "sentiment": sentiment,
            "language_pair": lang_pair,
        }

        return item

    def annotate_batch(self, input_file: str, output_file: str, sample_count: int = 100, batch_display: int = 1):
        """Annotate a batch of samples interactively."""
        logger.info(f"Loading samples from {input_file}")

        items = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                if idx >= sample_count:
                    break
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(items)} samples for annotation")
        self.display_annotation_guide()

        input("Press Enter to start annotation...")

        annotated = []
        for idx, item in enumerate(items, 1):
            try:
                annotated_item = self.annotate_sample(item, idx, len(items))
                annotated.append(annotated_item)
            except KeyboardInterrupt:
                logger.info("\nAnnotation interrupted by user")
                break

        # Save annotated data
        output_path = self.annotated_data_dir / output_file
        with open(output_path, "w") as f:
            for item in annotated:
                f.write(json.dumps(item) + "\n")

        logger.info(f"\n✓ Saved {len(annotated)} annotated samples to: {output_path}")
        return str(output_path)


def main():
    """Main entry point."""
    import sys

    logger.info("="*70)
    logger.info("Manual Annotation Tool")
    logger.info("="*70)

    if len(sys.argv) < 2:
        input_file = "data/annotated/manual_annotation_items.jsonl"
        sample_count = 100
    else:
        input_file = sys.argv[1]
        sample_count = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Usage: python manual_annotation_cli.py <input_file> [sample_count]")
        sys.exit(1)

    annotator = ManualAnnotator()
    output_file = annotator.annotate_batch(
        input_file,
        "manual_annotations.jsonl",
        sample_count=sample_count,
    )

    logger.info(f"\n✓ Annotation complete!")
    logger.info(f"  Annotated file: {output_file}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Run LLM annotation on same samples:")
    logger.info(f"     python scripts/04_llm_annotation.py {output_file}")
    logger.info(f"  2. Calculate Cohen's kappa agreement")


if __name__ == "__main__":
    main()
