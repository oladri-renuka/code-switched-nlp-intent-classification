#!/usr/bin/env python3
"""
Batch annotation - display 20-30 samples at once and annotate in bulk.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

INTENTS = ["question", "complaint", "recommendation", "humor", "emotional", "informational"]
SENTIMENTS = {"p": "positive", "n": "negative", "u": "neutral"}


class BatchAnnotator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.annotated_data_dir = Path(self.config["paths"]["annotated_data"])
        self.annotated_data_dir.mkdir(exist_ok=True)

    def display_batch(self, items: List[Dict], start_idx: int):
        """Display a batch of 20-30 samples."""
        print("\n" + "="*120)
        print(f"BATCH {start_idx//20 + 1} - Samples {start_idx+1} to {min(start_idx+20, start_idx+len(items))}")
        print("="*120)
        print(f"{'#':<3} {'Text (first 70 chars)':<75} {'ID':<20}")
        print("-"*120)

        for i, item in enumerate(items[:20]):
            text = item.get("text", "")[:70].replace("\n", " ")
            item_id = item.get("id", "")[:18]
            print(f"{i+1:<3} {text:<75} {item_id:<20}")

        print("="*120)

    def get_annotations_for_batch(self) -> List[tuple]:
        """Get annotations for the displayed batch."""
        annotations = []
        for i in range(20):
            while True:
                response = input(f"[{i+1}] Intent Sentiment (e.g., '1 n' or skip with Enter): ").strip()

                if not response:
                    break

                parts = response.split()
                if len(parts) < 2:
                    print("    ✗ Need both intent and sentiment. Try again.")
                    continue

                try:
                    intent_id = int(parts[0])
                    sentiment_id = parts[1].lower()

                    if intent_id not in range(6) or sentiment_id not in SENTIMENTS:
                        print(f"    ✗ Invalid. Intent must be 0-5, sentiment must be p/n/u")
                        continue

                    annotations.append((intent_id, sentiment_id))
                    print(f"    ✓ {INTENTS[intent_id]} + {SENTIMENTS[sentiment_id]}")
                    break

                except ValueError:
                    print("    ✗ Intent must be a number 0-5")
                    continue

        return annotations

    def annotate_batch(self, input_file: str, output_file: str, sample_count: int = 100):
        """Annotate samples in batches of 20-30."""
        items = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                if idx >= sample_count:
                    break
                try:
                    items.append(json.loads(line))
                except:
                    continue

        print("\n" + "="*120)
        print("BATCH ANNOTATION MODE (20-30 samples at once)")
        print("="*120)
        print("\nInstructions:")
        print("  1. Review the displayed samples")
        print("  2. For each sample, enter: INTENT (0-5) SENTIMENT (p/n/u)")
        print("  3. Press Enter to skip a sample")
        print("  4. Examples: '1 n' (complaint+negative), '4 p' (emotional+positive)")
        print("\nIntent codes:")
        for i, intent in enumerate(INTENTS):
            print(f"  {i} = {intent}")
        print("\nSentiment: p=positive, n=negative, u=neutral")
        print("="*120)

        input("\nPress Enter to start...")

        annotated = []
        batch_idx = 0

        try:
            while batch_idx < len(items):
                batch_end = min(batch_idx + 20, len(items))
                batch_items = items[batch_idx:batch_end]

                self.display_batch(batch_items, batch_idx)

                annotations = self.get_annotations_for_batch()

                for ann_idx, (intent_id, sentiment_id) in enumerate(annotations):
                    item = batch_items[ann_idx]
                    item["annotation"] = {
                        "intent": INTENTS[intent_id],
                        "sentiment": SENTIMENTS[sentiment_id],
                    }
                    annotated.append(item)

                batch_idx += 20

                if batch_idx < len(items):
                    cont = input(f"\nContinue to next batch? (y/n): ").strip().lower()
                    if cont != "y":
                        break

        except KeyboardInterrupt:
            print("\n\n✓ Annotation stopped by user")

        # Save
        output_path = self.annotated_data_dir / output_file
        with open(output_path, "w") as f:
            for item in annotated:
                f.write(json.dumps(item) + "\n")

        print(f"\n" + "="*120)
        print(f"✓ Saved {len(annotated)} annotations to: {output_path}")
        print("="*120)
        return str(output_path)


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/annotated/manual_annotation_items.jsonl"
    sample_count = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    annotator = BatchAnnotator()
    annotator.annotate_batch(input_file, "manual_annotations.jsonl", sample_count)
