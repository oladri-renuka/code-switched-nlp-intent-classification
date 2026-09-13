#!/usr/bin/env python3
"""
Quick batch annotation - annotate multiple samples efficiently.
Input: intent sentiment (e.g., "1 n" for complaint + negative)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

INTENTS = ["question", "complaint", "recommendation", "humor", "emotional", "informational"]
SENTIMENTS = {"p": "positive", "n": "negative", "u": "neutral"}


class QuickAnnotator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.annotated_data_dir = Path(self.config["paths"]["annotated_data"])
        self.annotated_data_dir.mkdir(exist_ok=True)

    def annotate_batch(self, input_file: str, output_file: str, sample_count: int = 100):
        """Quick batch annotation with compact display."""
        items = []
        with open(input_file) as f:
            for idx, line in enumerate(f):
                if idx >= sample_count:
                    break
                try:
                    items.append(json.loads(line))
                except:
                    continue

        print("\n" + "="*80)
        print("QUICK ANNOTATION MODE")
        print("="*80)
        print("Enter: INTENT (0-5) SENTIMENT (p/n/u)")
        print("Example: '1 n' = complaint + negative")
        print("Press Ctrl+C to stop, Enter to skip\n")

        annotated = []
        for idx, item in enumerate(items, 1):
            text = item.get("text", "")[:100]
            print(f"[{idx}/{len(items)}] {text}...")
            print(f"    ID: {item.get('id')}")

            try:
                response = input("    > ").strip()

                if not response:
                    continue

                parts = response.split()
                if len(parts) < 2:
                    print("    ✗ Invalid (need: intent sentiment)")
                    continue

                intent_id = int(parts[0])
                sentiment_id = parts[1].lower()

                if intent_id not in range(6) or sentiment_id not in SENTIMENTS:
                    print(f"    ✗ Invalid (intent 0-5, sentiment p/n/u)")
                    continue

                item["annotation"] = {
                    "intent": INTENTS[intent_id],
                    "sentiment": SENTIMENTS[sentiment_id],
                }
                annotated.append(item)
                print(f"    ✓ {INTENTS[intent_id]} + {SENTIMENTS[sentiment_id]}")

            except KeyboardInterrupt:
                print("\n\n✓ Annotation stopped by user")
                break
            except ValueError:
                print("    ✗ Invalid input")
                continue

        # Save
        output_path = self.annotated_data_dir / output_file
        with open(output_path, "w") as f:
            for item in annotated:
                f.write(json.dumps(item) + "\n")

        print(f"\n✓ Saved {len(annotated)} annotations to: {output_path}")
        return str(output_path)


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/annotated/manual_annotation_items.jsonl"
    sample_count = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    annotator = QuickAnnotator()
    annotator.annotate_batch(input_file, "manual_annotations.jsonl", sample_count)
