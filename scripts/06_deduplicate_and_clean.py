#!/usr/bin/env python3
"""Remove duplicates and clean data before training."""

import json
from pathlib import Path
from collections import Counter

def clean_data(input_file: str, output_file: str, min_words: int = 3):
    """Remove duplicates and clean data."""
    print("\n" + "="*60)
    print("DATA CLEANING & DEDUPLICATION")
    print("="*60)

    items = []
    seen_texts = set()
    duplicates = 0
    short_texts = 0
    kept = 0

    with open(input_file) as f:
        for line in f:
            try:
                item = json.loads(line)
                text = item.get("text", "").strip()

                # Skip duplicates
                if text in seen_texts:
                    duplicates += 1
                    continue
                seen_texts.add(text)

                # Skip very short texts
                if len(text.split()) < min_words:
                    short_texts += 1
                    continue

                items.append(item)
                kept += 1

            except json.JSONDecodeError:
                continue

    # Save cleaned data
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"\n✓ Kept: {kept}")
    print(f"✗ Duplicates removed: {duplicates}")
    print(f"✗ Short texts removed: {short_texts}")
    print(f"\nIntent distribution:")

    intents = Counter(item.get("intent") for item in items)
    total = sum(intents.values())
    for intent, count in intents.most_common():
        pct = (count / total) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")

    print(f"\n✓ Saved to: {output_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    clean_data(
        "data/processed/verified_codeswitched.jsonl",
        "data/processed/cleaned_codeswitched.jsonl",
        min_words=3
    )
