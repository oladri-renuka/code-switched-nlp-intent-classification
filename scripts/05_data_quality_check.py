#!/usr/bin/env python3
"""Quick data quality check before training."""

import json
from pathlib import Path
from collections import Counter

def check_data_quality(input_file: str):
    """Check data quality metrics."""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)

    items = []
    duplicates = set()
    seen_texts = set()

    with open(input_file) as f:
        for line in f:
            try:
                item = json.loads(line)
                items.append(item)

                text = item.get("text", "").strip()
                if text in seen_texts:
                    duplicates.add(text)
                seen_texts.add(text)
            except json.JSONDecodeError:
                continue

    print(f"\n✓ Total samples: {len(items)}")
    print(f"✓ Unique samples: {len(seen_texts)}")
    print(f"✗ Duplicates: {len(duplicates)}")

    # Text length distribution
    lengths = [len(item.get("text", "").split()) for item in items]
    print(f"\nText length (words):")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Avg: {sum(lengths)/len(lengths):.1f}")

    # Check for missing values
    missing_text = sum(1 for item in items if not item.get("text"))
    missing_intent = sum(1 for item in items if not item.get("intent"))
    print(f"\nMissing values:")
    print(f"  text: {missing_text}")
    print(f"  intent: {missing_intent}")

    # Intent distribution
    intents = Counter(item.get("intent") for item in items if item.get("intent"))
    print(f"\nIntent distribution:")
    for intent, count in intents.most_common():
        pct = (count / len(items)) * 100
        print(f"  {intent}: {count} ({pct:.1f}%)")

    # Check for very short texts (< 3 words)
    short_texts = sum(1 for l in lengths if l < 3)
    print(f"\n⚠ Very short texts (<3 words): {short_texts}")

    print("\n" + "="*60)
    print("QUALITY CHECK COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    check_data_quality("data/processed/verified_codeswitched.jsonl")
