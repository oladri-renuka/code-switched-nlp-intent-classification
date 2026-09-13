#!/usr/bin/env python3
"""
Directly create manual_annotations.jsonl with all annotations.
No interactive prompts - just load samples and add annotations.
"""

import json
from pathlib import Path
from typing import Dict, List

import yaml

INTENTS = ["question", "complaint", "recommendation", "humor", "emotional", "informational"]
SENTIMENTS = {"p": "positive", "n": "negative", "u": "neutral"}


def create_annotations(input_file: str, annotations_list: List[tuple], output_file: str):
    """
    Create JSONL file with annotations directly.

    Args:
        input_file: JSONL file with samples
        annotations_list: List of (intent_id, sentiment_id) tuples
        output_file: Output JSONL file
    """
    with open(input_file) as f:
        config = yaml.safe_load(open("config.yaml"))
        annotated_dir = Path(config["paths"]["annotated_data"])
        annotated_dir.mkdir(exist_ok=True)

    items = []
    with open(input_file) as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except:
                continue

    # Annotate samples
    annotated = []
    for idx, (intent_id, sentiment_id) in enumerate(annotations_list):
        if idx >= len(items):
            break

        item = items[idx]
        item["annotation"] = {
            "intent": INTENTS[intent_id],
            "sentiment": SENTIMENTS[sentiment_id],
        }
        annotated.append(item)

    # Save
    output_path = annotated_dir / output_file
    with open(output_path, "w") as f:
        for item in annotated:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Saved {len(annotated)} annotations to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Batch 1-10: All 200 annotations
    batch1 = [(4, "p"), (1, "n"), (3, "n"), (4, "p"), (4, "p"), (1, "n"), (1, "n"), (5, "u"), (1, "n"), (1, "n"), (5, "u"), (3, "u"), (1, "n"), (1, "n"), (5, "u"), (3, "u"), (4, "p"), (1, "n"), (5, "u"), (2, "p")]
    batch2 = [(2, "p"), (1, "n"), (5, "u"), (1, "n"), (2, "p"), (2, "p"), (1, "n"), (1, "n"), (2, "u"), (0, "u"), (5, "u"), (1, "n"), (1, "n"), (1, "n"), (1, "n"), (1, "n"), (4, "n"), (1, "n"), (4, "p"), (4, "p")]
    batch3 = [(3, "u"), (1, "n"), (4, "p"), (5, "u"), (2, "u"), (4, "p"), (1, "n"), (2, "p"), (0, "u"), (2, "p"), (1, "n"), (4, "p"), (3, "u"), (1, "n"), (1, "n"), (2, "u"), (3, "u"), (4, "n"), (3, "n"), (4, "p")]
    batch4 = [(4, "n"), (1, "n"), (1, "n"), (1, "n"), (2, "p"), (1, "n"), (1, "n"), (1, "n"), (4, "p"), (1, "n"), (1, "n"), (1, "n"), (1, "n"), (1, "n"), (4, "u"), (4, "p"), (2, "u"), (1, "n"), (2, "p"), (4, "p")]
    batch5 = [(2, "p"), (1, "n"), (5, "u"), (1, "n"), (3, "u"), (1, "n"), (4, "p"), (4, "p"), (4, "p"), (4, "p"), (5, "u"), (1, "n"), (5, "u"), (1, "n"), (4, "p"), (1, "n"), (4, "u"), (1, "n"), (1, "n"), (1, "n")]
    batch6 = [(1, "n"), (5, "u"), (5, "p"), (5, "u"), (1, "n"), (3, "u"), (1, "n"), (4, "p"), (1, "n"), (4, "p"), (5, "u"), (4, "p"), (3, "u"), (0, "u"), (1, "n"), (1, "n"), (1, "n"), (5, "u"), (1, "n"), (2, "p")]
    batch7 = [(5, "u"), (3, "u"), (1, "n"), (1, "n"), (4, "p"), (5, "n"), (1, "n"), (1, "n"), (5, "u"), (5, "u"), (1, "n"), (3, "n"), (1, "n"), (1, "n"), (1, "n"), (4, "n"), (3, "u"), (1, "n"), (5, "u"), (1, "n")]
    batch8 = [(1, "n"), (5, "u"), (1, "n"), (2, "p"), (5, "u"), (4, "p"), (5, "n"), (1, "n"), (1, "n"), (4, "n"), (1, "n"), (1, "n"), (4, "p"), (1, "n"), (1, "n"), (4, "u"), (1, "n"), (2, "p"), (2, "p"), (0, "u")]
    batch9 = [(2, "p"), (5, "u"), (3, "u"), (1, "n"), (4, "p"), (5, "u"), (4, "p"), (5, "u"), (5, "u"), (3, "u"), (4, "p"), (1, "n"), (1, "n"), (3, "u"), (4, "p"), (4, "p"), (4, "p"), (1, "n"), (1, "n"), (2, "p")]
    batch10 = [(4, "p"), (1, "n"), (3, "n"), (2, "p"), (5, "u"), (3, "n"), (2, "p"), (4, "n"), (4, "p"), (1, "n"), (1, "n"), (4, "p"), (1, "n"), (1, "n"), (1, "n"), (4, "p"), (5, "u"), (1, "n"), (4, "p"), (5, "u")]

    # Combine all 10 batches (200 total)
    all_annotations = batch1 + batch2 + batch3 + batch4 + batch5 + batch6 + batch7 + batch8 + batch9 + batch10

    # Create annotations file
    create_annotations(
        "data/annotated/manual_annotation_items.jsonl",
        all_annotations,
        "manual_annotations.jsonl"
    )

    print(f"\n✓ Total annotations: {len(all_annotations)}")
    print("Ready for LLM comparison and Cohen's kappa!")
