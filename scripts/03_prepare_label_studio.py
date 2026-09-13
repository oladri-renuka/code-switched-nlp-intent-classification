#!/usr/bin/env python3
"""
Prepare data for Label Studio annotation.
Converts verified code-switched data to Label Studio import format.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LabelStudioPreparator:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Label Studio data preparer."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.processed_data_dir = Path(self.config["paths"]["processed_data"])
        self.annotated_data_dir = Path(self.config["paths"]["annotated_data"])
        self.annotated_data_dir.mkdir(exist_ok=True)

    def load_verified_data(self, filepath: str) -> List[Dict]:
        """Load verified code-switched data."""
        logger.info(f"Loading verified data from {filepath}")

        items = []
        with open(filepath) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(items)} items")
        return items

    def create_label_studio_json(self, items: List[Dict]) -> List[Dict]:
        """
        Convert items to Label Studio JSON format.
        Label Studio expects: [{"id": ..., "data": {"text": ...}}, ...]
        """
        label_studio_data = []

        for idx, item in enumerate(items, 1):
            label_studio_item = {
                "id": idx,
                "data": {
                    "text": item["text"],
                    "reddit_id": item["id"],
                    "subreddit": item.get("subreddit", ""),
                    "language_pair": item.get("language_pair", ""),
                    "source_languages": str(item.get("languages", {})),
                },
            }
            label_studio_data.append(label_studio_item)

        return label_studio_data

    def create_label_config_xml(self) -> str:
        """Create Label Studio label configuration XML."""
        config_labels = self.config["annotation"]["labels"]

        xml = """<View>
  <Text name="text" value="$text"/>

  <Choices name="intent" toName="text" required="true">
    <Choice value="question" />
    <Choice value="complaint" />
    <Choice value="recommendation" />
    <Choice value="humor" />
    <Choice value="emotional" />
    <Choice value="informational" />
  </Choices>

  <Choices name="sentiment" toName="text" required="true">
    <Choice value="positive" />
    <Choice value="negative" />
    <Choice value="neutral" />
  </Choices>

  <Choices name="language_pair" toName="text" required="false">
    <Choice value="en-hi" />
    <Choice value="en-es" />
  </Choices>

  <TextArea name="notes" toName="text" placeholder="Additional notes..." />
</View>
"""
        return xml

    def split_data(
        self,
        items: List[Dict],
        manual_annotation_count: int = 500,
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Split data into manual annotation and LLM-assisted annotation sets.
        """
        random.shuffle(items)

        manual_set = items[:manual_annotation_count]
        llm_set = items[manual_annotation_count:]

        logger.info(f"Manual annotation set: {len(manual_set)}")
        logger.info(f"LLM-assisted annotation set: {len(llm_set)}")

        return manual_set, llm_set

    def prepare_for_annotation(self, filepath: str) -> Dict[str, str]:
        """Prepare all files for annotation."""
        logger.info("="*60)
        logger.info("Preparing Data for Label Studio")
        logger.info("="*60)

        # Load data
        items = self.load_verified_data(filepath)

        if len(items) < 3000:
            logger.warning(f"Warning: Only {len(items)} items. Target is 3000+")

        # Split data
        manual_set, llm_set = self.split_data(
            items,
            manual_annotation_count=self.config["annotation"]["manual_annotation"]["target_count"],
        )

        # Convert to Label Studio format
        manual_ls = self.create_label_studio_json(manual_set)
        llm_ls = self.create_label_studio_json(llm_set)

        # Save files
        manual_file = self.annotated_data_dir / "manual_annotation_import.json"
        llm_file = self.annotated_data_dir / "llm_annotation_import.json"

        with open(manual_file, "w") as f:
            json.dump(manual_ls, f, indent=2)

        with open(llm_file, "w") as f:
            json.dump(llm_ls, f, indent=2)

        logger.info(f"✓ Manual annotation file: {manual_file}")
        logger.info(f"✓ LLM annotation file: {llm_file}")

        # Save label config
        label_config = self.create_label_config_xml()
        config_file = self.annotated_data_dir / "label_config.xml"
        with open(config_file, "w") as f:
            f.write(label_config)

        logger.info(f"✓ Label config: {config_file}")

        # Save original splits for reference
        manual_original = self.annotated_data_dir / "manual_annotation_items.jsonl"
        llm_original = self.annotated_data_dir / "llm_annotation_items.jsonl"

        with open(manual_original, "w") as f:
            for item in manual_set:
                f.write(json.dumps(item) + "\n")

        with open(llm_original, "w") as f:
            for item in llm_set:
                f.write(json.dumps(item) + "\n")

        logger.info(f"✓ Manual original items: {manual_original}")
        logger.info(f"✓ LLM original items: {llm_original}")

        return {
            "manual_annotation": str(manual_file),
            "llm_annotation": str(llm_file),
            "label_config": str(config_file),
            "manual_items": str(manual_original),
            "llm_items": str(llm_original),
        }

    def print_instructions(self) -> None:
        """Print Label Studio setup instructions."""
        logger.info("\n" + "="*60)
        logger.info("Label Studio Setup Instructions")
        logger.info("="*60)
        logger.info("""
1. Start Label Studio:
   docker run -it -p 8080:8080 heartexlabs/label-studio

2. Open: http://localhost:8080

3. Create a new project:
   - Project name: "code-switched-intent"
   - Click "Create"

4. Import label configuration:
   - Go to Settings > Labeling Interface
   - Paste the XML from label_config.xml
   - Click "Save"

5. Import data:
   - Go to Data Import
   - Import manual_annotation_import.json (for 500 manual samples)
   - Import llm_annotation_import.json (for 2500 LLM-assisted samples)

6. Start annotating!
   - Assign manual_annotation_import.json to yourself
   - Create tasks for team members or AI annotation

7. Export annotations:
   - Format: JSON
   - Save to data/annotated/
""")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python 03_prepare_label_studio.py <verified_codeswitched.jsonl>")
        sys.exit(1)

    input_file = sys.argv[1]

    preparator = LabelStudioPreparator()
    output_files = preparator.prepare_for_annotation(input_file)

    logger.info(f"\n✓ Preparation complete!")
    for key, path in output_files.items():
        logger.info(f"  {key}: {path}")

    preparator.print_instructions()


if __name__ == "__main__":
    main()
