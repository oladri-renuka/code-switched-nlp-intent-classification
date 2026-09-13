#!/usr/bin/env python3
"""
Create sample code-switched data for demonstration.
Alternative to Reddit API when not available.

Options:
1. Generate realistic synthetic examples
2. Use pre-collected code-switched datasets
3. Load from CSV/JSON files
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generate realistic code-switched examples for testing."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize generator."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.raw_data_dir = Path(self.config["paths"]["raw_data"])
        self.raw_data_dir.mkdir(exist_ok=True)

    # Hindi-English examples (r/hinglish style)
    HINDI_ENGLISH_EXAMPLES = [
        "Bhai, naya iPhone 15 release ho gaya! Khareed lu kya? Bahut mehnga ho gaya na!",
        "Arre, tumhara project kaise chal raha hai? Mujhe bhi sikhna hai Python programming.",
        "Mein sach kahun toh yeh sab bakwas hai. 😂 Indians in USA dealing with this everyday!",
        "Dost, mere ghar mein new puppy aa gaya! So cute and fluffy! Dekh to beta.",
        "Isse ziada disappointing update kabhi nahi dekha. Devs ne kya kiya? Complete disaster!",
        "I miss home so much, yaar. Ghar ki khana, ghar ki yaadein... everything feels different here.",
        "Beta, tumhara exam kaisa tha? Preparation theek se ki thi na? Marks ache aa gaye?",
        "Guys, seriously this new restaurant in town has amazing biryani! Must try!",
        "Mera laptop crash ho gaya at the worst time! Ab kya karu? Koi solution hai?",
        "You won't believe what happened today! Mere boss ne kaha ki project ko kal hi submit karna hai!",
        "Arrey yaar, cricket match dekha? Virat ne century lag dii! Kya shot tha!",
        "I'm so tired of working from home. Office jaana padega ab par travel itna boring hai.",
        "Mummy ne mujhe pure din gaaliyan suni kyuki maine ghar ke kaam nahi kiye! 😅",
        "This new web framework is absolutely amazing! Documentation bhi bahut clear hai.",
        "I can't believe we're already in September! Saal kitna fast nikal gaya!",
    ]

    # Spanish-English examples (r/latinos style)
    SPANISH_ENGLISH_EXAMPLES = [
        "Ay dios mío, the weather today is absolutely gorgeous! No puedo esperar hasta el fin de semana.",
        "Hermano, have you tried the new taco place? Es muy delicioso and so authentic!",
        "I'm so frustrated with work today. Mi boss no entiende nada! Es imposible trabajar aquí.",
        "Hey, you should totally try this chai recipe I found. Es muy sabroso and so easy to make!",
        "No me digas! Your sister got married? Felicidades! Qué emoción for the whole familia!",
        "I miss home so much, hermano. El clima here no se compara with back home. Ay que triste.",
        "This assignment is so hard! Tengo que estudiar mucho más pero I'm already exhausted.",
        "Mira, your new car looks increíble! How much did you pay por eso? Must have been expensive!",
        "I can't believe they cancelled the festival! Todos están so disappointed right now.",
        "Ey, did you see the game last night? Nuestro team won! Qué alegría for everyone!",
        "This coffee tastes like water, no me gusta nada! Where's the good cafecito?",
        "I'm thinking about going back to school pero tengo mucho miedo. No sé if I can handle it.",
        "Tu hermano es muy guapo! But honestly, I think you're prettier. No seas modesta!",
        "This song is stuck in my head all day! Es tan pegadizo and I can't stop singing it.",
        "Seriously, the traffic en esta ciudad is absolutely ridiculous! Cuánto tiempo I wasted today!",
    ]

    def generate_synthetic_data(self, num_examples: int = 1500) -> List[Dict]:
        """Generate synthetic code-switched examples."""
        logger.info(f"Generating {num_examples} synthetic code-switched examples")

        examples = []
        subreddits = ["r/hinglish", "r/latinos", "r/IndiansInUSA"]

        # Alternate between Hindi-English and Spanish-English
        for i in range(num_examples):
            timestamp = int(datetime.now().timestamp()) - (i * 3600)  # Spread over time

            if i % 2 == 0:
                # Hindi-English
                text = self.HINDI_ENGLISH_EXAMPLES[i % len(self.HINDI_ENGLISH_EXAMPLES)]
                subreddit = subreddits[i % 2]  # r/hinglish or r/IndiansInUSA
            else:
                # Spanish-English
                text = self.SPANISH_ENGLISH_EXAMPLES[i % len(self.SPANISH_ENGLISH_EXAMPLES)]
                subreddit = subreddits[1]  # r/latinos

            examples.append({
                "id": f"synthetic_{i:05d}",
                "type": "post" if i % 3 == 0 else "comment",
                "text": text,
                "subreddit": subreddit,
                "timestamp": timestamp,
                "score": 10 + (i % 100),
                "num_comments": 5 + (i % 20),
                "url": f"https://reddit.com/r/{subreddit.replace('r/', '')}/synthetic_{i}",
            })

        return examples

    def load_from_csv(self, csv_path: str) -> List[Dict]:
        """
        Load code-switched data from CSV file.

        Expected columns: text, subreddit, type (optional)
        """
        import csv

        logger.info(f"Loading data from {csv_path}")

        examples = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                examples.append({
                    "id": f"csv_{i:05d}",
                    "type": row.get("type", "post"),
                    "text": row["text"],
                    "subreddit": row.get("subreddit", "r/codeswitched"),
                    "timestamp": int(datetime.now().timestamp()),
                    "score": 0,
                    "url": "",
                })

        logger.info(f"Loaded {len(examples)} examples from CSV")
        return examples

    def load_from_json(self, json_path: str) -> List[Dict]:
        """
        Load code-switched data from JSON/JSONL file.

        Expected format: List of dicts or JSONL (one dict per line)
        """
        logger.info(f"Loading data from {json_path}")

        examples = []

        try:
            # Try JSONL format first
            with open(json_path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            if "text" not in item:
                                logger.warning(f"Skipping line {i}: missing 'text' field")
                                continue
                            examples.append(item)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            # Try JSON format
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        examples = data
                    else:
                        logger.error("JSON file must be a list of objects")
            except Exception as e:
                logger.error(f"Failed to load JSON: {e}")

        logger.info(f"Loaded {len(examples)} examples from JSON")
        return examples

    def save_data(self, examples: List[Dict], filename: str = None) -> str:
        """Save examples to JSONL file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"codeswitched_sample_{timestamp}.jsonl"

        filepath = self.raw_data_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(examples)} examples to {filepath}")
        return str(filepath)

    def create_demo_dataset(self) -> str:
        """Create a small demo dataset for testing."""
        logger.info("Creating demo dataset...")

        # Generate synthetic data
        examples = self.generate_synthetic_data(num_examples=100)

        # Save to file
        output_file = self.save_data(examples, "demo_codeswitched_data.jsonl")

        return output_file


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Code-Switched Sample Data Generator")
    logger.info("="*60)

    generator = SampleDataGenerator()

    # Option 1: Create synthetic demo data (quick testing)
    logger.info("\nOption 1: Creating synthetic demo data...")
    demo_file = generator.create_demo_dataset()
    logger.info(f"✓ Demo data ready: {demo_file}")

    # Option 2: Generate larger synthetic dataset
    logger.info("\nOption 2: Generating full synthetic dataset (1500 examples)...")
    examples = generator.generate_synthetic_data(num_examples=1500)
    full_file = generator.save_data(examples, "codeswitched_synthetic_full.jsonl")
    logger.info(f"✓ Full dataset ready: {full_file}")

    logger.info("\n" + "="*60)
    logger.info("Data Generation Complete!")
    logger.info("="*60)
    logger.info(f"\nGenerated files in: {generator.raw_data_dir}")
    logger.info(f"  1. {demo_file} (100 examples - quick test)")
    logger.info(f"  2. {full_file} (1500 examples - full training)")

    logger.info("\nNext steps:")
    logger.info("  1. Verify code-switching: python scripts/02_verify_codeswitching.py <file>")
    logger.info("  2. Prepare for annotation: python scripts/03_prepare_label_studio.py <file>")

    logger.info("\nTo load your own data:")
    logger.info("  - CSV: columns must include 'text' and optionally 'subreddit', 'type'")
    logger.info("  - JSON/JSONL: list of dicts with 'text' field required")


if __name__ == "__main__":
    main()
