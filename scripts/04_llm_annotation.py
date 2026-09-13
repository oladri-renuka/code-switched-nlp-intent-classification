#!/usr/bin/env python3
"""
LLM-assisted annotation using Claude Sonnet.
Annotates code-switched text with intent, sentiment, and language pair.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import requests
import yaml
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LLMAnnotator:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize LLM annotator with Llama via HuggingFace Inference API (free)."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # HuggingFace Inference API setup (free, no token needed)
        self.api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        self.model = "meta-llama/Llama-2-7b-chat-hf"
        self.headers = {"Content-Type": "application/json"}

        self.annotated_data_dir = Path(self.config["paths"]["annotated_data"])
        self.annotated_data_dir.mkdir(exist_ok=True)

        self.few_shot_examples = self._create_few_shot_examples()

    def _create_few_shot_examples(self) -> str:
        """Create few-shot examples for LLM annotation."""
        examples = """
Example 1:
Text: "Bhai, tumhe pata hai ki naya iPhone release ho gaya? Mujhe lena hai but itna mehnga hai!"
(Brother, do you know the new iPhone was released? I want to get it but it's so expensive!)
Intent: question
Sentiment: neutral
Language Pair: en-hi

Example 2:
Text: "Isse ziada disappointing update kabhi nahi dekha. Devs ne kya kiya? Complete disaster!"
(Never seen such a disappointing update. What did the devs do? Complete disaster!)
Intent: complaint
Sentiment: negative
Language Pair: en-hi

Example 3:
Text: "Hey, you should totally try this new chai recipe I found. Es muy delicioso and so easy to make!"
(Hey, you should totally try this new chai recipe I found. It's very delicious and so easy to make!)
Intent: recommendation
Sentiment: positive
Language Pair: en-es

Example 4:
Text: "Mein sach kahun toh yeh sab bakwas hai. 😂 Indians in USA dealing with this everyday!"
(Honestly this is all nonsense. 😂 Indians in USA dealing with this everyday!)
Intent: humor
Sentiment: neutral
Language Pair: en-hi

Example 5:
Text: "I miss home so much, yaar. El clima here no se compara with back home. My heart aches."
(I miss home so much, buddy. The weather here doesn't compare with back home. My heart aches.)
Intent: emotional
Sentiment: negative
Language Pair: en-hi/en-es (mixed)
"""
        return examples

    def _create_annotation_prompt(self, text: str) -> str:
        """Create prompt for LLM annotation."""
        prompt = f"""You are an expert in code-switched text analysis. Analyze the following code-switched text (mixing English with Hindi/Spanish) and provide annotations.

{self.few_shot_examples}

Now annotate this text:

Text: "{text}"

Respond with JSON in this exact format (no markdown, just JSON):
{{
    "intent": "<one of: question, complaint, recommendation, humor, emotional, informational>",
    "sentiment": "<one of: positive, negative, neutral>",
    "language_pair": "<'en-hi' or 'en-es' or 'en-hi/en-es'>",
    "confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}
"""
        return prompt

    def annotate_text(self, text: str) -> Optional[Dict]:
        """Annotate a single text using Llama via HuggingFace Inference API (free)."""
        try:
            prompt = self._create_annotation_prompt(text)

            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": self.config["annotation"]["llm_annotation"]["temperature"],
                }
            }

            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()

            response_json = response.json()

            # HuggingFace returns list of dicts with 'generated_text' key
            if isinstance(response_json, list) and len(response_json) > 0:
                response_text = response_json[0].get("generated_text", "")
                # Extract JSON from response (Llama may add extra text)
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    response_text = response_text[json_start:json_end]
            else:
                logger.warning(f"Unexpected response format: {response_json}")
                return None

            # Parse JSON response
            annotation = json.loads(response_text)
            return annotation

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"HuggingFace API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during annotation: {e}")
            return None

    def annotate_batch(
        self,
        items: list[Dict],
        batch_size: int = 50,
        output_file: str = "llm_annotations.jsonl",
    ) -> str:
        """Annotate a batch of items."""
        logger.info(f"Annotating {len(items)} items (batch size: {batch_size})")

        output_path = self.annotated_data_dir / output_file
        annotated_count = 0
        failed_count = 0

        with open(output_path, "w") as f:
            for item in tqdm(items, desc="Annotating with LLM"):
                text = item.get("text", "")

                if not text:
                    failed_count += 1
                    continue

                annotation = self.annotate_text(text)

                if annotation:
                    item["llm_annotation"] = annotation
                    f.write(json.dumps(item) + "\n")
                    annotated_count += 1
                else:
                    failed_count += 1

        logger.info(f"✓ Annotated: {annotated_count}")
        logger.info(f"✗ Failed: {failed_count}")
        logger.info(f"Output: {output_path}")

        return str(output_path)

    def compare_with_manual(self, manual_file: str, llm_file: str) -> Dict:
        """Compare LLM annotations with manual annotations."""
        from sklearn.metrics import cohen_kappa_score

        logger.info("Calculating agreement between manual and LLM annotations...")

        manual_annotations = []
        llm_annotations = []

        # Load manual annotations (assuming Label Studio export format)
        with open(manual_file) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "annotation" in item:
                        manual_annotations.append(item["annotation"])
                except json.JSONDecodeError:
                    continue

        # Load LLM annotations
        with open(llm_file) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "llm_annotation" in item:
                        llm_annotations.append(item["llm_annotation"])
                except json.JSONDecodeError:
                    continue

        if not manual_annotations or not llm_annotations:
            logger.warning("Not enough annotations for comparison")
            return {}

        # Calculate Cohen's kappa for intent classification
        sample_size = min(len(manual_annotations), len(llm_annotations))

        intent_labels_manual = [ann.get("intent", "") for ann in manual_annotations[:sample_size]]
        intent_labels_llm = [ann.get("intent", "") for ann in llm_annotations[:sample_size]]

        # Map to numeric labels for kappa calculation
        all_intents = set(intent_labels_manual + intent_labels_llm)
        intent_to_id = {intent: idx for idx, intent in enumerate(all_intents)}

        manual_ids = [intent_to_id[l] for l in intent_labels_manual]
        llm_ids = [intent_to_id[l] for l in intent_labels_llm]

        kappa = cohen_kappa_score(manual_ids, llm_ids)

        return {
            "sample_size": sample_size,
            "cohens_kappa": kappa,
            "agreement_percentage": sum(1 for m, l in zip(intent_labels_manual, intent_labels_llm) if m == l)
            / sample_size
            * 100,
        }


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        logger.error("Usage: python 04_llm_annotation.py <llm_annotation_items.jsonl>")
        sys.exit(1)

    input_file = sys.argv[1]

    logger.info("="*60)
    logger.info("LLM-Assisted Annotation with Claude")
    logger.info("="*60)

    annotator = LLMAnnotator()

    # Load items
    items = []
    with open(input_file) as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(items)} items for annotation")

    # Annotate batch
    output_file = annotator.annotate_batch(
        items,
        batch_size=int(annotator.config["annotation"]["llm_annotation"]["batch_size"]),
    )

    logger.info(f"\n✓ LLM annotation complete!")
    logger.info(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
