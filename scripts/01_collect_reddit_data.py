#!/usr/bin/env python3
"""
Collect Reddit data from specified subreddits.
This script uses PRAW to fetch posts and comments, then verifies code-switching.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import praw
import yaml
from dotenv import load_dotenv
from langdetect import detect_langs
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RedditDataCollector:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize Reddit data collector with configuration."""
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.reddit = self._init_reddit()
        self.raw_data_dir = Path(self.config["paths"]["raw_data"])
        self.raw_data_dir.mkdir(exist_ok=True)

        self.collected_data = []

    def _init_reddit(self) -> praw.Reddit:
        """Initialize PRAW Reddit client."""
        try:
            reddit = praw.Reddit(
                client_id="YOUR_CLIENT_ID",
                client_secret="YOUR_CLIENT_SECRET",
                user_agent="code-switched-nlp-collector/1.0",
            )
            logger.info("PRAW client initialized successfully")
            return reddit
        except Exception as e:
            logger.error(f"Failed to initialize PRAW: {e}")
            raise

    def _detect_languages(self, text: str) -> Dict[str, float]:
        """Detect languages in text using langdetect."""
        try:
            detections = detect_langs(text)
            return {str(d).split(":")[0]: float(str(d).split(":")[1]) for d in detections}
        except Exception:
            return {}

    def _is_codeswitched(self, text: str, min_languages: int = 2) -> bool:
        """
        Check if text contains code-switching.
        Must have at least min_languages detected languages with reasonable confidence.
        """
        langs = self._detect_languages(text)

        # Check minimum language requirement
        if len(langs) < min_languages:
            return False

        # Check for required language pairs
        required_langs = set(self.config["data_collection"]["language_detection"]["required_languages"])
        detected_langs = set(langs.keys())

        # Must have English + (Hindi or Spanish)
        has_english = "en" in detected_langs
        has_target = ("hi" in detected_langs) or ("es" in detected_langs)

        return has_english and has_target

    def _is_valid_text(self, text: str) -> bool:
        """Validate text based on length and content requirements."""
        config = self.config["data_collection"]["reddit"]

        if not text or len(text) < config["min_chars"] or len(text) > config["max_chars"]:
            return False

        # Skip deleted/removed content
        if text in ["[deleted]", "[removed]"]:
            return False

        return True

    def collect_from_subreddit(self, subreddit_name: str, limit: int = 1000) -> List[Dict]:
        """Collect posts and comments from a single subreddit."""
        logger.info(f"Collecting from {subreddit_name} (limit: {limit})")

        collected = []
        subreddit = self.reddit.subreddit(subreddit_name.replace("r/", ""))
        config = self.config["data_collection"]["reddit"]

        # Collect posts
        if config.get("collect_posts", True):
            logger.info(f"  Collecting posts...")
            for post in tqdm(
                subreddit.new(limit=limit // 2),
                total=limit // 2,
                desc=f"Posts from {subreddit_name}",
            ):
                if self._is_valid_text(post.selftext) and self._is_codeswitched(post.selftext):
                    collected.append({
                        "id": post.id,
                        "type": "post",
                        "text": post.selftext,
                        "title": post.title,
                        "subreddit": subreddit_name,
                        "timestamp": post.created_utc,
                        "url": post.url,
                        "score": post.score,
                        "num_comments": post.num_comments,
                    })

        # Collect comments
        if config.get("collect_comments", True):
            logger.info(f"  Collecting comments...")
            for post in tqdm(
                subreddit.new(limit=limit // 4),
                total=limit // 4,
                desc=f"Posts (for comments) from {subreddit_name}",
            ):
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list()[:10]:  # Limit comments per post
                        if self._is_valid_text(comment.body) and self._is_codeswitched(comment.body):
                            collected.append({
                                "id": comment.id,
                                "type": "comment",
                                "text": comment.body,
                                "parent_id": comment.parent_id,
                                "subreddit": subreddit_name,
                                "timestamp": comment.created_utc,
                                "score": comment.score,
                            })
                except Exception as e:
                    logger.warning(f"Error collecting comments: {e}")
                    continue

        logger.info(f"  ✓ Collected {len(collected)} items from {subreddit_name}")
        return collected

    def collect_all(self) -> None:
        """Collect data from all configured subreddits."""
        config = self.config["data_collection"]
        subreddits = config["reddit"]["subreddits"]
        limit_per_subreddit = config["reddit"]["limit_per_subreddit"]

        logger.info(f"Starting collection from {len(subreddits)} subreddits")

        for subreddit in subreddits:
            try:
                data = self.collect_from_subreddit(subreddit, limit=limit_per_subreddit)
                self.collected_data.extend(data)
            except Exception as e:
                logger.error(f"Error collecting from {subreddit}: {e}")
                continue

        logger.info(f"Total collected: {len(self.collected_data)} items")

    def save_to_file(self, filename: Optional[str] = None) -> str:
        """Save collected data to JSONL file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_codeswitched_{timestamp}.jsonl"

        filepath = self.raw_data_dir / filename

        with open(filepath, "w") as f:
            for item in self.collected_data:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Data saved to {filepath}")
        return str(filepath)

    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        if not self.collected_data:
            return {}

        subreddit_counts = {}
        for item in self.collected_data:
            sub = item["subreddit"]
            subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1

        return {
            "total_items": len(self.collected_data),
            "by_subreddit": subreddit_counts,
            "post_count": sum(1 for item in self.collected_data if item["type"] == "post"),
            "comment_count": sum(1 for item in self.collected_data if item["type"] == "comment"),
        }


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Reddit Code-Switched Data Collector")
    logger.info("="*60)

    collector = RedditDataCollector()

    # Collect data
    collector.collect_all()

    # Save to file
    output_file = collector.save_to_file()

    # Print statistics
    stats = collector.get_statistics()
    logger.info("Collection Statistics:")
    logger.info(f"  Total items: {stats.get('total_items', 0)}")
    logger.info(f"  Posts: {stats.get('post_count', 0)}")
    logger.info(f"  Comments: {stats.get('comment_count', 0)}")
    logger.info(f"  By subreddit: {stats.get('by_subreddit', {})}")

    logger.info(f"\nData saved to: {output_file}")


if __name__ == "__main__":
    main()
