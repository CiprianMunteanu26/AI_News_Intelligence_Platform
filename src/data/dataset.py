"""
News dataset — handles loading, processing, and management of news data.

Uses Hugging Face `datasets` for efficient storage and access.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk

from src.data.preprocess import NewsPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NewsDataset:
    """
    Wrapper around Hugging Face Dataset for news articles.
    """

    def __init__(self, dataset: Dataset | DatasetDict) -> None:
        self.dataset = dataset

    @classmethod
    def from_json_files(cls, input_dir: Path, preprocessor: NewsPreprocessor | None = None) -> NewsDataset:
        """
        Create a dataset from raw JSON files in a directory.

        Parameters
        ----------
        input_dir : Path
            Directory containing daily JSON files.
        preprocessor : NewsPreprocessor, optional
            Preprocessor to apply to each article.

        Returns
        -------
        NewsDataset
            Initialized dataset object.
        """
        all_articles = []
        json_files = list(input_dir.glob("*.json"))

        if not json_files:
            logger.warning("No JSON files found in %s", input_dir)
            return cls(Dataset.from_dict({"title": [], "body": [], "url": [], "source": [], "published_at": []}))

        for filepath in json_files:
            try:
                content = json.loads(filepath.read_text(encoding="utf-8"))
                if isinstance(content, list):
                    all_articles.extend(content)
                else:
                    logger.warning("File %s does not contain a list of articles", filepath)
            except Exception as e:
                logger.error("Failed to load %s: %s", filepath, e)

        if preprocessor:
            logger.info("Applying preprocessing to %d articles", len(all_articles))
            all_articles = [preprocessor.preprocess_article(a) for a in all_articles]

        # Convert list of dicts to dict of lists for HF Dataset
        if not all_articles:
            return cls(Dataset.from_dict({"title": [], "body": [], "url": [], "source": [], "published_at": []}))

        data_dict = {key: [a.get(key, "") for a in all_articles] for key in all_articles[0].keys()}

        return cls(Dataset.from_dict(data_dict))

    def save(self, output_path: Path) -> None:
        """Save dataset to disk."""
        self.dataset.save_to_disk(str(output_path))
        logger.info("Dataset saved to %s", output_path)

    @classmethod
    def load(cls, path: Path) -> NewsDataset:
        """Load dataset from disk."""
        return cls(load_from_disk(str(path)))

    def split(self, test_size: float = 0.1, seed: int = 42) -> NewsDataset:
        """Split dataset into train and test sets."""
        if isinstance(self.dataset, Dataset):
            split_dataset = self.dataset.train_test_split(test_size=test_size, seed=seed)
            return NewsDataset(split_dataset)
        return self

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx]
