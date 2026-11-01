from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.data.dataset import NewsDataset


class NewsDatasetTests(unittest.TestCase):
    def test_latest_only_loads_most_recently_modified_raw_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            older_file = raw_dir / "2026-03-16_articles.json"
            newer_file = raw_dir / "2026-10-31_articles.json"
            older_file.write_text(
                json.dumps([{"title": "Old story", "body": "Old article text.", "url": "https://example.com/old"}]),
                encoding="utf-8",
            )
            newer_file.write_text(
                json.dumps([{"title": "Current story", "body": "Current article text.", "url": "https://example.com/new"}]),
                encoding="utf-8",
            )
            os.utime(older_file, (1_700_000_000, 1_700_000_000))
            os.utime(newer_file, (1_800_000_000, 1_800_000_000))

            dataset = NewsDataset.from_json_files(raw_dir, latest_only=True)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["title"], "Current story")


if __name__ == "__main__":
    unittest.main()
