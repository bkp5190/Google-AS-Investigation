from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import json
import logging
from typing import List

import requests

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DOWNLOAD_LINK = (
    "https://huggingface.co/datasets/"
    "sentence-transformers/embedding-training-data/"
    "resolve/main/gooaq_pairs.jsonl.gz"
)


@dataclass(frozen=True)
class SearchPair:
    source: str
    target: str


def _ensure_data_exists(data_path: Path) -> None:
    """
    Ensure the dataset exists locally. If not, download it.
    """
    if data_path.exists():
        logger.debug("Dataset already exists at %s", data_path)
        return

    logger.info("Dataset not found. Downloading %s", data_path.name)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(DOWNLOAD_LINK, stream=True, timeout=30)
    response.raise_for_status()

    with open(data_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    logger.info("Download complete: %s", data_path)


def load_search_pairs(
    data_dir: str | Path = "data",
    filename: str = "gooaq_pairs.jsonl.gz",
) -> List[SearchPair]:
    """
    Load GooAQ search pairs, downloading the dataset if necessary.
    """
    data_path = Path(data_dir) / filename

    _ensure_data_exists(data_path)

    search_pairs: list[SearchPair] = []

    with gzip.open(data_path, "rb") as file:
        for line in file:
            if not line:
                continue

            obj = json.loads(line)
            search_pairs.append(
                SearchPair(
                    source=obj[0],
                    target=obj[1],
                )
            )

    logger.info("Loaded %d search pairs", len(search_pairs))
    return search_pairs


if __name__ == "__main__":
    load_search_pairs()
