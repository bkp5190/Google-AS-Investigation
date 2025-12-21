from dataclasses import dataclass
import gzip
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_DIR = "./data"

with gzip.open(f"{DATA_DIR}/gooaq_pairs.jsonl.gz", "rb") as file:
    for count, line in enumerate(file):
        line_str = line.decode("utf-8").strip()
        if line_str:
            # Each object is a tuple of sentences
            obj = json.loads(line_str)


@dataclass
class PairsDataset:
    source: str  # The source text the user typed in
    target: str  # The target autosuggest result retrieved
