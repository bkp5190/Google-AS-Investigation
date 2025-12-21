from dataclasses import dataclass
import gzip
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DATA_DIR = "./data"


@dataclass
class SearchPair:
    source: str  # The source text the user typed in
    target: str  # The target autosuggest result retrieved


def load_search_pairs(data_dir: str = DATA_DIR, filename: str = "gooaq_pairs.jsonl.gz"):
    search_pairs: list[SearchPair] = []
    with gzip.open(f"{data_dir}/{filename}", "rb") as file:
        for line in file:
            line_str = line.decode("utf-8").strip()
            if line_str:
                # Each object is a tuple of sentences
                obj = json.loads(line_str)
                # logger.info(obj[0])
                search_pair = SearchPair(source=obj[0], target=obj[1])
                search_pairs.append(search_pair)

    return search_pairs
