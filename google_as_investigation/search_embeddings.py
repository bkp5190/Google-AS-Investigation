from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import logging
import numpy as np

from load_data import SearchPair, load_search_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = "TaylorAI/gte-tiny"


class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def encode(self, sentences: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )


def _embed_search_pairs(
    search_pairs: list[SearchPair], embedder: Embedder = Embedder(MODEL_NAME)
) -> tuple[np.ndarray, np.ndarray]:
    sources = [p.source for p in search_pairs]
    targets = [p.target for p in search_pairs]

    source_embs = embedder.encode(sources)
    logger.debug("Finished computing source embeddings")
    target_embs = embedder.encode(targets)
    logger.debug("Finished computing target embeddings")
    return source_embs, target_embs


if __name__ == "__main__":
    logger.debug("Loading in the search pairs dataset")
    search_pairs = load_search_pairs()

    source_embs, target_embs = _embed_search_pairs(search_pairs)

    # Combine
    embeddings = np.vstack([source_embs, target_embs])
    labels = np.array(["source"] * len(source_embs) + ["target"] * len(target_embs))

    # Visualize
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    emb_2d = reducer.fit_transform(embeddings)  # (N, 2)

    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=3, alpha=0.6)
    plt.show()
