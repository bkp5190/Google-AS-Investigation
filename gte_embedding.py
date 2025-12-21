from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODEL_NAME = "TaylorAI/gte-tiny"


# Lightweight wrapper over the huggingface SentenceTransformer
class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def encode(self, sentences: list[str]):
        return self.model.encode(sentences)

    def __str__(self) -> str:
        return f"{self.model_name}"


if __name__ == "__main__":
    embedder = Embedder(MODEL_NAME)
    sentences = [f"{i * 'hello world '}" for i in range(100)]
    logger.info(f"Length of sentences: {len(sentences)}")
    logging.info(f"First few: {sentences[0:5]}")
    logger.info(f"This is the embedder: {embedder}")
    logger.info("Computing embeddings...")
    embeddings = embedder.encode(sentences)
    logging.info(f"This is the shape of the resulting embeddings: {embeddings.shape}")
    kmeans = KMeans(n_clusters=1, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
    emb_2d = reducer.fit_transform(embeddings)  # (N, 2)

    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=3, alpha=0.6)
    plt.show()
