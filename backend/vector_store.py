
import faiss
import numpy as np


class VectorStore:
    """
    FAISS-based vector store for semantic similarity search.
    """

    def __init__(self, dimension: int):

        self.dimension = dimension

        # Inner product on normalized vectors = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

        # Maps FAISS positions to original query strings
        self.texts = []

    def add(self, text: str, embedding: np.ndarray):
        """
        Add an embedding and its corresponding query.
        """

        vector = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if vector.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        faiss.normalize_L2(vector)

        self.index.add(vector)

        self.texts.append(text)

    def search(self, embedding: np.ndarray, top_k: int = 3):
        """
        Find the top-k most similar stored queries.
        """

        if self.index.ntotal == 0:
            return []

        vector = np.asarray(
            embedding,
            dtype=np.float32
        ).reshape(1, -1)

        if vector.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch")

        faiss.normalize_L2(vector)

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(vector, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            results.append({
                "query": self.texts[idx],
                "similarity": float(score),
                "index": int(idx)
            })

        return results

    def size(self):
        """
        Return the total number of stored vectors.
        """

        return self.index.ntotal
