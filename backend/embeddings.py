
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Converts text queries into numerical embeddings
    using a pre-trained Sentence Transformer model.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")

    def generate_embedding(self, text: str):
        """
        Convert a single text query into a normalized
        numerical vector.
        """

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype("float32")

    def generate_embeddings(self, texts: list[str]):
        """
        Generate embeddings for multiple text queries.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.astype("float32")
