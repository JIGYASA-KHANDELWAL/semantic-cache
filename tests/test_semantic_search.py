
from backend.embeddings import EmbeddingGenerator
from backend.vector_store import VectorStore


def main():

    # Step 1: Initialize embedding model
    embedding_model = EmbeddingGenerator()

    # Step 2: Initialize FAISS vector store
    vector_store = VectorStore(
        dimension=embedding_model.dimension
    )

    # Step 3: Add sample queries
    cached_queries = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "What is the capital of India?",
        "How do neural networks work?",
        "What is deep learning?"
    ]

    for query in cached_queries:

        embedding = embedding_model.generate_embedding(query)

        vector_store.add(query, embedding)

    print("\nTotal stored vectors:", vector_store.size())

    # Step 4: Search using a new query
    new_query = "Can you explain AI?"

    query_embedding = embedding_model.generate_embedding(
        new_query
    )

    results = vector_store.search(
        query_embedding,
        top_k=3
    )

    # Step 5: Display similarity results
    print("\nNew Query:", new_query)
    print("\nMost Similar Queries:")

    for result in results:

        print("\nMatched Query:", result["query"])

        print(
            "Similarity:",
            round(result["similarity"], 4)
        )


if __name__ == "__main__":
    main()
