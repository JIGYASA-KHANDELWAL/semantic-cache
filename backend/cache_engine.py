
import time

from backend.embeddings import EmbeddingGenerator
from backend.vector_store import VectorStore
from backend.llm_client import generate_response


class SemanticCache:
    """
    Semantic caching engine.

    Checks whether a similar query already exists.
    If found, returns its cached response.
    Otherwise, calls Gemini and stores the new response.
    """

    def __init__(self, similarity_threshold=0.70):

        self.similarity_threshold = similarity_threshold

        # Initialize embedding model
        self.embedding_model = EmbeddingGenerator()

        # Initialize FAISS vector store
        self.vector_store = VectorStore(
            dimension=self.embedding_model.dimension
        )

        # Responses correspond to FAISS index positions
        self.responses = []

        # Basic analytics
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def get_response(self, query: str) -> dict:

        start_time = time.perf_counter()

        self.total_requests += 1

        # Step 1: Generate query embedding
        query_embedding = self.embedding_model.generate_embedding(
            query
        )

        # Step 2: Search existing cache
        matches = self.vector_store.search(
            query_embedding,
            top_k=1
        )

        # Step 3: Check similarity threshold
        if matches:

            best_match = matches[0]

            similarity = best_match["similarity"]

            if similarity >= self.similarity_threshold:

                self.cache_hits += 1

                index = best_match["index"]

                cached_response = self.responses[index]

                latency_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                return {
                    "query": query,
                    "answer": cached_response,
                    "cache_status": "HIT",
                    "matched_query": best_match["query"],
                    "similarity": round(similarity, 4),
                    "latency_ms": round(latency_ms, 2),
                    "llm_latency_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }

        # Step 4: Cache miss - call Gemini
        self.cache_misses += 1

        llm_result = generate_response(query)

        # Step 5: Store query embedding in FAISS
        self.vector_store.add(
            query,
            query_embedding
        )

        # Step 6: Store corresponding response
        self.responses.append(llm_result["answer"])

        total_latency = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "query": query,
            "answer": llm_result["answer"],
            "cache_status": "MISS",
            "matched_query": None,
            "similarity": (
                round(matches[0]["similarity"], 4)
                if matches else None
            ),
            "latency_ms": round(total_latency, 2),
            "llm_latency_ms": llm_result["latency_ms"],
            "prompt_tokens": llm_result["prompt_tokens"],
            "completion_tokens": llm_result["completion_tokens"],
            "total_tokens": llm_result["total_tokens"]
        }

    def get_statistics(self):

        hit_rate = (
            self.cache_hits / self.total_requests * 100
            if self.total_requests else 0
        )

        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(hit_rate, 2),
            "cache_size": self.vector_store.size()
        }
