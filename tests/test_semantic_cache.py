
from backend.cache_engine import SemanticCache


def main():

    # Initialize semantic cache
    cache = SemanticCache(
        similarity_threshold=0.70
    )

    # Test queries
    queries = [
        "What is artificial intelligence?",
        "Can you explain AI?",
        "What is the capital of France?"
    ]

    for query in queries:

        print("\n" + "=" * 60)
        print("USER QUERY:", query)

        result = cache.get_response(query)

        print("\nCACHE STATUS:", result["cache_status"])

        print("MATCHED QUERY:", result["matched_query"])

        print("SIMILARITY:", result["similarity"])

        print("LATENCY:", result["latency_ms"], "ms")

        print("LLM LATENCY:", result["llm_latency_ms"], "ms")

        print("\nRESPONSE:")
        print(result["answer"])

    # Display cache statistics
    print("\n" + "=" * 60)
    print("CACHE STATISTICS")

    stats = cache.get_statistics()

    for key, value in stats.items():

        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
