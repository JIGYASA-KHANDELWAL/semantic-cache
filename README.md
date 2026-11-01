# Semantic Cache

An intelligent semantic caching layer for Large Language Model (LLM) applications that reduces redundant API calls by identifying semantically similar queries and reusing previously generated responses.

## Overview

Traditional caching relies on exact query matching. Semantic Cache uses vector embeddings and similarity search to identify queries with similar meanings, even when their wording differs.

## Key Features

- Semantic similarity-based query matching
- Vector search using FAISS
- Embedding generation using Sentence Transformers
- LLM integration using Gemini API
- Persistent cache storage using SQLite
- Configurable similarity threshold
- Cache hit rate and latency measurement
- API cost savings analysis
- Interactive analytics dashboard

## Tech Stack

Python | FastAPI | Sentence Transformers | FAISS | SQLite | Gemini API | Streamlit

## Architecture

User Query → Embedding Generation → Semantic Search → Cache Hit / Cache Miss → Response

## Performance Metrics

- Cache Hit Rate
- Average Response Latency
- Latency Reduction
- API Cost Savings
- Cache Precision

## Project Status

Currently under development.

## Author

Developed as an AI Engineering project.
