
import os
import time

from dotenv import load_dotenv
from google import genai

# Load environment variables from the project root
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)


def generate_response(prompt: str) -> dict:
    """
    Generate a response using Gemini and measure
    latency and token usage.
    """

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    usage = response.usage_metadata

    prompt_tokens = (
        usage.prompt_token_count if usage else 0
    )

    completion_tokens = (
        usage.candidates_token_count if usage else 0
    )

    total_tokens = (
        usage.total_token_count if usage else 0
    )

    return {
        "answer": response.text or "",
        "model": MODEL,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }
