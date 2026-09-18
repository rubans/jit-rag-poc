"""
scripts/test_implicit_cache.py
Tests pure implicit prompt caching in Gemini Flash on a 50-page PDF (>4,096 tokens).
Does NOT use any explicit caching (no client.caches.create, no TTL, no storage fees).
"""

import os
import sys
import time
from pathlib import Path
from google import genai
from google.genai import types


def test_implicit_caching():
    pdf_path = Path("data/documents/sample_50page.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        sys.exit(1)

    project_id = os.environ.get("GCP_PROJECT_ID", "vertex-ai-poc-416620")
    model_name = os.environ.get("DIRECT_GEMINI_MODEL", "gemini-3.8-flash")
    location = "global" if "3." in model_name else "us-central1"

    print(f"=== Testing Pure Implicit (Automatic) Prompt Caching ===")
    print(f"Project ID   : {project_id}")
    print(f"Model Name   : {model_name}")
    print(f"Location     : {location}")
    print(f"Document     : {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    print(f"Caching Mode : 100% PURE IMPLICIT (Zero explicit cache objects)")
    print("-" * 65)

    client = genai.Client(vertexai=True, project=project_id, location=location)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")

    sys_instruction = (
        "You are an enterprise research assistant. Answer accurately based ONLY on the attached document. "
        "Be concise and cite specific pages, tables, or sections."
    )
    config = types.GenerateContentConfig(system_instruction=sys_instruction)

    queries = [
        "What was the consolidated revenue of Vertex Global Enterprise in FY 2024?",
        "What was the operating margin expansion reported in FY 2024?",
        "What percentage of core analytical workloads were migrated to BigQuery?",
        "What was the customer churn rate reported in Q4 2024?",
        "What was the Global Data Center Power Usage Effectiveness (PUE) achieved in 2024?",
    ]

    results = []

    for i, q in enumerate(queries, 1):
        print(f"\n[Turn {i}/5] Query: {q}")
        t0 = time.perf_counter()

        # Strict static-first prefix: [pdf_part] with trailing question
        response = client.models.generate_content(
            model=model_name,
            contents=[pdf_part, f"Question: {q}\n\nAnswer:"],
            config=config
        )
        latency_s = time.perf_counter() - t0

        prompt_tokens = 0
        cached_tokens = 0
        output_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            raw_cached = getattr(response.usage_metadata, "cached_content_token_count", 0)
            if isinstance(raw_cached, (int, float)):
                cached_tokens = int(raw_cached)

        hit_rate = (cached_tokens / prompt_tokens * 100.0) if prompt_tokens > 0 else 0.0

        ans_snippet = (response.text or "").strip().replace("\n", " ")[:90]
        print(f"  -> Answer : {ans_snippet}...")
        print(f"  -> Latency: {latency_s:.2f}s")
        print(f"  -> Tokens : Prompt={prompt_tokens}, Cached={cached_tokens} ({hit_rate:.1f}% hit rate), Output={output_tokens}")

        results.append({
            "turn": i,
            "query": q,
            "latency_s": round(latency_s, 2),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "hit_rate_pct": round(hit_rate, 1),
            "output_tokens": output_tokens
        })

        time.sleep(1)

    print("\n" + "=" * 65)
    print("PURE IMPLICIT CACHE TEST SUMMARY:")
    print("=" * 65)
    print(f"{'Turn':<6} | {'Latency':<9} | {'Prompt Tokens':<14} | {'Cached Tokens':<14} | {'Hit Rate %':<10}")
    print("-" * 65)
    for r in results:
        status = "⚡ HIT" if r['cached_tokens'] > 0 else "❄️ COLD"
        print(f"{r['turn']:<6} | {r['latency_s']:>6.2f}s   | {r['prompt_tokens']:>13} | {r['cached_tokens']:>13} | {r['hit_rate_pct']:>8.1f}%  ({status})")
    print("=" * 65)


if __name__ == "__main__":
    test_implicit_caching()

