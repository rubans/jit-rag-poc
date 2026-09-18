"""
Main CLI entry point for comparative RAG benchmarking:
Method 1 (Modern Cloud-Native BigQuery with Autonomous AI.EMBED & AI.SEARCH)
vs. Method 2 (Legacy Gemini Flash Markdown)
vs. Method 3 (PyMuPDF4LLM Local CPU Layout Engine)
vs. Method 4 (Direct Long-Context Gemini Flash Zero-Embedding JIT-RAG)
on 50-page complex PDF with DeepEval metrics, position-stratified degradation, and cost break-even analysis.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List, Optional
from src.common.config import load_config
from src.legacy_flash.pipeline import FlashMarkdownRAGPipeline
from src.bigquery_native.pipeline import BigQueryNativeRAGPipeline
from src.pymupdf4llm_pipeline.pipeline import PyMuPDF4LLMRAGPipeline
from src.docling_pipeline.pipeline import DoclingRAGPipeline
from src.direct_gemini.pipeline import DirectGeminiFlashRAGPipeline
from src.eval.profiler import LatencyProfiler
from src.eval.cost_calculator import CostCalculator
from src.eval.deepeval_runner import DeepEvalBenchmarkRunner
from src.eval.reporter import BenchmarkReporter


def resolve_dataset_path(pdf_path: str, dataset_path: Optional[str] = None) -> str:
    """Resolves matching evaluation dataset based on target document if not explicitly specified."""
    if dataset_path and Path(dataset_path).exists():
        return dataset_path
    pdf_lower = str(pdf_path).lower()
    if "bp" in pdf_lower or "300" in pdf_lower:
        for candidate in ["data/eval_dataset_bp.json", "data/eval_dataset_300page.json", "data/eval_bp_dataset.json"]:
            if Path(candidate).exists():
                return candidate
    for candidate in ["data/eval_dataset_50page.json", "data/eval_dataset.json"]:
        if Path(candidate).exists():
            return candidate
    return dataset_path or "data/eval_dataset_50page.json"


METHOD_ALIASES: Dict[str, str] = {
    # Method 1: BigQuery Native (Zero-Copy)
    "1": "method1",
    "method1": "method1",
    "method1_bigquery": "method1",
    "method1-bigquery": "method1",
    "bigquery": "method1",
    "bigquery_native": "method1",
    "bigquery-native": "method1",
    "bq": "method1",
    "bq-zero-copy": "method1",
    "zero-copy": "method1",
    # Method 2: Gemini Flash Multimodal MD (Vision)
    "2": "method2",
    "method2": "method2",
    "method2_flash": "method2",
    "method2-flash": "method2",
    "method2_flash_vision": "method2",
    "method2-flash-vision": "method2",
    "flash": "method2",
    "legacy_flash": "method2",
    "legacy-flash": "method2",
    "gemini-flash-md": "method2",
    "flash-vision-md": "method2",
    "flash_vision_md": "method2",
    "multimodal-flash": "method2",
    # Method 3: PyMuPDF4LLM (Local CPU Layout)
    "3": "method3",
    "method3": "method3",
    "method3_pymupdf": "method3",
    "method3-pymupdf": "method3",
    "method3_pymupdf_cpu": "method3",
    "method3-pymupdf-cpu": "method3",
    "pymupdf": "method3",
    "pymupdf4llm": "method3",
    "pymupdf-cpu": "method3",
    "pymupdf_cpu": "method3",
    "local-cpu": "method3",
    "local_cpu": "method3",
    # Method 3b: Docling Local CPU (TableFormer)
    "3b": "method3b",
    "method3b": "method3b",
    "method3b_docling": "method3b",
    "method3b-docling": "method3b",
    "docling": "method3b",
    "docling_cpu": "method3b",
    "docling-cpu": "method3b",
    "docling-local": "method3b",
    # Method 4: Direct Long-Context Flash (Zero-Embedding)
    "4": "method4",
    "method4": "method4",
    "method4_direct": "method4",
    "method4-direct": "method4",
    "method4_direct_flash": "method4",
    "method4-direct-flash": "method4",
    "direct": "method4",
    "direct_gemini": "method4",
    "direct-gemini": "method4",
    "direct_flash": "method4",
    "direct-flash": "method4",
    "long-context": "method4",
    "long_context": "method4",
    "zero-embedding": "method4",
    "zero_embedding": "method4",
    "stateless": "method4",
    # Method 4b: Direct Long-Context Flash (Stateful Session Caching)
    "4b": "method4b",
    "method4b": "method4b",
    "method4_stateful": "method4b",
    "method4b_stateful": "method4b",
    "direct-stateful": "method4b",
    "stateful": "method4b",
    "session": "method4b",
    # Method 4c: Direct Long-Context Flash (Explicit Context Caching)
    "4c": "method4c",
    "method4c": "method4c",
    "method4_explicit": "method4c",
    "method4c_explicit": "method4c",
    "direct-explicit": "method4c",
    "explicit": "method4c",
    "explicit-cache": "method4c",
    # Direct Cache Comparison
    "compare-cache": "compare-cache",
    "compare-direct": "compare-cache",
    "compare-all-cache": "compare-all-cache",
    # All Methods
    "all": "all",
    "*": "all"
}

METHOD_DESCRIPTIONS: Dict[str, str] = {
    "method1": "Method 1: BigQuery Native (Zero-Copy Architecture, Document AI Layout Chunker, AI.EMBED)",
    "method2": "Method 2: Gemini Flash Multimodal MD (Page Vision OCR, text-embedding-004)",
    "method3": "Method 3: PyMuPDF4LLM (Local CPU Layout Parser, $0 Cloud Parsing Cost)",
    "method3b": "Method 3b: Docling Local CPU (TableFormer Layout Parser, IBM Research)",
    "method4": "Method 4: Direct Long-Context Gemini Flash (Stateless Prefix Caching)",
    "method4b": "Method 4b: Direct Long-Context Gemini Flash (Stateful Session Caching)",
    "method4c": "Method 4c: Direct Long-Context Gemini Flash (Explicit Context Caching, client.caches.create)",
    "compare-cache": "Compare Method 4 (Stateless) vs Method 4b (Stateful Session Caching)",
    "compare-all-cache": "Compare Method 4 (Stateless), Method 4b (Stateful), and Method 4c (Explicit Caching)",
    "all": "All Pipelines (Methods 1, 2, 3, 3b, 4 Stateless, 4b Stateful, 4c Explicit)"
}


def normalize_method_name(method: str) -> str:
    """Normalizes any method alias or shortcode into canonical identifiers."""
    norm = str(method).strip().lower().replace(" ", "-")
    if norm in METHOD_ALIASES:
        return METHOD_ALIASES[norm]
    supported = "method1, method2, method3, method3b, method4 (stateless), method4b (stateful), method4c (explicit), compare-cache, compare-all-cache, all"
    raise ValueError(f"Unknown method '{method}'. Supported methods: {supported}")


def run_benchmark(
    pdf_path: str = "data/documents/sample_50page.pdf",
    dataset_path: Optional[str] = None,
    project_id: Optional[str] = None,
    top_k: int = 5,
    output_report: str = "output/benchmark_report.md",
    max_eval_items: int = None,
    method: str = "all",
    run_id: Optional[str] = None,
    auto_cleanup: bool = True
) -> Dict[str, Any]:
    project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
    norm_method = normalize_method_name(method)
    dataset_path = resolve_dataset_path(pdf_path, dataset_path)
    print("=" * 80)
    print(f"STARTING COMPARATIVE RAG BENCHMARK")
    print(f"Target Project : {project_id}")
    print(f"Test Document  : {pdf_path}")
    print(f"Golden Dataset : {dataset_path}")
    print(f"Method Filter  : {norm_method} ({METHOD_DESCRIPTIONS.get(norm_method, norm_method)})")
    print("=" * 80)

    config = load_config()
    cost_calc = CostCalculator(config)
    eval_runner = DeepEvalBenchmarkRunner(threshold=0.70)

    # 1. Load Golden Dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_items = json.load(f)
    if max_eval_items:
        eval_items = eval_items[:max_eval_items]
    print(f"\nLoaded {len(eval_items)} evaluation test cases.\n")

    # Load existing report metrics if available (for partial re-runs)
    existing_report = {}
    json_path = Path(output_report).with_suffix(".json")
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_report = json.load(f)
        except Exception:
            pass

    # 2. Ingestion Phase
    print("--- [STAGE 1: Ingestion & Indexing] ---")
    
    # Method 2: Legacy Flash Markdown Ingestion
    flash_pipeline = None
    flash_ingest_metrics = None
    flash_client_time_s = 0.0
    flash_embed_time_s = 0.0
    flash_ingest_time_s = 0.0

    if norm_method in ("all", "method2"):
        print("\n[Method 2] Ingesting PDF with Gemini Flash Page-Split Pipeline (Isolated Run ID, Auto-Cleaned Cache)...")
        flash_pipeline = FlashMarkdownRAGPipeline(
            project_id=project_id,
            use_cache=False,
            run_id=run_id,
            auto_cleanup=auto_cleanup
        )
        print(f"  -> Run ID: {flash_pipeline.run_id}")
        print(f"  -> Run Artifacts Directory: {flash_pipeline.work_dir}")
        t_flash_ingest_start = time.perf_counter()
        flash_ingest_metrics = flash_pipeline.ingest(pdf_path)
        flash_ingest_time_s = time.perf_counter() - t_flash_ingest_start
        flash_embed_time_s = flash_ingest_metrics.embedding_latency_ms / 1000.0
        flash_client_time_s = max(0.0, flash_ingest_time_s - flash_embed_time_s)

        print(f"  -> Client Ingestion (Split + Vision MD): {flash_client_time_s:.2f}s")
        print(f"  -> Local Vector Embedding Time: {flash_embed_time_s:.2f}s")
        print(f"  -> Total E2E Ingestion Time (100% Ready): {flash_ingest_time_s:.2f}s")
        print(f"  -> Total Chunks Indexed: {len(flash_pipeline.chunks)}")
        print(f"  -> Ingestion Cost: ${flash_ingest_metrics.total_cost_usd:.4f}")

    # Method 1: Modern BigQuery Ingestion (Autonomous AI.EMBED)
    bq_pipeline = None
    bq_ingest_metrics = None
    bq_client_time_s = 0.0
    bq_embed_ready_time_s = 0.0
    bq_ingest_time_s = 0.0

    if norm_method in ("all", "method1"):
        print("\n[Method 1] Ingesting via BigQuery Object Tables & Autonomous AI.EMBED...")
        bq_pipeline = BigQueryNativeRAGPipeline(project_id=project_id)
        t_bq_ingest_start = time.perf_counter()
        bq_ingest_metrics = bq_pipeline.ingest(pdf_path, wait_for_embeddings=True)
        bq_ingest_time_s = time.perf_counter() - t_bq_ingest_start
        bq_client_time_s = bq_ingest_metrics.ingestion_latency_ms / 1000.0
        bq_embed_ready_time_s = bq_ingest_metrics.embedding_latency_ms / 1000.0

        print(f"  -> Client Ingestion Time (Synchronous): {bq_client_time_s:.2f}s")
        print(f"  -> Embedding 100% Ready Time (Autonomous Queue): {bq_embed_ready_time_s:.2f}s")
        print(f"  -> Total E2E Ingestion Time (100% Complete): {bq_ingest_time_s:.2f}s")
        print(f"  -> Ingestion Cost: ${bq_ingest_metrics.total_cost_usd:.4f}")

    # Method 3: PyMuPDF4LLM Local CPU Ingestion
    mupdf_pipeline = None
    mupdf_ingest_metrics = None
    mupdf_parse_time_s = 0.0
    mupdf_embed_time_s = 0.0
    mupdf_ingest_time_s = 0.0

    if norm_method in ("all", "method3"):
        print("\n[Method 3] Ingesting PDF with PyMuPDF4LLM (Local CPU Layout Engine, Zero Cloud Parsing Cost)...")
        mupdf_pipeline = PyMuPDF4LLMRAGPipeline(
            project_id=project_id,
            run_id=run_id,
            auto_cleanup=auto_cleanup
        )
        print(f"  -> Run ID: {mupdf_pipeline.run_id}")
        print(f"  -> Run Artifacts Directory: {mupdf_pipeline.work_dir}")
        t_mupdf_start = time.perf_counter()
        mupdf_ingest_metrics = mupdf_pipeline.ingest(pdf_path)
        mupdf_ingest_time_s = time.perf_counter() - t_mupdf_start
        mupdf_embed_time_s = mupdf_ingest_metrics.embedding_latency_ms / 1000.0
        mupdf_parse_time_s = max(0.0, mupdf_ingest_time_s - mupdf_embed_time_s)

        print(f"  -> Local CPU Parse Time: {mupdf_parse_time_s:.2f}s")
        print(f"  -> Local Vector Embedding Time: {mupdf_embed_time_s:.2f}s")
        print(f"  -> Total E2E Ingestion Time (100% Ready): {mupdf_ingest_time_s:.2f}s")
        print(f"  -> Total Chunks Indexed: {len(mupdf_pipeline.chunks)}")
        print(f"  -> Ingestion Cost: ${mupdf_ingest_metrics.total_cost_usd:.4f}")

    # Method 3b: Docling Local CPU Ingestion (TableFormer Layout Parser)
    docling_pipeline = None
    docling_ingest_metrics = None
    docling_parse_time_s = 0.0
    docling_embed_time_s = 0.0
    docling_ingest_time_s = 0.0

    if norm_method in ("all", "method3b"):
        print("\n[Method 3b] Ingesting PDF with Docling (TableFormer Layout Engine, Local CPU)...")
        docling_pipeline = DoclingRAGPipeline(
            project_id=project_id,
            run_id=run_id,
            auto_cleanup=auto_cleanup
        )
        print(f"  -> Run ID: {docling_pipeline.run_id}")
        t_docling_start = time.perf_counter()
        docling_ingest_metrics = docling_pipeline.ingest(pdf_path)
        docling_ingest_time_s = time.perf_counter() - t_docling_start
        docling_embed_time_s = docling_ingest_metrics.embedding_latency_ms / 1000.0
        docling_parse_time_s = docling_ingest_metrics.ingestion_latency_ms / 1000.0

        print(f"  -> Local CPU Parse Time (TableFormer): {docling_parse_time_s:.2f}s")
        print(f"  -> Local Vector Embedding Time: {docling_embed_time_s:.2f}s")
        print(f"  -> Total E2E Ingestion Time (100% Ready): {docling_ingest_time_s:.2f}s")
        print(f"  -> Total Chunks Indexed: {len(docling_pipeline.chunks)}")
        print(f"  -> Ingestion Cost: ${docling_ingest_metrics.total_cost_usd:.4f}")

    # Method 4: Direct Long-Context Gemini Flash (Zero-Embedding)
    direct_pipeline = None
    # Method 4: Direct Long-Context Gemini Flash (Stateless Prefix Caching)
    direct_pipeline = None
    direct_ingest_metrics = None
    direct_ingest_time_s = 0.0

    if norm_method in ("all", "method4", "compare-cache"):
        print("\n[Method 4] Ingesting PDF via Direct Long-Context Gemini Flash (Stateless Prefix Caching)...")
        direct_pipeline = DirectGeminiFlashRAGPipeline(
            project_id=project_id,
            run_id=run_id,
            auto_cleanup=auto_cleanup,
            stateful=False
        )
        print(f"  -> Run ID: {direct_pipeline.run_id}")
        print(f"  -> Model Name: {direct_pipeline.model_name}")
        t_direct_start = time.perf_counter()
        direct_ingest_metrics = direct_pipeline.ingest(pdf_path)
        direct_ingest_time_s = time.perf_counter() - t_direct_start

        print(f"  -> Client Ingestion (File read / Part payload): {direct_ingest_time_s:.4f}s")
        print(f"  -> Total E2E Ingestion Time (Instant Ready): {direct_ingest_time_s:.4f}s")
        print(f"  -> Total Ingestion Cost: ${direct_ingest_metrics.total_cost_usd:.4f} (Zero indexing cost)")

    # Method 4b: Direct Gemini Flash (Stateful Session Caching)
    direct_stateful_pipeline = None
    direct_stateful_ingest_metrics = None
    direct_stateful_ingest_time_s = 0.0

    if norm_method in ("all", "method4b", "compare-cache"):
        print("\n[Method 4b] Ingesting PDF via Direct Gemini Flash (Stateful Session Caching)...")
        direct_stateful_pipeline = DirectGeminiFlashRAGPipeline(
            project_id=project_id,
            run_id=run_id,
            auto_cleanup=auto_cleanup,
            stateful=True
        )
        print(f"  -> Run ID: {direct_stateful_pipeline.run_id}")
        print(f"  -> Model Name: {direct_stateful_pipeline.model_name}")
        t_d4b_start = time.perf_counter()
        direct_stateful_ingest_metrics = direct_stateful_pipeline.ingest(pdf_path)
        direct_stateful_ingest_time_s = time.perf_counter() - t_d4b_start

        print(f"  -> Client Ingestion (File read / Part payload): {direct_stateful_ingest_time_s:.4f}s")
        print(f"  -> Total E2E Ingestion Time (Instant Ready): {direct_stateful_ingest_time_s:.4f}s")
        print(f"  -> Total Ingestion Cost: ${direct_stateful_ingest_metrics.total_cost_usd:.4f} (Zero indexing cost)")

    # Method 4c: Direct Gemini Flash (Explicit Context Caching)
    direct_explicit_pipeline = None
    direct_explicit_ingest_metrics = None
    direct_explicit_ingest_time_s = 0.0

    if norm_method in ("all", "method4c", "compare-all-cache"):
        print("\n[Method 4c] Ingesting PDF via Direct Gemini Flash (Explicit Context Caching)...")
        direct_explicit_pipeline = DirectGeminiFlashRAGPipeline(
            project_id=project_id,
            run_id=run_id,
            auto_cleanup=auto_cleanup,
            explicit_cache=True,
            cache_ttl="1800s"
        )
        print(f"  -> Run ID: {direct_explicit_pipeline.run_id}")
        print(f"  -> Model Name: {direct_explicit_pipeline.model_name}")
        t_d4c_start = time.perf_counter()
        direct_explicit_ingest_metrics = direct_explicit_pipeline.ingest(pdf_path)
        direct_explicit_ingest_time_s = time.perf_counter() - t_d4c_start

        c_name = direct_explicit_ingest_metrics.cost_breakdown.get("explicit_cache_name", "N/A")
        print(f"  -> Explicit Cache Resource: {c_name}")
        print(f"  -> Client Ingestion / Cache Setup: {direct_explicit_ingest_time_s:.4f}s")
        print(f"  -> Total E2E Ingestion Time (Locked in TPU HBM): {direct_explicit_ingest_time_s:.4f}s")
        print(f"  -> Ingestion Cost: ${direct_explicit_ingest_metrics.total_cost_usd:.4f} ($0 upfront indexing)")

    # 3. Query & DeepEval Evaluation Phase
    print("\n--- [STAGE 2: Query Execution & DeepEval Scoring] ---")
    
    profiler_flash = LatencyProfiler()
    profiler_bq = LatencyProfiler()
    profiler_mupdf = LatencyProfiler()
    profiler_docling = LatencyProfiler()
    profiler_direct = LatencyProfiler()
    profiler_direct_stateful = LatencyProfiler()
    profiler_direct_explicit = LatencyProfiler()

    flash_scores_list: List[Dict[str, Any]] = []
    bq_scores_list: List[Dict[str, Any]] = []
    mupdf_scores_list: List[Dict[str, Any]] = []
    docling_scores_list: List[Dict[str, Any]] = []
    direct_scores_list: List[Dict[str, Any]] = []
    direct_stateful_scores_list: List[Dict[str, Any]] = []
    direct_explicit_scores_list: List[Dict[str, Any]] = []

    try:
        for idx, item in enumerate(eval_items, 1):
            q = item["input"]
            expected = item["expected_output"]
            page_num = item.get("page", 1)
            gt_context = item["context_ground_truth"]
            print(f"  Evaluating query {idx}/{len(eval_items)} (p.{page_num}): {q[:55]}...")

            # Run Method 2 (Gemini Flash Multimodal MD)
            if norm_method in ("all", "method2") and flash_pipeline:
                t0 = time.perf_counter()
                flash_ans, flash_ret, flash_m = flash_pipeline.query(q, top_k=top_k)
                flash_lat_ms = (time.perf_counter() - t0) * 1000
                profiler_flash.record("end_to_end", flash_lat_ms)
                flash_contexts = [r.chunk.content for r in flash_ret]
                
                flash_eval = eval_runner.evaluate_test_case(q, flash_ans, expected, flash_contexts)
                flash_scores_list.append({"page": page_num, "scores": flash_eval})

            # Run Method 1 (BigQuery Native Zero-Copy with AI.SEARCH)
            if norm_method in ("all", "method1") and bq_pipeline:
                t1 = time.perf_counter()
                bq_ans, bq_ret, bq_m = bq_pipeline.query(q, top_k=top_k)
                bq_lat_ms = (time.perf_counter() - t1) * 1000
                profiler_bq.record("end_to_end", bq_lat_ms)
                bq_contexts = [r.chunk.content for r in bq_ret]

                bq_eval = eval_runner.evaluate_test_case(q, bq_ans, expected, bq_contexts)
                bq_scores_list.append({"page": page_num, "scores": bq_eval})

            # Run Method 3 (PyMuPDF4LLM Local CPU)
            if norm_method in ("all", "method3") and mupdf_pipeline:
                t2 = time.perf_counter()
                mu_ans, mu_ret, mu_m = mupdf_pipeline.query(q, top_k=top_k)
                mu_lat_ms = (time.perf_counter() - t2) * 1000
                profiler_mupdf.record("end_to_end", mu_lat_ms)
                mu_contexts = [r.chunk.content for r in mu_ret]

                mu_eval = eval_runner.evaluate_test_case(q, mu_ans, expected, mu_contexts)
                mupdf_scores_list.append({"page": page_num, "scores": mu_eval})

            # Run Method 3b (Docling Local CPU)
            if norm_method in ("all", "method3b") and docling_pipeline:
                t3b = time.perf_counter()
                d_ans, d_ret, d_m = docling_pipeline.query(q, top_k=top_k)
                d_lat_ms = (time.perf_counter() - t3b) * 1000
                profiler_docling.record("end_to_end", d_lat_ms)
                d_contexts = [r.chunk.content for r in d_ret]

                d_eval = eval_runner.evaluate_test_case(q, d_ans, expected, d_contexts)
                docling_scores_list.append({"page": page_num, "scores": d_eval})

            # Run Method 4 (Direct Long-Context Gemini Flash - Stateless)
            if norm_method in ("all", "method4", "compare-cache", "compare-all-cache", "compare-direct") and direct_pipeline:
                t3 = time.perf_counter()
                direct_ans, direct_ret, direct_m = direct_pipeline.query(q)
                direct_lat_ms = (time.perf_counter() - t3) * 1000
                profiler_direct.record("end_to_end", direct_lat_ms)
                direct_contexts = [r.chunk.content for r in direct_ret]

                direct_eval = eval_runner.evaluate_test_case(q, direct_ans, expected, direct_contexts)
                c_tokens = direct_m.cost_breakdown.get("cached_tokens", 0)
                c_rate = direct_m.cost_breakdown.get("cache_hit_rate_pct", 0.0)
                direct_scores_list.append({
                    "page": page_num,
                    "scores": direct_eval,
                    "cached_tokens": c_tokens,
                    "cache_hit_rate_pct": c_rate
                })
                print(f"      [Method 4]  Cached: {c_tokens:,} tok ({c_rate:.1f}% hit) | Latency: {direct_lat_ms/1000:.1f}s")

            # Run Method 4b (Direct Gemini Flash - Stateful Session Caching)
            if norm_method in ("all", "method4b", "compare-cache", "compare-all-cache", "compare-direct") and direct_stateful_pipeline:
                t4b = time.perf_counter()
                d4b_ans, d4b_ret, d4b_m = direct_stateful_pipeline.query(q)
                d4b_lat_ms = (time.perf_counter() - t4b) * 1000
                profiler_direct_stateful.record("end_to_end", d4b_lat_ms)
                d4b_contexts = [r.chunk.content for r in d4b_ret]

                d4b_eval = eval_runner.evaluate_test_case(q, d4b_ans, expected, d4b_contexts)
                c_tokens_b = d4b_m.cost_breakdown.get("cached_tokens", 0)
                c_rate_b = d4b_m.cost_breakdown.get("cache_hit_rate_pct", 0.0)
                direct_stateful_scores_list.append({
                    "page": page_num,
                    "scores": d4b_eval,
                    "cached_tokens": c_tokens_b,
                    "cache_hit_rate_pct": c_rate_b
                })
                print(f"      [Method 4b] Cached: {c_tokens_b:,} tok ({c_rate_b:.1f}% hit) | Latency: {d4b_lat_ms/1000:.1f}s")

            # Run Method 4c (Direct Gemini Flash - Explicit Context Caching)
            if norm_method in ("all", "method4c", "compare-all-cache") and direct_explicit_pipeline:
                t4c = time.perf_counter()
                d4c_ans, d4c_ret, d4c_m = direct_explicit_pipeline.query(q)
                d4c_lat_ms = (time.perf_counter() - t4c) * 1000
                profiler_direct_explicit.record("end_to_end", d4c_lat_ms)
                d4c_contexts = [r.chunk.content for r in d4c_ret]

                d4c_eval = eval_runner.evaluate_test_case(q, d4c_ans, expected, d4c_contexts)
                c_tokens_c = d4c_m.cost_breakdown.get("cached_tokens", 0)
                c_rate_c = d4c_m.cost_breakdown.get("cache_hit_rate_pct", 0.0)
                direct_explicit_scores_list.append({
                    "page": page_num,
                    "scores": d4c_eval,
                    "cached_tokens": c_tokens_c,
                    "cache_hit_rate_pct": c_rate_c
                })
                print(f"      [Method 4c] Cached: {c_tokens_c:,} tok ({c_rate_c:.1f}% hit) | Latency: {d4c_lat_ms/1000:.1f}s")
    finally:
        if direct_explicit_pipeline:
            print("\n[Method 4c] Cleaning up explicit cache resource...")
            direct_explicit_pipeline.cleanup()
            print("  -> Explicit cache resource deleted (Zero ongoing storage fees).")

    # 4. Compile Metrics & Position-Stratified Summaries
    flash_lat_summary = profiler_flash.get_summary()
    bq_lat_summary = profiler_bq.get_summary()
    mupdf_lat_summary = profiler_mupdf.get_summary()
    docling_lat_summary = profiler_docling.get_summary()
    direct_lat_summary = profiler_direct.get_summary()
    direct_stateful_lat_summary = profiler_direct_stateful.get_summary()
    direct_explicit_lat_summary = profiler_direct_explicit.get_summary()

    def avg_score(scores: List[Dict[str, Any]], key: str) -> float:
        vals = []
        for s in scores:
            d = s.get("scores", s)
            if key in d:
                vals.append(d[key])
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    # Compute position-stratified degradation metrics
    flash_pos = eval_runner.compute_position_stratified_metrics(flash_scores_list) if flash_scores_list else {}
    bq_pos = eval_runner.compute_position_stratified_metrics(bq_scores_list) if bq_scores_list else {}
    mupdf_pos = eval_runner.compute_position_stratified_metrics(mupdf_scores_list) if mupdf_scores_list else {}
    docling_pos = eval_runner.compute_position_stratified_metrics(docling_scores_list) if docling_scores_list else {}
    direct_pos = eval_runner.compute_position_stratified_metrics(direct_scores_list) if direct_scores_list else {}
    direct_stateful_pos = eval_runner.compute_position_stratified_metrics(direct_stateful_scores_list) if direct_stateful_scores_list else {}
    direct_explicit_pos = eval_runner.compute_position_stratified_metrics(direct_explicit_scores_list) if direct_explicit_scores_list else {}

    bq_payload = existing_report.get("method1_bigquery") or existing_report.get("bigquery_native", {})
    if norm_method in ("all", "method1") and bq_ingest_metrics:
        bq_payload = {
            "ingestion_cost_usd": bq_ingest_metrics.total_cost_usd,
            "client_ingest_time_s": round(bq_client_time_s, 2),
            "embedding_ready_time_s": round(bq_embed_ready_time_s, 2),
            "ingestion_time_s": round(bq_ingest_time_s, 2),
            "cost_per_query_usd": cost_calc.calculate_query_cost("bigquery"),
            "query_p50_ms": bq_lat_summary["end_to_end"]["p50"],
            "query_p90_ms": bq_lat_summary["end_to_end"]["p90"],
            "query_p99_ms": bq_lat_summary["end_to_end"]["p99"],
            "faithfulness": avg_score(bq_scores_list, "faithfulness"),
            "answer_relevancy": avg_score(bq_scores_list, "answer_relevancy"),
            "contextual_precision": avg_score(bq_scores_list, "contextual_precision"),
            "contextual_recall": avg_score(bq_scores_list, "contextual_recall"),
            "contextual_relevancy": avg_score(bq_scores_list, "contextual_relevancy"),
            "document_completeness": bq_pos.get("document_completeness", avg_score(bq_scores_list, "completeness")),
            "early_recall": bq_pos.get("early_recall", 0.0),
            "middle_recall": bq_pos.get("middle_recall", 0.0),
            "late_recall": bq_pos.get("late_recall", 0.0),
            "lost_in_middle_degradation_pct": bq_pos.get("lost_in_middle_degradation_pct", 0.0),
            "cost_breakdown": {
                "docai": bq_ingest_metrics.cost_breakdown.get("docai_processing_cost_usd", 1.50),
                "embedding": bq_ingest_metrics.cost_breakdown.get("embedding_cost_usd", 0.0031),
                "bq_compute": bq_ingest_metrics.cost_breakdown.get("bq_compute_cost_usd", 0.0003)
            }
        }

    flash_payload = existing_report.get("method2_flash_vision") or existing_report.get("legacy_flash", {})
    if norm_method in ("all", "method2") and flash_ingest_metrics:
        flash_payload = {
            "ingestion_cost_usd": flash_ingest_metrics.total_cost_usd,
            "client_ingest_time_s": round(flash_client_time_s, 2),
            "embedding_ready_time_s": round(flash_embed_time_s, 2),
            "ingestion_time_s": round(flash_ingest_time_s, 2),
            "cost_per_query_usd": cost_calc.calculate_query_cost("legacy"),
            "query_p50_ms": flash_lat_summary["end_to_end"]["p50"],
            "query_p90_ms": flash_lat_summary["end_to_end"]["p90"],
            "query_p99_ms": flash_lat_summary["end_to_end"]["p99"],
            "faithfulness": avg_score(flash_scores_list, "faithfulness"),
            "answer_relevancy": avg_score(flash_scores_list, "answer_relevancy"),
            "contextual_precision": avg_score(flash_scores_list, "contextual_precision"),
            "contextual_recall": avg_score(flash_scores_list, "contextual_recall"),
            "contextual_relevancy": avg_score(flash_scores_list, "contextual_relevancy"),
            "document_completeness": flash_pos.get("document_completeness", avg_score(flash_scores_list, "completeness")),
            "early_recall": flash_pos.get("early_recall", 0.0),
            "middle_recall": flash_pos.get("middle_recall", 0.0),
            "late_recall": flash_pos.get("late_recall", 0.0),
            "lost_in_middle_degradation_pct": flash_pos.get("lost_in_middle_degradation_pct", 0.0),
            "cost_breakdown": {
                "vision": flash_ingest_metrics.cost_breakdown.get("page_conversion_cost_usd", 0.010),
                "markdown_output": 0.009,
                "embedding": flash_ingest_metrics.cost_breakdown.get("embedding_cost_usd", 0.0031)
            }
        }

    mupdf_payload = existing_report.get("method3_pymupdf_cpu") or existing_report.get("pymupdf4llm", {})
    if norm_method in ("all", "method3") and mupdf_ingest_metrics:
        mupdf_payload = {
            "ingestion_cost_usd": mupdf_ingest_metrics.total_cost_usd,
            "client_ingest_time_s": round(mupdf_parse_time_s, 2),
            "embedding_ready_time_s": round(mupdf_embed_time_s, 2),
            "ingestion_time_s": round(mupdf_ingest_time_s, 2),
            "cost_per_query_usd": cost_calc.calculate_query_cost("legacy"),
            "query_p50_ms": mupdf_lat_summary["end_to_end"]["p50"],
            "query_p90_ms": mupdf_lat_summary["end_to_end"]["p90"],
            "query_p99_ms": mupdf_lat_summary["end_to_end"]["p99"],
            "faithfulness": avg_score(mupdf_scores_list, "faithfulness"),
            "answer_relevancy": avg_score(mupdf_scores_list, "answer_relevancy"),
            "contextual_precision": avg_score(mupdf_scores_list, "contextual_precision"),
            "contextual_recall": avg_score(mupdf_scores_list, "contextual_recall"),
            "contextual_relevancy": avg_score(mupdf_scores_list, "contextual_relevancy"),
            "document_completeness": mupdf_pos.get("document_completeness", avg_score(mupdf_scores_list, "completeness")),
            "early_recall": mupdf_pos.get("early_recall", 0.0),
            "middle_recall": mupdf_pos.get("middle_recall", 0.0),
            "late_recall": mupdf_pos.get("late_recall", 0.0),
            "lost_in_middle_degradation_pct": mupdf_pos.get("lost_in_middle_degradation_pct", 0.0),
            "cost_breakdown": {
                "parsing": 0.0,
                "embedding_cost_usd": mupdf_ingest_metrics.cost_breakdown.get("embedding_cost_usd", 0.0031)
            }
        }

    docling_payload = existing_report.get("method3b_docling", {})
    if norm_method in ("all", "method3b") and docling_ingest_metrics:
        docling_payload = {
            "ingestion_cost_usd": docling_ingest_metrics.total_cost_usd,
            "client_ingest_time_s": round(docling_parse_time_s, 2),
            "embedding_ready_time_s": round(docling_embed_time_s, 2),
            "ingestion_time_s": round(docling_parse_time_s + docling_embed_time_s, 2),
            "cost_per_query_usd": cost_calc.calculate_query_cost("legacy"),
            "query_p50_ms": docling_lat_summary["end_to_end"]["p50"],
            "query_p90_ms": docling_lat_summary["end_to_end"]["p90"],
            "query_p99_ms": docling_lat_summary["end_to_end"]["p99"],
            "faithfulness": avg_score(docling_scores_list, "faithfulness"),
            "answer_relevancy": avg_score(docling_scores_list, "answer_relevancy"),
            "contextual_precision": avg_score(docling_scores_list, "contextual_precision"),
            "contextual_recall": avg_score(docling_scores_list, "contextual_recall"),
            "contextual_relevancy": avg_score(docling_scores_list, "contextual_relevancy"),
            "document_completeness": docling_pos.get("document_completeness", avg_score(docling_scores_list, "completeness")),
            "early_recall": docling_pos.get("early_recall", 0.0),
            "middle_recall": docling_pos.get("middle_recall", 0.0),
            "late_recall": docling_pos.get("late_recall", 0.0),
            "lost_in_middle_degradation_pct": docling_pos.get("lost_in_middle_degradation_pct", 0.0),
            "cost_breakdown": {
                "parsing": 0.0,
                "embedding_cost_usd": docling_ingest_metrics.cost_breakdown.get("embedding_cost_usd", 0.0006)
            }
        }

    direct_payload = existing_report.get("method4_direct_flash") or existing_report.get("direct_gemini", {})
    if norm_method in ("all", "method4", "compare-cache", "compare-direct") and direct_ingest_metrics:
        avg_cached_tokens = (
            sum(item.get("cached_tokens", 0) for item in direct_scores_list) / len(direct_scores_list)
            if direct_scores_list else 0.0
        )
        avg_cache_hit_rate = (
            sum(item.get("cache_hit_rate_pct", 0.0) for item in direct_scores_list) / len(direct_scores_list)
            if direct_scores_list else 0.0
        )
        direct_payload = {
            "ingestion_cost_usd": direct_ingest_metrics.total_cost_usd,
            "client_ingest_time_s": round(direct_ingest_time_s, 4),
            "embedding_ready_time_s": 0.0,
            "ingestion_time_s": round(direct_ingest_time_s, 4),
            "cost_per_query_usd": cost_calc.calculate_query_cost("direct"),
            "query_p50_ms": direct_lat_summary["end_to_end"]["p50"],
            "query_p90_ms": direct_lat_summary["end_to_end"]["p90"],
            "query_p99_ms": direct_lat_summary["end_to_end"]["p99"],
            "faithfulness": avg_score(direct_scores_list, "faithfulness"),
            "answer_relevancy": avg_score(direct_scores_list, "answer_relevancy"),
            "contextual_precision": avg_score(direct_scores_list, "contextual_precision"),
            "contextual_recall": avg_score(direct_scores_list, "contextual_recall"),
            "contextual_relevancy": avg_score(direct_scores_list, "contextual_relevancy"),
            "document_completeness": direct_pos.get("document_completeness", avg_score(direct_scores_list, "completeness")),
            "early_recall": direct_pos.get("early_recall", 0.0),
            "middle_recall": direct_pos.get("middle_recall", 0.0),
            "late_recall": direct_pos.get("late_recall", 0.0),
            "lost_in_middle_degradation_pct": direct_pos.get("lost_in_middle_degradation_pct", 0.0),
            "avg_cached_tokens": round(avg_cached_tokens, 1),
            "cache_hit_rate_pct": round(avg_cache_hit_rate, 2),
            "cost_breakdown": {
                "parsing": 0.0,
                "embedding": 0.0,
                "avg_cached_tokens": round(avg_cached_tokens, 1),
                "cache_hit_rate_pct": round(avg_cache_hit_rate, 2)
            }
        }

    direct_stateful_payload = existing_report.get("method4b_direct_stateful") or existing_report.get("direct_gemini_stateful", {})
    if norm_method in ("all", "method4b", "compare-cache", "compare-direct") and direct_stateful_ingest_metrics:
        avg_cached_tokens_b = (
            sum(item.get("cached_tokens", 0) for item in direct_stateful_scores_list) / len(direct_stateful_scores_list)
            if direct_stateful_scores_list else 0.0
        )
        avg_cache_hit_rate_b = (
            sum(item.get("cache_hit_rate_pct", 0.0) for item in direct_stateful_scores_list) / len(direct_stateful_scores_list)
            if direct_stateful_scores_list else 0.0
        )
        direct_stateful_payload = {
            "ingestion_cost_usd": direct_stateful_ingest_metrics.total_cost_usd,
            "client_ingest_time_s": round(direct_stateful_ingest_time_s, 4),
            "embedding_ready_time_s": 0.0,
            "ingestion_time_s": round(direct_stateful_ingest_time_s, 4),
            "cost_per_query_usd": cost_calc.calculate_query_cost("direct"),
            "query_p50_ms": direct_stateful_lat_summary["end_to_end"]["p50"],
            "query_p90_ms": direct_stateful_lat_summary["end_to_end"]["p90"],
            "query_p99_ms": direct_stateful_lat_summary["end_to_end"]["p99"],
            "faithfulness": avg_score(direct_stateful_scores_list, "faithfulness"),
            "answer_relevancy": avg_score(direct_stateful_scores_list, "answer_relevancy"),
            "contextual_precision": avg_score(direct_stateful_scores_list, "contextual_precision"),
            "contextual_recall": avg_score(direct_stateful_scores_list, "contextual_recall"),
            "contextual_relevancy": avg_score(direct_stateful_scores_list, "contextual_relevancy"),
            "document_completeness": direct_stateful_pos.get("document_completeness", avg_score(direct_stateful_scores_list, "completeness")),
            "early_recall": direct_stateful_pos.get("early_recall", 0.0),
            "middle_recall": direct_stateful_pos.get("middle_recall", 0.0),
            "late_recall": direct_stateful_pos.get("late_recall", 0.0),
            "lost_in_middle_degradation_pct": direct_stateful_pos.get("lost_in_middle_degradation_pct", 0.0),
            "avg_cached_tokens": round(avg_cached_tokens_b, 1),
            "cache_hit_rate_pct": round(avg_cache_hit_rate_b, 2),
            "cost_breakdown": {
                "parsing": 0.0,
                "embedding": 0.0,
                "avg_cached_tokens": round(avg_cached_tokens_b, 1),
                "cache_hit_rate_pct": round(avg_cache_hit_rate_b, 2)
            }
        }

    direct_explicit_payload = existing_report.get("method4c_direct_explicit") or existing_report.get("direct_gemini_explicit", {})
    if norm_method in ("all", "method4c", "compare-all-cache") and direct_explicit_ingest_metrics:
        avg_cached_tokens_c = (
            sum(item.get("cached_tokens", 0) for item in direct_explicit_scores_list) / len(direct_explicit_scores_list)
            if direct_explicit_scores_list else 0.0
        )
        avg_cache_hit_rate_c = (
            sum(item.get("cache_hit_rate_pct", 0.0) for item in direct_explicit_scores_list) / len(direct_explicit_scores_list)
            if direct_explicit_scores_list else 0.0
        )
        direct_explicit_payload = {
            "ingestion_cost_usd": direct_explicit_ingest_metrics.total_cost_usd,
            "client_ingest_time_s": round(direct_explicit_ingest_time_s, 4),
            "embedding_ready_time_s": 0.0,
            "ingestion_time_s": round(direct_explicit_ingest_time_s, 4),
            "cost_per_query_usd": cost_calc.calculate_query_cost("direct"),
            "query_p50_ms": direct_explicit_lat_summary["end_to_end"]["p50"],
            "query_p90_ms": direct_explicit_lat_summary["end_to_end"]["p90"],
            "query_p99_ms": direct_explicit_lat_summary["end_to_end"]["p99"],
            "faithfulness": avg_score(direct_explicit_scores_list, "faithfulness"),
            "answer_relevancy": avg_score(direct_explicit_scores_list, "answer_relevancy"),
            "contextual_precision": avg_score(direct_explicit_scores_list, "contextual_precision"),
            "contextual_recall": avg_score(direct_explicit_scores_list, "contextual_recall"),
            "contextual_relevancy": avg_score(direct_explicit_scores_list, "contextual_relevancy"),
            "document_completeness": direct_explicit_pos.get("document_completeness", avg_score(direct_explicit_scores_list, "completeness")),
            "early_recall": direct_explicit_pos.get("early_recall", 0.0),
            "middle_recall": direct_explicit_pos.get("middle_recall", 0.0),
            "late_recall": direct_explicit_pos.get("late_recall", 0.0),
            "lost_in_middle_degradation_pct": direct_explicit_pos.get("lost_in_middle_degradation_pct", 0.0),
            "avg_cached_tokens": round(avg_cached_tokens_c, 1),
            "cache_hit_rate_pct": round(avg_cache_hit_rate_c, 2),
            "cost_breakdown": {
                "parsing": 0.0,
                "embedding": 0.0,
                "avg_cached_tokens": round(avg_cached_tokens_c, 1),
                "cache_hit_rate_pct": round(avg_cache_hit_rate_c, 2)
            }
        }

    report_payload = {
        "project_id": project_id,
        "document": pdf_path,
        "num_test_cases": len(eval_items),
        # Historical keys
        "bigquery_native": bq_payload,
        "legacy_flash": flash_payload,
        "pymupdf4llm": mupdf_payload,
        "direct_gemini": direct_payload,
        "direct_gemini_stateful": direct_stateful_payload,
        "direct_gemini_explicit": direct_explicit_payload,
        # Canonical architecture keys
        "method1_bigquery": bq_payload,
        "method2_flash_vision": flash_payload,
        "method3_pymupdf_cpu": mupdf_payload,
        "method3b_docling": docling_payload,
        "method4_direct_flash": direct_payload,
        "method4b_direct_stateful": direct_stateful_payload,
        "method4c_direct_explicit": direct_explicit_payload,
    }

    # 5. Generate Markdown Report
    report_md = BenchmarkReporter.generate_markdown_report(report_payload, output_report)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Report written to: {output_report}")
    print("=" * 80)
    print("\n" + report_md[:1200] + "\n...\n")

    return report_payload


def main():
    parser = argparse.ArgumentParser(description="Run Comparative RAG Benchmark")
    parser.add_argument("--pdf", default="data/documents/sample_50page.pdf", help="Path to 50-page PDF")
    parser.add_argument("--dataset", default=None, help="Path to golden dataset (defaults automatically based on --pdf)")
    parser.add_argument("--project", default=os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id"), help="GCP project ID")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--output", default="output/benchmark_report.md", help="Output markdown report path")
    parser.add_argument("--max-items", type=int, default=None, help="Max test cases to evaluate")
    parser.add_argument(
        "--method",
        default="all",
        help=(
            "Which method to evaluate. Options: "
            "all (default), "
            "method1 (aliases: bigquery, bq, bigquery-native, 1), "
            "method2 (aliases: flash, flash-vision-md, gemini-flash-md, 2), "
            "method3 (aliases: pymupdf, pymupdf4llm, pymupdf-cpu, 3), "
            "method3b (aliases: docling, docling-cpu, docling-local, 3b), "
            "method4 (aliases: direct, direct-flash, long-context, 4)"
        )
    )
    parser.add_argument("--run-id", default=None, help="Custom run ID for artifact isolation (defaults to auto-generated)")
    parser.add_argument("--no-cleanup", action="store_true", help="Disable automatic purging of prior run caches")
    parser.add_argument("--compare", nargs="?", const="output/benchmark_report.json", default=None, help="Re-generate comparative reports from an existing JSON report without running cloud pipelines")
    args = parser.parse_args()

    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"Error: JSON report '{args.compare}' not found.")
            sys.exit(1)
        with open(compare_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        md = BenchmarkReporter.generate_markdown_report(data, args.output)
        print(f"Successfully regenerated comparative benchmark reports:")
        print(f"  Markdown: {args.output}")
        print(f"  CSV     : {Path(args.output).with_suffix('.csv')}")
        print(f"  JSON    : {Path(args.output).with_suffix('.json')}")
        return

    run_benchmark(
        pdf_path=args.pdf,
        dataset_path=args.dataset,
        project_id=args.project,
        top_k=args.top_k,
        output_report=args.output,
        max_eval_items=args.max_items,
        method=args.method,
        run_id=args.run_id,
        auto_cleanup=not args.no_cleanup
    )


if __name__ == "__main__":
    main()
