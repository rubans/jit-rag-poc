import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.direct_gemini.pipeline import DirectGeminiFlashRAGPipeline
from src.eval.deepeval_runner import DeepEvalBenchmarkRunner
from src.eval.cost_calculator import CostCalculator
from src.eval.reporter import BenchmarkReporter


def test_direct_gemini_pipeline_ingest_and_query(tmp_path):
    # Create dummy 1-page sample file
    dummy_pdf = tmp_path / "sample.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy pdf binary stream for unit testing")

    pipeline = DirectGeminiFlashRAGPipeline(
        project_id="test-project",
        model_name="gemini-3.8-flash",
        work_dir=str(tmp_path / "direct_work")
    )

    # Ingestion test
    metrics = pipeline.ingest(str(dummy_pdf))
    assert metrics.ingestion_latency_ms >= 0.0
    assert metrics.embedding_latency_ms == 0.0
    assert metrics.total_cost_usd == 0.0

    # Query test (mocking client generation)
    mock_resp = MagicMock()
    mock_resp.text = "In FY 2024, consolidated revenue reached $84.20 billion."
    mock_resp.usage_metadata.prompt_token_count = 25000
    mock_resp.usage_metadata.candidates_token_count = 45

    pipeline.client = MagicMock()
    pipeline.client.models.generate_content.return_value = mock_resp
    pipeline.pdf_part = MagicMock()

    answer, results, q_metrics = pipeline.query("What was the FY 2024 revenue?")
    assert "$84.20 billion" in answer
    assert len(results) == 1
    assert results[0].score == 1.0
    assert q_metrics.cost_breakdown["input_tokens"] == 25000
    assert q_metrics.cost_breakdown["cached_tokens"] == 0
    assert q_metrics.cost_breakdown["cache_hit_rate_pct"] == 0.0
    assert q_metrics.total_cost_usd > 0.0


def test_direct_gemini_implicit_cache_measurement(tmp_path):
    pipeline = DirectGeminiFlashRAGPipeline(
        project_id="test-project",
        model_name="gemini-3.8-flash",
        work_dir=str(tmp_path / "direct_work")
    )

    mock_resp = MagicMock()
    mock_resp.text = "Q3 Operating Income was $12.4 billion."
    mock_resp.usage_metadata.prompt_token_count = 30000
    mock_resp.usage_metadata.candidates_token_count = 50
    mock_resp.usage_metadata.cached_content_token_count = 24000  # 80% cache hit

    pipeline.client = MagicMock()
    pipeline.client.models.generate_content.return_value = mock_resp
    pipeline.pdf_part = MagicMock()

    answer, results, q_metrics = pipeline.query("What was operating income?")
    assert "$12.4 billion" in answer
    assert q_metrics.cost_breakdown["input_tokens"] == 30000
    assert q_metrics.cost_breakdown["cached_tokens"] == 24000
    assert q_metrics.cost_breakdown["uncached_tokens"] == 6000
    assert q_metrics.cost_breakdown["cache_hit_rate_pct"] == 80.0
    
    # Verify cached tokens receive the 75% discount (cost is lower than full uncached)
    full_uncached_cost = (30000 / 1_000_000.0) * 0.075 + (50 / 1_000_000.0) * 0.30
    assert q_metrics.total_cost_usd < full_uncached_cost


def test_position_stratified_metrics_and_completeness():
    runner = DeepEvalBenchmarkRunner()
    
    # Evaluate completeness
    scores_full = runner.evaluate_test_case(
        input_text="What was revenue?",
        actual_output="Consolidated revenue was $84.20 billion, representing an 18.5% year-over-year increase.",
        expected_output="Consolidated revenue was $84.20 billion, representing an 18.5% year-over-year increase.",
        retrieved_contexts=["Context"]
    )
    assert "completeness" in scores_full
    assert scores_full["completeness"] == 1.0

    scores_partial = runner.evaluate_test_case(
        input_text="What was revenue?",
        actual_output="Consolidated revenue for FY 2024 was $84.20B with 18.5% growth.",
        expected_output="Consolidated revenue was $84.20 billion, representing an 18.5% year-over-year increase.",
        retrieved_contexts=["Context"]
    )
    assert scores_partial["completeness"] >= 0.40


    # Mock 25 test cases with early (p1-15), middle (p16-35), late (p36-50) distribution
    test_cases = [
        # Early: strong recall (0.95)
        {"page": 3, "scores": {"contextual_recall": 0.95, "completeness": 0.95}},
        {"page": 10, "scores": {"contextual_recall": 0.95, "completeness": 0.95}},
        # Middle: dipped recall (0.80) -> lost in middle
        {"page": 20, "scores": {"contextual_recall": 0.80, "completeness": 0.80}},
        {"page": 28, "scores": {"contextual_recall": 0.80, "completeness": 0.80}},
        # Late: strong recall (0.95)
        {"page": 40, "scores": {"contextual_recall": 0.95, "completeness": 0.95}},
        {"page": 50, "scores": {"contextual_recall": 0.95, "completeness": 0.95}},
    ]

    strat_metrics = runner.compute_position_stratified_metrics(test_cases)
    assert strat_metrics["early_recall"] == 0.95
    assert strat_metrics["middle_recall"] == 0.80
    assert strat_metrics["late_recall"] == 0.95
    assert strat_metrics["lost_in_middle_degradation_pct"] > 10.0  # (0.95 - 0.80)/0.95 = 15.79%
    assert strat_metrics["document_completeness"] > 0.80


def test_cost_calculator_direct_gemini():
    calc = CostCalculator()
    ingest_cost = calc.calculate_direct_gemini_ingestion_cost(num_pages=50)
    assert ingest_cost["total_ingestion_cost_usd"] == 0.0

    query_cost = calc.calculate_query_cost("direct", input_tokens=26000, output_tokens=100)
    assert query_cost > 0.0010  # ~0.001980


def test_4way_reporter_generation(tmp_path):
    report_file = tmp_path / "report_4way.md"
    sample_data = {
        "project_id": "vertex-ai-poc",
        "bigquery_native": {"ingestion_cost_usd": 1.50, "query_p50_ms": 1700, "cost_per_query_usd": 0.000060},
        "legacy_flash": {"ingestion_cost_usd": 0.008, "query_p50_ms": 3800, "cost_per_query_usd": 0.000079},
        "pymupdf4llm": {"ingestion_cost_usd": 0.0031, "query_p50_ms": 3700, "cost_per_query_usd": 0.000079},
        "direct_gemini": {
            "ingestion_cost_usd": 0.0,
            "query_p50_ms": 2200,
            "cost_per_query_usd": 0.001980,
            "early_recall": 0.96,
            "middle_recall": 0.84,
            "late_recall": 0.95,
            "lost_in_middle_degradation_pct": 12.04
        }
    }
    md = BenchmarkReporter.generate_markdown_report(sample_data, str(report_file))
    assert report_file.exists()
    assert "Method 4: Direct Flash" in md
    assert "Lost in the Middle" in md
    assert "Break-Even Point" in md


def test_direct_gemini_explicit_cache(tmp_path):
    dummy_pdf = tmp_path / "sample_bp.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 large 392-page enterprise pdf binary stream")

    pipeline = DirectGeminiFlashRAGPipeline(
        project_id="test-project",
        model_name="gemini-3.8-flash",
        work_dir=str(tmp_path / "direct_work"),
        explicit_cache=True,
        cache_ttl="600s"
    )

    mock_client = MagicMock()
    mock_cache = MagicMock()
    mock_cache.name = "cachedContents/test_cache_12345"
    mock_client.caches.create.return_value = mock_cache

    mock_resp = MagicMock()
    mock_resp.text = "BP reported upstream production of 1,450 mboe/d."
    mock_resp.usage_metadata.prompt_token_count = 210000
    mock_resp.usage_metadata.candidates_token_count = 60
    mock_resp.usage_metadata.cached_content_token_count = 209950  # 99.98% cache hit
    mock_client.models.generate_content.return_value = mock_resp

    pipeline.client = mock_client
    pipeline.pdf_part = MagicMock()

    # Test explicit cache creation during ingest
    metrics = pipeline.ingest(str(dummy_pdf))
    assert pipeline.cached_content is not None
    assert pipeline.cached_content.name == "cachedContents/test_cache_12345"
    assert metrics.cost_breakdown["explicit_cache_name"] == "cachedContents/test_cache_12345"
    mock_client.caches.create.assert_called_once()

    # Test query using explicit cache
    answer, results, q_metrics = pipeline.query("What was upstream production?")
    assert "1,450 mboe/d" in answer
    assert q_metrics.cost_breakdown["cached_tokens"] == 209950
    assert q_metrics.cost_breakdown["cache_hit_rate_pct"] > 99.0

    # Test explicit cache cleanup
    pipeline.cleanup()
    mock_client.caches.delete.assert_called_once_with(name="cachedContents/test_cache_12345")
    assert pipeline.cached_content is None

