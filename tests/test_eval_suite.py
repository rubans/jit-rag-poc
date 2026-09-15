from src.eval.profiler import LatencyProfiler
from src.eval.cost_calculator import CostCalculator
from src.eval.deepeval_runner import DeepEvalBenchmarkRunner
from src.eval.reporter import BenchmarkReporter
from pathlib import Path


def test_latency_profiler():
    profiler = LatencyProfiler()
    for t in [100.0, 200.0, 300.0, 400.0, 500.0]:
        profiler.record("retrieval", t)
    summary = profiler.get_summary()
    assert "retrieval" in summary
    assert summary["retrieval"]["p50"] == 300.0
    assert summary["retrieval"]["count"] == 5


def test_cost_calculator():
    calc = CostCalculator()
    flash_costs = calc.calculate_legacy_flash_ingestion_cost(num_pages=50)
    assert flash_costs["total_ingestion_cost_usd"] < 0.05
    assert flash_costs["cost_per_page_usd"] > 0

    bq_costs = calc.calculate_bigquery_native_ingestion_cost(num_pages=50)
    assert bq_costs["total_ingestion_cost_usd"] >= 1.50 # 50 * 0.030
    assert bq_costs["cost_per_page_usd"] > flash_costs["cost_per_page_usd"]


def test_deepeval_runner_scoring():
    runner = DeepEvalBenchmarkRunner()
    scores = runner.evaluate_test_case(
        input_text="What was the FY 2024 revenue?",
        actual_output="The consolidated revenue was $84.20 billion.",
        expected_output="Consolidated revenue was $84.20 billion in FY 2024.",
        retrieved_contexts=["Consolidated Revenue: FY 2024: $84.20 B (+18.5% YoY)"]
    )
    assert "faithfulness" in scores
    assert "answer_relevancy" in scores
    assert "contextual_precision" in scores
    assert "contextual_recall" in scores
    assert "contextual_relevancy" in scores
    assert scores["faithfulness"] >= 0.70
    assert scores["contextual_recall"] >= 0.70


def test_benchmark_reporter(tmp_path):
    report_file = tmp_path / "report.md"
    sample_data = {
        "project_id": "vertex-ai-poc",
        "bigquery_native": {
            "ingestion_cost_usd": 1.5034,
            "ingestion_time_s": 45.2,
            "query_p50_ms": 1420,
            "query_p90_ms": 2100,
            "faithfulness": 0.88,
            "answer_relevancy": 0.91,
            "contextual_precision": 0.86,
            "contextual_recall": 0.85,
            "contextual_relevancy": 0.87
        },
        "legacy_flash": {
            "ingestion_cost_usd": 0.0131,
            "ingestion_time_s": 27.8,
            "query_p50_ms": 380,
            "query_p90_ms": 610,
            "faithfulness": 0.94,
            "answer_relevancy": 0.93,
            "contextual_precision": 0.90,
            "contextual_recall": 0.92,
            "contextual_relevancy": 0.89
        }
    }
    md = BenchmarkReporter.generate_markdown_report(sample_data, str(report_file))
    assert report_file.exists()
    assert report_file.with_suffix(".json").exists()
    assert "Benchmark Report" in md
    assert "vertex-ai-poc" in md

