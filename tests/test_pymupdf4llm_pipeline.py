from pathlib import Path
from src.pymupdf4llm_pipeline.pipeline import PyMuPDF4LLMRAGPipeline


def test_pymupdf4llm_pipeline_init(tmp_path):
    p = PyMuPDF4LLMRAGPipeline(work_dir=str(tmp_path), run_id="test_run_123")
    assert p.run_id == "test_run_123"
    assert p.work_dir == tmp_path / "runs" / "test_run_123"
    assert p.markdown_dir.exists()


def test_pymupdf4llm_ingest_mocked(tmp_path, monkeypatch):
    p = PyMuPDF4LLMRAGPipeline(work_dir=str(tmp_path), run_id="test_run_mock")

    # Mock pymupdf4llm.to_markdown to return 2 sample pages quickly
    sample_pages = [
        {"text": "# Page 1\nExecutive summary text.", "metadata": {"page": 1}},
        {"text": "# Page 2\n| Metric | Value |\n| Rev | $84.2B |", "metadata": {"page": 2}}
    ]
    monkeypatch.setattr("pymupdf4llm.to_markdown", lambda path, page_chunks: sample_pages)

    metrics = p.ingest("data/documents/sample_50page.pdf")
    assert metrics.ingestion_latency_ms >= 0
    assert len(p.chunks) >= 2
    assert (p.markdown_dir / "page_001.md").exists()
    assert (p.markdown_dir / "page_002.md").exists()
    assert metrics.cost_breakdown["parsing_cost_usd"] == 0.0
