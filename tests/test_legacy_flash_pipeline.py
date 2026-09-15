from pathlib import Path
from src.legacy_flash.pdf_splitter import split_pdf_to_images
from src.legacy_flash.markdown_chunker import MarkdownChunker
from src.legacy_flash.retriever import FlashMarkdownRetriever
from src.legacy_flash.pipeline import FlashMarkdownRAGPipeline


def test_pdf_splitter_on_sample(tmp_path):
    pdf_path = "data/documents/sample_50page.pdf"
    assert Path(pdf_path).exists()
    
    # Test splitting first few pages or full doc
    out_dir = tmp_path / "images"
    images = split_pdf_to_images(pdf_path, str(out_dir), dpi=100) # Fast DPI for test
    assert len(images) == 50
    assert Path(images[0]).exists()


def test_markdown_chunker():
    chunker = MarkdownChunker(target_chunk_size=200)
    sample_md = (
        "# Executive Summary\n\n"
        "This is paragraph one of the executive report.\n\n"
        "## Financials\n\n"
        "| Metric | Value |\n"
        "| Revenue | $84.2B |\n\n"
        "### Notes\n\n"
        "Audited results."
    )
    chunks = chunker.chunk_text(sample_md, doc_id="test_doc", page_number=1)
    assert len(chunks) >= 2
    assert any("Executive Summary" in c.content for c in chunks)
    assert any("Financials" in c.content for c in chunks)


def test_retriever_similarity():
    retriever = FlashMarkdownRetriever()
    chunker = MarkdownChunker()
    chunks = chunker.chunk_text(
        "# Strategic Highlights\nRevenue reached $84.2 billion.\n\n# ESG Review\nCarbon emissions down 15%.",
        doc_id="doc1",
        page_number=1
    )
    retriever.index_chunks(chunks)
    results = retriever.retrieve("What was the annual revenue?", top_k=1)
    assert len(results) == 1
    assert "Revenue" in results[0].chunk.content


def test_run_id_isolation_and_auto_cleanup(tmp_path):
    # 1. Simulate legacy un-isolated folders
    legacy_images = tmp_path / "images"
    legacy_images.mkdir()
    (legacy_images / "stale.png").write_text("dummy")
    
    # 2. Simulate an older run
    old_run_dir = tmp_path / "runs" / "run_old_123" / "markdown"
    old_run_dir.mkdir(parents=True)
    (old_run_dir / "stale.md").write_text("# Old Run")
    
    # 3. Instantiate pipeline with auto_cleanup=True (default)
    p = FlashMarkdownRAGPipeline(work_dir=str(tmp_path), run_id="run_new_456")
    
    # Verify legacy folders and old runs were cleaned automatically without prompting
    assert not legacy_images.exists()
    assert not (tmp_path / "runs" / "run_old_123").exists()
    
    # Verify new run is isolated under its unique run_id
    assert p.run_id == "run_new_456"
    assert p.work_dir == tmp_path / "runs" / "run_new_456"
    assert p.images_dir.exists()
    assert p.markdown_dir.exists()


def test_full_pipeline_query(tmp_path, monkeypatch):
    pipeline = FlashMarkdownRAGPipeline(work_dir=str(tmp_path), run_id="test_run")
    
    # Mock splitting & vision conversion so unit test runs in milliseconds without live API calls
    fake_md = pipeline.markdown_dir / "page_001.md"
    fake_md.write_text("# Executive Summary\nAnnual revenue was $84.2 billion.", encoding="utf-8")
    
    monkeypatch.setattr("src.legacy_flash.pipeline.split_pdf_to_images", lambda pdf, out, dpi, **kw: ["fake_page.png"])
    monkeypatch.setattr(pipeline.converter, "convert_all_pages", lambda imgs, out, **kw: [str(fake_md)])
    
    metrics = pipeline.ingest("data/documents/sample_50page.pdf")
    assert metrics.ingestion_latency_ms >= 0
    assert len(pipeline.chunks) > 0
    
    answer, results, q_metrics = pipeline.query("What was the revenue?")
    assert len(results) > 0
    assert answer is not None
    assert q_metrics.retrieval_latency_ms >= 0

