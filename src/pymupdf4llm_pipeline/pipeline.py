import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pymupdf4llm

from src.common.models import DocumentChunk, RetrievalResult, ExecutionMetrics
from src.legacy_flash.markdown_chunker import MarkdownChunker
from src.legacy_flash.retriever import FlashMarkdownRetriever
from src.legacy_flash.generator import FlashGenerator
from src.common.config import load_config


class PyMuPDF4LLMRAGPipeline:
    """
    Method 3: PyMuPDF4LLM Local CPU Just-In-Time RAG Pipeline.
    Extracts structured Markdown directly from PDF layout streams on local CPU in ~1s
    with $0.00 cloud vision cost, then reuses structure-aware chunking and dense vector search.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        work_dir: str = "output/pymupdf4llm",
        run_id: Optional[str] = None,
        auto_cleanup: bool = True
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        self.base_work_dir = Path(work_dir)
        self.auto_cleanup = auto_cleanup

        if self.auto_cleanup:
            self._cleanup_previous_cache(preserve_run_id=run_id)

        if not run_id:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        else:
            self.run_id = run_id

        self.work_dir = self.base_work_dir / "runs" / self.run_id
        self.markdown_dir = self.work_dir / "markdown"
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.config = load_config()
        gen_model = getattr(self.config.models.gemini_flash, "generator_model_name", "gemini-3.8-flash")
        emb_model = getattr(self.config.models.embedding, "model_name", "text-embedding-004")

        self.chunker = MarkdownChunker()
        self.retriever = FlashMarkdownRetriever(project_id=project_id, model_name=emb_model)
        self.generator = FlashGenerator(project_id=project_id, model_name=gen_model)

        self.chunks: List[DocumentChunk] = []
        self.metrics = ExecutionMetrics(pipeline_name="PyMuPDF4LLM Local CPU Markdown (JIT-RAG)")

    def _cleanup_previous_cache(self, preserve_run_id: Optional[str] = None):
        """
        Automatically cleans up previous runs to avoid disk bloat.
        """
        runs_dir = self.base_work_dir / "runs"
        if runs_dir.exists() and runs_dir.is_dir():
            for item in runs_dir.iterdir():
                if item.is_dir():
                    if preserve_run_id and item.name == preserve_run_id:
                        continue
                    try:
                        shutil.rmtree(item, ignore_errors=True)
                    except Exception:
                        pass

    def cleanup_run(self):
        """
        Removes the current run's artifacts directory.
        """
        if self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir, ignore_errors=True)
            except Exception:
                pass

    def ingest(self, pdf_path: str) -> ExecutionMetrics:
        t0 = time.perf_counter()

        # 1. Local CPU Markdown Extraction via PyMuPDF4LLM (Zero API / Cloud cost)
        t_parse_start = time.perf_counter()
        pages_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
        parse_ms = (time.perf_counter() - t_parse_start) * 1000
        self.metrics.ingestion_latency_ms += parse_ms

        # 2. Write individual markdown pages and chunk
        all_chunks: List[DocumentChunk] = []
        for idx, page_item in enumerate(pages_data, start=1):
            text = page_item.get("text", "")
            md_file = self.markdown_dir / f"page_{idx:03d}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            # Chunk page content using shared structure-aware chunker
            chunks = self.chunker.chunk_text(text, doc_id="pymupdf4llm_doc", page_number=idx)
            all_chunks.extend(chunks)

        self.chunks = all_chunks

        # 3. Embed chunks into vector retriever
        t_emb_start = time.perf_counter()
        self.retriever.index_chunks(self.chunks)
        emb_ms = (time.perf_counter() - t_emb_start) * 1000
        self.metrics.embedding_latency_ms = emb_ms

        self.metrics.total_latency_ms = (time.perf_counter() - t0) * 1000

        # Cost Calculation: Parsing is $0.00 (local CPU). Only embedding characters apply.
        emb_cfg = self.config.models.embedding
        total_chars = sum(len(c.content) for c in self.chunks)
        emb_cost = (total_chars / 1_000_000) * emb_cfg.price_per_1m_characters

        self.metrics.cost_breakdown["parsing_cost_usd"] = 0.0000
        self.metrics.cost_breakdown["embedding_cost_usd"] = emb_cost
        self.metrics.total_cost_usd = emb_cost

        return self.metrics

    def query(self, user_query: str, top_k: int = 5) -> Tuple[str, List[RetrievalResult], ExecutionMetrics]:
        t0 = time.perf_counter()

        # 1. Retrieve top-k
        t_ret_start = time.perf_counter()
        results = self.retriever.retrieve(user_query, top_k=top_k)
        ret_ms = (time.perf_counter() - t_ret_start) * 1000

        # 2. Generate answer
        t_gen_start = time.perf_counter()
        answer, meta = self.generator.generate_answer(user_query, results)
        gen_ms = (time.perf_counter() - t_gen_start) * 1000

        flash_cfg = self.config.models.gemini_flash
        query_input_cost = (meta.get("input_tokens", 0) / 1_000_000) * flash_cfg.input_price_per_1m_tokens
        query_output_cost = (meta.get("output_tokens", 0) / 1_000_000) * flash_cfg.output_price_per_1m_tokens
        query_cost = query_input_cost + query_output_cost

        q_metrics = ExecutionMetrics(
            pipeline_name="PyMuPDF4LLM Local CPU Markdown (JIT-RAG)",
            retrieval_latency_ms=ret_ms,
            generation_latency_ms=gen_ms,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            total_cost_usd=query_cost,
            cost_breakdown={"generation_cost_usd": query_cost}
        )

        return answer, results, q_metrics
