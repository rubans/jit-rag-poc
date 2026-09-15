import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from src.common.models import DocumentChunk, RetrievalResult, ExecutionMetrics
from src.legacy_flash.pdf_splitter import split_pdf_to_images
from src.legacy_flash.page_converter import PageConverter
from src.legacy_flash.markdown_chunker import MarkdownChunker
from src.legacy_flash.retriever import FlashMarkdownRetriever
from src.legacy_flash.generator import FlashGenerator
from src.common.config import load_config


class FlashMarkdownRAGPipeline:
    """
    Method 2: Legacy Client-Orchestrated Gemini Flash Markdown RAG Pipeline.
    Supports unique run_id isolation and automated cache cleanup across invocations.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        work_dir: str = "output/legacy_flash",
        use_cache: bool = False,
        run_id: Optional[str] = None,
        auto_cleanup: bool = True
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        self.base_work_dir = Path(work_dir)
        self.auto_cleanup = auto_cleanup
        self.use_cache = use_cache

        # Auto-remove legacy or stale caches from previous runs without prompting
        if self.auto_cleanup:
            self._cleanup_previous_cache(preserve_run_id=run_id if use_cache else None)

        # Generate or assign unique run_id so this run's artifacts are isolated
        if not run_id:
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        else:
            self.run_id = run_id

        self.work_dir = self.base_work_dir / "runs" / self.run_id
        self.images_dir = self.work_dir / "images"
        self.markdown_dir = self.work_dir / "markdown"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        self.config = load_config()
        vision_model = getattr(self.config.models.gemini_flash, "vision_model_name", self.config.models.gemini_flash.model_name)
        gen_model = getattr(self.config.models.gemini_flash, "generator_model_name", self.config.models.gemini_flash.model_name)
        emb_model = getattr(self.config.models.embedding, "model_name", "text-embedding-004")
        self.converter = PageConverter(project_id=project_id, model_name=vision_model, use_cache=use_cache)
        self.chunker = MarkdownChunker()
        self.retriever = FlashMarkdownRetriever(project_id=project_id, model_name=emb_model, allow_fallback=False)
        self.generator = FlashGenerator(project_id=project_id, model_name=gen_model)
        
        self.chunks: List[DocumentChunk] = []
        self.metrics = ExecutionMetrics(pipeline_name="Legacy Gemini Flash Markdown")

    def _cleanup_previous_cache(self, preserve_run_id: Optional[str] = None):
        """
        Automatically cleans up legacy un-isolated folders and previous run caches
        to avoid disk bloat and guarantee no stale cache is reused across runs.
        """
        # 1. Clean legacy root images/markdown folders if present
        for folder_name in ["images", "markdown"]:
            legacy_path = self.base_work_dir / folder_name
            if legacy_path.exists() and legacy_path.is_dir():
                try:
                    shutil.rmtree(legacy_path, ignore_errors=True)
                except Exception:
                    pass

        # 2. Clean previous runs in runs/ directory
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
        Removes the current run's artifacts/cache directory.
        """
        if self.work_dir.exists():
            try:
                shutil.rmtree(self.work_dir, ignore_errors=True)
            except Exception:
                pass

    def ingest(self, pdf_path: str) -> ExecutionMetrics:
        t0 = time.perf_counter()
        
        # 1. Split PDF to page images (150 DPI JPEG for rapid I/O & network transmission)
        t_split_start = time.perf_counter()
        image_paths = split_pdf_to_images(pdf_path, str(self.images_dir), dpi=150, image_format="jpg")
        self.metrics.ingestion_latency_ms += (time.perf_counter() - t_split_start) * 1000

        # 2. Convert each page image to markdown via Gemini Flash (25 concurrent workers)
        t_conv_start = time.perf_counter()
        self.converter.pdf_path = pdf_path
        md_paths = self.converter.convert_all_pages(image_paths, str(self.markdown_dir), max_workers=25)
        self.metrics.ingestion_latency_ms += (time.perf_counter() - t_conv_start) * 1000

        # 3. Chunk markdown files
        all_chunks: List[DocumentChunk] = []
        for md_p in md_paths:
            chunks = self.chunker.chunk_markdown_file(md_p)
            all_chunks.extend(chunks)
        self.chunks = all_chunks

        # 4. Embed chunks
        t_emb_start = time.perf_counter()
        self.retriever.index_chunks(self.chunks)
        self.metrics.embedding_latency_ms = (time.perf_counter() - t_emb_start) * 1000
        
        self.metrics.total_latency_ms = (time.perf_counter() - t0) * 1000
        
        # Calculate cost for 50-page processing
        num_pages = len(image_paths)
        flash_cfg = self.config.models.gemini_flash
        emb_cfg = self.config.models.embedding
        
        # Ingestion cost: Vision tokens per page + estimated output tokens (~600 tokens/page)
        vision_tokens = num_pages * flash_cfg.image_tokens_per_page
        output_tokens = num_pages * 600
        conv_cost = (vision_tokens / 1_000_000 * flash_cfg.input_price_per_1m_tokens) + \
                    (output_tokens / 1_000_000 * flash_cfg.output_price_per_1m_tokens)
        
        # Embedding cost: ~2,500 characters per page
        total_chars = num_pages * 2500
        emb_cost = (total_chars / 1_000_000) * emb_cfg.price_per_1m_characters
        
        self.metrics.cost_breakdown["page_conversion_cost_usd"] = conv_cost
        self.metrics.cost_breakdown["embedding_cost_usd"] = emb_cost
        self.metrics.total_cost_usd = conv_cost + emb_cost
        
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

        # Query cost
        flash_cfg = self.config.models.gemini_flash
        query_input_cost = (meta.get("input_tokens", 0) / 1_000_000) * flash_cfg.input_price_per_1m_tokens
        query_output_cost = (meta.get("output_tokens", 0) / 1_000_000) * flash_cfg.output_price_per_1m_tokens
        query_cost = query_input_cost + query_output_cost

        q_metrics = ExecutionMetrics(
            pipeline_name="Legacy Gemini Flash Markdown",
            retrieval_latency_ms=ret_ms,
            generation_latency_ms=gen_ms,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            total_cost_usd=query_cost,
            cost_breakdown={"generation_cost_usd": query_cost}
        )

        return answer, results, q_metrics

