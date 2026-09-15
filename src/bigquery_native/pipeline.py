import os
import time
from typing import List, Tuple, Optional
from src.common.models import DocumentChunk, RetrievalResult, ExecutionMetrics
from src.bigquery_native.object_table import BigQueryObjectTableManager
from src.bigquery_native.document_parser import BigQueryDocumentParser
from src.bigquery_native.embedder import BigQueryEmbedder
from src.bigquery_native.retriever import BigQueryVectorRetriever
from src.bigquery_native.generator import BigQuerySQLGenerator
from src.common.config import load_config


class BigQueryNativeRAGPipeline:
    """
    Method 1: Modern Cloud-Native BigQuery RAG Pipeline utilizing
    Autonomous Embedding Generation (AI.EMBED) and AI.SEARCH.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: str = "rag_benchmark",
        bucket_name: Optional[str] = None,
        connection_id: Optional[str] = None
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        self.dataset_id = dataset_id
        self.bucket_name = bucket_name or os.environ.get("GCS_CORPUS_BUCKET", "your-gcs-bucket-name")
        self.connection_id = connection_id or os.environ.get("BQ_CONNECTION_ID", f"{self.project_id}.us.vertex-connection")
        self.config = load_config()

        self.object_table_mgr = BigQueryObjectTableManager(project_id, dataset_id, bucket_name=bucket_name)
        self.doc_parser = BigQueryDocumentParser(project_id, dataset_id)
        self.embedder = BigQueryEmbedder(project_id, dataset_id, connection_id=connection_id)
        self.retriever = BigQueryVectorRetriever(project_id, dataset_id, embedder=self.embedder)
        self.generator = BigQuerySQLGenerator(project_id, dataset_id)

        self.metrics = ExecutionMetrics(pipeline_name="Modern Cloud-Native BigQuery RAG (Autonomous Embeddings)")

    def ingest(self, pdf_path: str, doc_id: Optional[str] = None, wait_for_embeddings: bool = True) -> ExecutionMetrics:
        t0 = time.perf_counter()
        
        # 1. Determine unique doc_id for this document
        import os, re
        base_name = os.path.basename(pdf_path)
        if not doc_id:
            doc_id = f"{base_name}_{int(time.time())}"
        self.current_doc_id = doc_id
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', doc_id)

        # 2. Upload raw PDF to Cloud Storage (Zero-Copy Object Storage)
        t_upload_start = time.perf_counter()
        gcs_uri = self.object_table_mgr.upload_pdf(pdf_path)
        
        # Ensure BigQuery Object Table exists and metadata cache is refreshed
        try:
            self.object_table_mgr.setup_object_table()
        except Exception as e:
            print(f"[INFO] Object table check/refresh: {e}")

        # 3. Cloud-Native In-Warehouse Parsing via ML.PROCESS_DOCUMENT (Document AI Layout Parser)
        t_parse_start = time.perf_counter()
        temp_parsed_table = f"parsed_{safe_id}"
        parse_sql = self.doc_parser.get_process_document_sql(
            source_table="pdf_object_table",
            target_table=temp_parsed_table,
            model_name="docai_layout_model",
            chunk_size=1000,
            uri_filter=gcs_uri
        )
        print(f"[Method 1 BigQuery] Executing in-warehouse Document AI Layout Parser via ML.PROCESS_DOCUMENT on {gcs_uri}...")
        self.embedder.client.query(parse_sql).result()
        t_parse_end = time.perf_counter()
        print(f"[Method 1 BigQuery] Document AI parsing completed in {t_parse_end - t_parse_start:.2f}s")

        # 4. Extract Structured Semantic Chunks into document_chunks table
        extract_sql = self.doc_parser.get_extract_chunks_sql(
            parsed_table=temp_parsed_table,
            chunks_table="document_chunks",
            doc_id=doc_id
        )
        print(f"[Method 1 BigQuery] Extracting semantic chunks into {self.dataset_id}.document_chunks...")
        self.embedder.client.query(extract_sql).result()
        self.metrics.ingestion_latency_ms = (time.perf_counter() - t_upload_start) * 1000

        # Query chunk count and total characters for accurate cost and metrics
        count_sql = f"""
        SELECT COUNT(*) as chunk_cnt, COALESCE(SUM(LENGTH(content)), 0) as total_chars, COALESCE(MAX(page_number), 1) as max_page
        FROM `{self.project_id}.{self.dataset_id}.document_chunks`
        WHERE doc_id = '{doc_id}'
        """
        stat_rows = list(self.embedder.client.query(count_sql).result())
        chunk_count = stat_rows[0].chunk_cnt if stat_rows else 0
        total_chars = stat_rows[0].total_chars if stat_rows else 0
        num_pages = stat_rows[0].max_page if stat_rows else 50
        print(f"[Method 1 BigQuery] Extracted {chunk_count} chunks ({total_chars} chars, ~{num_pages} pages) for doc_id='{doc_id}'")

        # 5. Autonomous Embedding Generation via BigQuery AI.EMBED (asynchronous background queue)
        t_emb_start = time.perf_counter()
        if wait_for_embeddings:
            is_ready = self.embedder.wait_for_completion(
                table_name="document_chunks",
                doc_id=doc_id,
                timeout_sec=120,
                poll_interval_sec=2
            )
            if not is_ready:
                print(f"[WARNING] Autonomous embeddings not 100% complete for doc_id '{doc_id}' within timeout.")
        self.metrics.embedding_latency_ms = (time.perf_counter() - t_emb_start) * 1000

        self.metrics.total_latency_ms = (time.perf_counter() - t0) * 1000

        # 6. Accurate Cost Calculation: Document AI Layout Parser + embeddings + BQ compute
        docai_cfg = self.config.document_ai
        bq_cfg = self.config.bigquery
        emb_cfg = self.config.models.embedding

        # Layout Parser cost ($30.00 / 1,000 pages)
        docai_cost = (num_pages / 1000.0) * docai_cfg.layout_parser_price_per_1k_pages
        # Autonomous embeddings ($0.025 / 1M characters)
        emb_cost = (total_chars / 1_000_000.0) * emb_cfg.price_per_1m_characters
        # BQ storage and query scanning
        bq_compute_cost = ((num_pages * 0.001) / 1024.0) * bq_cfg.query_price_per_tb

        self.metrics.cost_breakdown["docai_processing_cost_usd"] = docai_cost
        self.metrics.cost_breakdown["embedding_cost_usd"] = emb_cost
        self.metrics.cost_breakdown["bq_compute_cost_usd"] = bq_compute_cost
        self.metrics.total_cost_usd = docai_cost + emb_cost + bq_compute_cost

        return self.metrics

    def query(self, user_query: str, top_k: int = 5, doc_id: Optional[str] = None) -> Tuple[str, List[RetrievalResult], ExecutionMetrics]:
        t0 = time.perf_counter()
        target_doc_id = doc_id or getattr(self, "current_doc_id", None)

        # 1. BigQuery AI.SEARCH (Direct semantic retrieval)
        t_ret_start = time.perf_counter()
        results = self.retriever.retrieve(user_query, top_k=top_k, doc_id=target_doc_id)
        ret_ms = (time.perf_counter() - t_ret_start) * 1000

        # 2. In-Database SQL Generation via ML.GENERATE_TEXT over AI.SEARCH context
        t_gen_start = time.perf_counter()
        answer, stats = self.generator.generate(user_query, results)
        gen_ms = (time.perf_counter() - t_gen_start) * 1000

        # Query cost: BigQuery bytes billed (~10 MB scanned)
        bytes_billed = stats.get("bytes_billed", 10_000_000)
        tb_billed = bytes_billed / (1024.0 ** 4)
        query_cost = tb_billed * self.config.bigquery.query_price_per_tb

        q_metrics = ExecutionMetrics(
            pipeline_name="Modern Cloud-Native BigQuery RAG (Autonomous Embeddings)",
            retrieval_latency_ms=ret_ms,
            generation_latency_ms=gen_ms,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            total_cost_usd=query_cost,
            cost_breakdown={"bq_query_cost_usd": query_cost}
        )

        return answer, results, q_metrics
