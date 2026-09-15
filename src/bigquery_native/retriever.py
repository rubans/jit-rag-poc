import os
from typing import List, Dict, Any, Optional
from src.common.models import DocumentChunk, RetrievalResult
from src.bigquery_native.embedder import BigQueryEmbedder


class BigQueryVectorRetriever:
    """
    Manages BigQuery data-centric vector indexing and retrieval via AI.SEARCH
    and native VECTOR_SEARCH integration with autonomous embedding columns,
    guaranteeing 100% embedding readiness before executing retrieval.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: str = "rag_benchmark",
        source_table: str = "document_chunks",
        embedding_column: str = "content_embedding.result",
        embedder: Optional[BigQueryEmbedder] = None
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        self.dataset_id = dataset_id
        self.source_table = source_table
        self.embedding_column = embedding_column
        self.embedder = embedder or BigQueryEmbedder(project_id=self.project_id, dataset_id=dataset_id)
        self.client = None
        self._verified_docs: set = set()
        self._init_client()

    def _init_client(self):
        try:
            from google.cloud import bigquery
            self.client = bigquery.Client(project=self.project_id)
        except Exception:
            self.client = None

    def get_create_vector_index_sql(self, index_name: str = "doc_vector_index") -> str:
        """
        Creates an IVF vector index directly on the autonomous embedding column.
        """
        return f"""
CREATE VECTOR INDEX IF NOT EXISTS `{index_name}`
ON `{self.project_id}.{self.dataset_id}.{self.source_table}`({self.embedding_column})
OPTIONS (
  index_type = 'IVF',
  distance_type = 'COSINE'
);
""".strip()

    def get_ai_search_sql(self, query_text: str, top_k: int = 5, doc_id: Optional[str] = None) -> str:
        """
        Generates BigQuery AI.SEARCH SQL statement. AI.SEARCH automatically leverages
        the embedding model configured on the table's autonomous embedding column.
        """
        escaped_query = query_text.replace("'", "\\'")
        where_clause = f"WHERE base.doc_id = '{doc_id}'" if doc_id else ""
        search_k = max(top_k * 10, 50) if doc_id else top_k
        return f"""
SELECT
  base.chunk_id,
  base.doc_id,
  base.page_number,
  base.content,
  distance AS cosine_distance,
  (1.0 - distance) AS similarity_score
FROM
  AI.SEARCH(
    TABLE `{self.project_id}.{self.dataset_id}.{self.source_table}`,
    'content',
    '{escaped_query}',
    top_k => {search_k}
  )
{where_clause}
ORDER BY distance ASC
LIMIT {top_k};
""".strip()

    def get_vector_search_sql(self, query_text: str, top_k: int = 5, doc_id: Optional[str] = None) -> str:
        """
        Compatibility method returning AI.SEARCH SQL as the modern retrieval technique.
        """
        return self.get_ai_search_sql(query_text, top_k=top_k, doc_id=doc_id)

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        ensure_ready: bool = True
    ) -> List[RetrievalResult]:
        """
        Executes semantic retrieval via AI.SEARCH. If ensure_ready is True, checks that
        autonomous embeddings are 100% complete before firing the search query.
        """
        if ensure_ready and self.embedder and (doc_id not in self._verified_docs):
            # Block until 100% of chunks in the table have generated embeddings
            is_ready = self.embedder.wait_for_completion(table_name=self.source_table, doc_id=doc_id, timeout_sec=60)
            if not is_ready:
                print(f"[ERROR] Autonomous embeddings not 100% complete before retrieval on {self.source_table} for doc_id '{doc_id}'. Aborting retrieval.")
                return []
            if doc_id:
                self._verified_docs.add(doc_id)

        sql = self.get_ai_search_sql(query_text, top_k, doc_id=doc_id)
        if not self.client:
            return []

        try:
            job = self.client.query(sql)
            rows = list(job.result())
            results: List[RetrievalResult] = []
            for rank_idx, row in enumerate(rows, 1):
                chunk = DocumentChunk(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    page_number=int(row["page_number"]) if row["page_number"] else None,
                    content=str(row["content"])
                )
                score = float(row.get("similarity_score", 1.0 - float(row.get("cosine_distance", 0.0))))
                results.append(RetrievalResult(chunk=chunk, score=score, rank=rank_idx))
            return results
        except Exception as e:
            print(f"[ERROR] BigQuery retrieval failed: {e}")
            return []
