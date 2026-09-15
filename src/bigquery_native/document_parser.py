import os
from typing import Dict, Any, Optional


class BigQueryDocumentParser:
    """
    Manages in-warehouse BigQuery document parsing using ML.PROCESS_DOCUMENT
    to extract structured layout chunks, text, and tables from Object Tables.

    WHY ML.PROCESS_DOCUMENT IS USED INSTEAD OF AI.PARSE_DOCUMENT:
    Google Cloud's Zero-Copy RAG repository (GoogleCloudPlatform/document-analytics-on-bigquery)
    mentions `AI.PARSE_DOCUMENT` in its conceptual diagrams. However, `AI.PARSE_DOCUMENT` does
    NOT work in live BigQuery:
        - The BigQuery GoogleSQL compiler accepts at most 2 arguments (document, options).
        - But the backend execution runtime fails with:
          "400 Invalid number of parameters passed to function: expected 3, got 2;
           error in ai.parse_document expression"
    Due to this internal platform engine mismatch, Google's own reference implementation
    (sql/Parameterized_Clinical_Trials.sql) and production architectures use `ML.PROCESS_DOCUMENT`
    with a Document AI Layout Parser remote model (`CLOUD_AI_DOCUMENT_v1`) instead.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: str = "rag_benchmark",
        processor_id: Optional[str] = None,
        connection_id: Optional[str] = None
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        self.dataset_id = dataset_id
        self.processor_id = processor_id or os.environ.get("DOCUMENT_AI_PROCESSOR_ID", "your-docai-processor-id")
        self.connection_id = connection_id or os.environ.get("BQ_CONNECTION_ID", f"{self.project_id}.us.vertex-connection")

    def get_process_document_sql(
        self,
        source_table: str = "pdf_object_table",
        target_table: str = "document_parsed",
        model_name: str = "docai_layout_model",
        chunk_size: int = 1000,
        uri_filter: Optional[str] = None
    ) -> str:
        """
        Executes ML.PROCESS_DOCUMENT against the object table with layout chunking options.
        This is the working, production SQL implementation for in-warehouse PDF parsing.
        """
        where_clause = f"WHERE uri = '{uri_filter}'" if uri_filter else ""
        return f"""
CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.{target_table}` AS
SELECT * FROM ML.PROCESS_DOCUMENT(
  MODEL `{self.project_id}.{self.dataset_id}.{model_name}`,
  TABLE `{self.project_id}.{self.dataset_id}.{source_table}`,
  PROCESS_OPTIONS => (JSON '{{"layout_config": {{"chunking_config": {{"chunk_size": {chunk_size}, "include_ancestor_headings": true}}}}}}')
)
{where_clause};
""".strip()

    def get_create_docai_model_sql(self, model_name: str = "docai_layout_model") -> str:
        """
        Creates the BigQuery Cloud AI Document remote model required by ML.PROCESS_DOCUMENT.
        """
        return f"""
CREATE OR REPLACE MODEL `{self.project_id}.{self.dataset_id}.{model_name}`
REMOTE WITH CONNECTION `{self.connection_id}`
OPTIONS (
  REMOTE_SERVICE_TYPE = 'CLOUD_AI_DOCUMENT_v1',
  DOCUMENT_PROCESSOR = 'projects/26258447032/locations/us/processors/{self.processor_id}'
);
""".strip()

    def get_extract_chunks_sql(
        self,
        parsed_table: str = "document_parsed",
        chunks_table: str = "document_chunks",
        doc_id: Optional[str] = None
    ) -> str:
        """
        Extracts structured semantic chunks from ML.PROCESS_DOCUMENT output into the chunks table.
        """
        doc_expr = f"'{doc_id}'" if doc_id else "uri"
        return f"""
INSERT INTO `{self.project_id}.{self.dataset_id}.{chunks_table}` (chunk_id, doc_id, page_number, content)
SELECT
  CONCAT({doc_expr}, '-', COALESCE(JSON_EXTRACT_SCALAR(json, '$.chunkId'), GENERATE_UUID())) AS chunk_id,
  {doc_expr} AS doc_id,
  CAST(COALESCE(JSON_EXTRACT_SCALAR(json, '$.pageSpan.pageStart'), '1') AS INT64) AS page_number,
  JSON_EXTRACT_SCALAR(json, '$.content') AS content
FROM
  `{self.project_id}.{self.dataset_id}.{parsed_table}`,
  UNNEST(JSON_EXTRACT_ARRAY(ml_process_document_result.chunkedDocument.chunks, '$')) json
WHERE
  JSON_EXTRACT_SCALAR(json, '$.content') IS NOT NULL;
""".strip()
