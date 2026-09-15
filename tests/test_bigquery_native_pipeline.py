import os
import pytest
from src.bigquery_native.object_table import BigQueryObjectTableManager
from src.bigquery_native.document_parser import BigQueryDocumentParser
from src.bigquery_native.embedder import BigQueryEmbedder
from src.bigquery_native.retriever import BigQueryVectorRetriever
from src.bigquery_native.generator import BigQuerySQLGenerator
from src.bigquery_native.pipeline import BigQueryNativeRAGPipeline


def test_object_table_ddl():
    mgr = BigQueryObjectTableManager(project_id="test-project", dataset_id="rag_benchmark", bucket_name="test-bucket")
    ddl = mgr.get_create_object_table_sql()
    assert "CREATE EXTERNAL TABLE" in ddl
    assert "object_metadata = 'SIMPLE'" in ddl
    assert "gs://test-bucket/documents/*.pdf" in ddl


def test_document_parser_sql():
    parser = BigQueryDocumentParser(project_id="test-project", dataset_id="rag_benchmark")
    process_sql = parser.get_process_document_sql()
    assert "ML.PROCESS_DOCUMENT" in process_sql
    assert "chunk_size" in process_sql
    assert "include_ancestor_headings" in process_sql

    model_sql = parser.get_create_docai_model_sql()
    assert "CLOUD_AI_DOCUMENT_v1" in model_sql

    extract_sql = parser.get_extract_chunks_sql()
    assert "ml_process_document_result" in extract_sql
    assert "UNNEST(JSON_EXTRACT_ARRAY" in extract_sql


def test_autonomous_embedder_sql():
    embedder = BigQueryEmbedder(
        project_id="test-project",
        dataset_id="rag_benchmark",
        model_id="text-embedding-005",
        connection_id="us.vertex-connection"
    )
    create_sql = embedder.get_create_table_with_autonomous_embedding_sql()
    assert "AI.EMBED" in create_sql
    assert "STORED OPTIONS( asynchronous = TRUE )" in create_sql
    assert "text-embedding-005" in create_sql
    assert "us.vertex-connection" in create_sql

    alter_sql = embedder.get_alter_table_add_autonomous_embedding_sql()
    assert "ALTER TABLE `test-project.rag_benchmark.document_chunks`" in alter_sql
    assert "AI.EMBED" in alter_sql
    assert "STORED OPTIONS( asynchronous = TRUE )" in alter_sql

    readiness_sql = embedder.get_readiness_sql()
    assert "COUNTIF(content_embedding IS NOT NULL AND content_embedding.status = '')" in readiness_sql
    embedder.client = None
    status = embedder.check_readiness()
    assert status["is_ready"] is True
    assert status["percent_complete"] == 100.0


def test_ai_search_retriever_sql():
    retriever = BigQueryVectorRetriever(project_id="test-project", dataset_id="rag_benchmark")
    index_sql = retriever.get_create_vector_index_sql()
    assert "CREATE VECTOR INDEX" in index_sql
    assert "content_embedding.result" in index_sql
    assert "index_type = 'IVF'" in index_sql
    assert "distance_type = 'COSINE'" in index_sql
    
    search_sql = retriever.get_ai_search_sql("Revenue FY2024", top_k=5)
    assert "AI.SEARCH" in search_sql
    assert "'content'" in search_sql
    assert "Revenue FY2024" in search_sql
    assert "top_k => 5" in search_sql


def test_sql_generator_query():
    gen = BigQuerySQLGenerator(project_id="test-project", dataset_id="rag_benchmark")
    sql = gen.get_end_to_end_sql_rag_query("What was the net income?", top_k=5)
    assert "ML.GENERATE_TEXT" in sql
    assert "AI.SEARCH" in sql
    assert "What was the net income?" in sql


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_BQ_TESTS") != "true",
    reason="Requires live BigQuery GCP connection and RUN_LIVE_BQ_TESTS=true"
)
def test_bigquery_pipeline_ingest_and_query():
    project_id = os.environ.get("GCP_PROJECT_ID", "test-project")
    pipeline = BigQueryNativeRAGPipeline(project_id=project_id, dataset_id="rag_benchmark")
    metrics = pipeline.ingest("data/documents/sample_50page.pdf")
    assert metrics.total_cost_usd > 0
    assert "docai_processing_cost_usd" in metrics.cost_breakdown
    assert metrics.cost_breakdown["docai_processing_cost_usd"] > 0

    answer, results, q_metrics = pipeline.query("What was the net income?", top_k=3)
    assert answer is not None
    assert isinstance(results, list)
    assert q_metrics.total_cost_usd >= 0
