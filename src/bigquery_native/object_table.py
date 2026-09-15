from typing import Dict, Any, Optional
import os


class BigQueryObjectTableManager:
    """
    Manages Cloud Storage object tables in BigQuery for unstructured PDF documents.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: str = "rag_benchmark",
        connection_id: Optional[str] = None,
        bucket_name: Optional[str] = None
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        self.dataset_id = dataset_id
        self.connection_id = connection_id or os.environ.get("BQ_CONNECTION_ID", "us-central1.vertex-connection")
        self.bucket_name = bucket_name or os.environ.get("GCS_CORPUS_BUCKET", "your-gcs-bucket-name")
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from google.cloud import bigquery
            self.client = bigquery.Client(project=self.project_id)
        except Exception:
            self.client = None

    def get_create_object_table_sql(self, table_name: str = "document_objects") -> str:
        """
        Generates BigQuery DDL for creating an external Object Table referencing GCS.
        """
        return f"""
CREATE EXTERNAL TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.{table_name}`
WITH CONNECTION `{self.project_id}.{self.connection_id}`
OPTIONS (
  object_metadata = 'SIMPLE',
  uris = ['gs://{self.bucket_name}/documents/*.pdf']
);
""".strip()

    def get_query_objects_sql(self, table_name: str = "document_objects") -> str:
        return f"""
SELECT uri, content_type, size, updated
FROM `{self.project_id}.{self.dataset_id}.{table_name}`
ORDER BY updated DESC;
""".strip()

    def provision_object_table(self, table_name: str = "pdf_object_table") -> Dict[str, Any]:
        """
        Executes or dry-runs object table creation.
        """
        ddl = self.get_create_object_table_sql(table_name)
        if self.client:
            try:
                from google.cloud import bigquery
                job = self.client.query(ddl)
                job.result()
                return {"status": "SUCCESS", "ddl": ddl, "table": f"{self.dataset_id}.{table_name}"}
            except Exception as e:
                return {"status": "ERROR", "error": str(e), "ddl": ddl}
        return {"status": "DRY_RUN", "ddl": ddl, "table": f"{self.dataset_id}.{table_name}"}

    def setup_object_table(self, table_name: str = "pdf_object_table"):
        """Ensures object table exists and refreshes metadata cache."""
        self.provision_object_table(table_name)
        if self.client:
            try:
                refresh_sql = f"CALL BQ.REFRESH_EXTERNAL_METADATA_CACHE('{self.project_id}.{self.dataset_id}.{table_name}');"
                self.client.query(refresh_sql).result()
            except Exception:
                pass

    def upload_pdf(self, pdf_path: str, target_blob_name: Optional[str] = None) -> str:
        """
        Uploads a PDF file to the GCS corpus bucket under /documents/ and returns the gs:// URI.
        """
        from google.cloud import storage
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)
        blob_name = target_blob_name or f"documents/{os.path.basename(pdf_path)}"
        blob = bucket.blob(blob_name)
        if not blob.exists():
            print(f"[ObjectTableManager] Uploading {pdf_path} to gs://{self.bucket_name}/{blob_name}...")
            blob.upload_from_filename(pdf_path)
        return f"gs://{self.bucket_name}/{blob_name}"

