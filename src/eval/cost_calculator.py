from typing import Dict, Any
from src.common.config import AppConfig, load_config


class CostCalculator:
    """
    Computes precise dollar costs based on resource consumption and pricing rate cards.
    """
    def __init__(self, config: AppConfig = None):
        self.config = config or load_config()

    def calculate_legacy_flash_ingestion_cost(self, num_pages: int = 50) -> Dict[str, float]:
        cfg = self.config.models.gemini_flash
        emb_cfg = self.config.models.embedding

        # Vision tokens
        vision_tokens = num_pages * cfg.image_tokens_per_page
        vision_cost = (vision_tokens / 1_000_000.0) * cfg.input_price_per_1m_tokens

        # Markdown generation output (~600 tokens/page)
        gen_tokens = num_pages * 600
        gen_cost = (gen_tokens / 1_000_000.0) * cfg.output_price_per_1m_tokens

        # Embeddings (~2,500 characters/page)
        emb_chars = num_pages * 2500
        emb_cost = (emb_chars / 1_000_000.0) * emb_cfg.price_per_1m_characters

        total = vision_cost + gen_cost + emb_cost
        return {
            "page_image_vision_cost_usd": round(vision_cost, 6),
            "markdown_generation_cost_usd": round(gen_cost, 6),
            "chunk_embedding_cost_usd": round(emb_cost, 6),
            "total_ingestion_cost_usd": round(total, 6),
            "cost_per_page_usd": round(total / num_pages, 6)
        }

    def calculate_bigquery_native_ingestion_cost(self, num_pages: int = 50) -> Dict[str, float]:
        docai_cfg = self.config.document_ai
        bq_cfg = self.config.bigquery
        emb_cfg = self.config.models.embedding

        # Document AI Layout Parser ($30 / 1k pages)
        docai_cost = (num_pages / 1000.0) * docai_cfg.layout_parser_price_per_1k_pages

        # Embeddings (~2,500 chars/page)
        emb_chars = num_pages * 2500
        emb_cost = (emb_chars / 1_000_000.0) * emb_cfg.price_per_1m_characters

        # BigQuery query execution for table build (approx 50 MB scanned)
        bq_compute_cost = (0.05 / 1024.0) * bq_cfg.query_price_per_tb

        total = docai_cost + emb_cost + bq_compute_cost
        return {
            "document_ai_layout_parser_cost_usd": round(docai_cost, 6),
            "ml_generate_embedding_cost_usd": round(emb_cost, 6),
            "bigquery_compute_cost_usd": round(bq_compute_cost, 6),
            "total_ingestion_cost_usd": round(total, 6),
            "cost_per_page_usd": round(total / num_pages, 6)
        }

    def calculate_direct_gemini_ingestion_cost(self, num_pages: int = 50) -> Dict[str, float]:
        """Method 4 bypasses document parsing and embedding indexing entirely."""
        return {
            "parsing_cost_usd": 0.0,
            "embedding_cost_usd": 0.0,
            "storage_cost_usd": 0.0,
            "total_ingestion_cost_usd": 0.0,
            "cost_per_page_usd": 0.0
        }

    def calculate_query_cost(self, pipeline_type: str, input_tokens: int = 1500, output_tokens: int = 150, bq_bytes_billed: int = 10_485_760) -> float:
        ptype = pipeline_type.lower()
        if ptype.startswith("legacy") or ptype == "flash" or ptype == "pymupdf4llm":
            cfg = self.config.models.gemini_flash
            in_cost = (input_tokens / 1_000_000.0) * cfg.input_price_per_1m_tokens
            out_cost = (output_tokens / 1_000_000.0) * cfg.output_price_per_1m_tokens
            return round(in_cost + out_cost, 6)
        elif ptype.startswith("direct"):
            cfg = self.config.direct_gemini
            # For direct long-context JIT, the whole 50-page PDF (~26,000 tokens) is sent on every query
            direct_input_tokens = 26000 if input_tokens == 1500 else input_tokens
            in_cost = (direct_input_tokens / 1_000_000.0) * cfg.input_price_per_1m_tokens
            out_cost = (output_tokens / 1_000_000.0) * cfg.output_price_per_1m_tokens
            return round(in_cost + out_cost, 6)
        else:
            # BigQuery VECTOR_SEARCH scanned bytes
            tb_billed = bq_bytes_billed / (1024.0 ** 4)
            return round(tb_billed * self.config.bigquery.query_price_per_tb, 6)


