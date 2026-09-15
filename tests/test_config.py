from src.common.config import load_config


def test_load_config():
    config = load_config("config/pricing.yaml")
    assert bool(config.project_id)
    assert config.models.gemini_flash.model_name in ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert config.models.embedding.model_name == "text-embedding-004"
    assert config.bigquery.query_price_per_tb == 6.25
    assert config.document_ai.layout_parser_price_per_1k_pages == 30.00

