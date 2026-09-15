import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Automatically load .env file from project root or current working directory
load_dotenv(override=True)


class GeminiFlashPricing(BaseModel):
    model_name: str = "gemini-3.5-flash-lite"
    vision_model_name: str = "gemini-3.5-flash-lite"
    generator_model_name: str = "gemini-3.8-flash"
    input_price_per_1m_tokens: float = 0.0375
    output_price_per_1m_tokens: float = 0.15
    image_tokens_per_page: int = 258


class EmbeddingPricing(BaseModel):
    model_name: str = "text-embedding-004"
    price_per_1m_characters: float = 0.025


class ModelsPricing(BaseModel):
    gemini_flash: GeminiFlashPricing = Field(default_factory=GeminiFlashPricing)
    embedding: EmbeddingPricing = Field(default_factory=EmbeddingPricing)


class BigQueryPricing(BaseModel):
    query_price_per_tb: float = 6.25
    active_storage_per_gb_month: float = 0.02
    long_term_storage_per_gb_month: float = 0.01
    embedding_model: str = "text-embedding-005"
    generator_model: str = "gemini-3.8-flash"


class DocumentAIPricing(BaseModel):
    layout_parser_price_per_1k_pages: float = 30.00
    ocr_processor_price_per_1k_pages: float = 1.50
    processor_id: str = "your-docai-processor-id"


class DirectGeminiPricing(BaseModel):
    model_name: str = "gemini-3.8-flash"
    input_price_per_1m_tokens: float = 0.075
    output_price_per_1m_tokens: float = 0.30


class AppConfig(BaseModel):
    project_id: str = "your-gcp-project-id"
    location: str = "us-central1"
    models: ModelsPricing = Field(default_factory=ModelsPricing)
    bigquery: BigQueryPricing = Field(default_factory=BigQueryPricing)
    document_ai: DocumentAIPricing = Field(default_factory=DocumentAIPricing)
    direct_gemini: DirectGeminiPricing = Field(default_factory=DirectGeminiPricing)


def load_config(config_path: str = "config/pricing.yaml") -> AppConfig:
    path = Path(config_path)
    data = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    config = AppConfig(**data)

    # Allow .env and process environment variables to override settings easily per run
    env_project = os.environ.get("GCP_PROJECT_ID")
    if env_project:
        config.project_id = env_project

    env_location = os.environ.get("GCP_LOCATION")
    if env_location:
        config.location = env_location

    env_vision_model = os.environ.get("FLASH_VISION_MODEL") or os.environ.get("FLASH_MODEL")
    if env_vision_model:
        config.models.gemini_flash.vision_model_name = env_vision_model
        config.models.gemini_flash.model_name = env_vision_model
        if "lite" in env_vision_model.lower():
            config.models.gemini_flash.input_price_per_1m_tokens = 0.0375
            config.models.gemini_flash.output_price_per_1m_tokens = 0.15
        else:
            config.models.gemini_flash.input_price_per_1m_tokens = 0.075
            config.models.gemini_flash.output_price_per_1m_tokens = 0.30

    env_gen_model = os.environ.get("FLASH_GENERATOR_MODEL")
    if env_gen_model:
        config.models.gemini_flash.generator_model_name = env_gen_model

    env_emb_model = os.environ.get("FLASH_EMBEDDING_MODEL")
    if env_emb_model:
        config.models.embedding.model_name = env_emb_model

    env_bq_emb = os.environ.get("BQ_EMBEDDING_MODEL")
    if env_bq_emb:
        config.bigquery.embedding_model = env_bq_emb

    env_bq_gen = os.environ.get("BQ_GENERATOR_MODEL")
    if env_bq_gen:
        config.bigquery.generator_model = env_bq_gen

    env_docai = os.environ.get("DOCUMENT_AI_PROCESSOR_ID")
    if env_docai:
        config.document_ai.processor_id = env_docai

    env_direct = os.environ.get("DIRECT_GEMINI_MODEL")
    if env_direct:
        config.direct_gemini.model_name = env_direct
        if "lite" in env_direct.lower():
            config.direct_gemini.input_price_per_1m_tokens = 0.0375
            config.direct_gemini.output_price_per_1m_tokens = 0.15
        else:
            config.direct_gemini.input_price_per_1m_tokens = 0.075
            config.direct_gemini.output_price_per_1m_tokens = 0.30

    return config

