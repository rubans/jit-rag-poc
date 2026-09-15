from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_number: Optional[int] = None
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk: DocumentChunk
    score: float
    rank: int


class ExecutionMetrics(BaseModel):
    pipeline_name: str
    ingestion_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)


class BenchmarkItemResult(BaseModel):
    test_case_id: str
    pipeline_name: str
    input: str
    actual_output: str
    retrieved_contexts: List[str]
    expected_output: str
    context_ground_truth: List[str]
    scores: Dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0

