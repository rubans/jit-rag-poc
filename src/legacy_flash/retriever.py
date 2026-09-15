import logging
import math
import os
import time
from typing import List, Optional
from src.common.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class FlashMarkdownRetriever:
    """
    Computes dense vector embeddings for Markdown chunks and retrieves top-k
    semantically similar chunks for a given query.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        model_name: str = "text-embedding-004",
        allow_fallback: bool = False
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.client = None
        self.chunks: List[DocumentChunk] = []
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self.client = genai.Client(vertexai=True, project=self.project_id, location="us-central1")
        except Exception as e:
            logger.error("Failed to initialize Vertex AI client for project '%s': %s", self.project_id, e)
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Vertex AI Client initialization failed for project '{self.project_id}': {e}. "
                    f"Fallback embeddings are strictly disabled for runs."
                ) from e
            self.client = None

    def get_embedding(self, text: str) -> List[float]:
        """
        Retrieves 768-dimensional embedding from Vertex AI.
        Fallback embeddings are strictly blocked unless allow_fallback=True is set.
        """
        if not text or not text.strip():
            return [0.0] * 768

        if not self.client:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Vertex AI Client is not initialized for project '{self.project_id}'. "
                    f"Cannot generate live embeddings for model '{self.model_name}'."
                )
            return self._fallback_embedding(text)

        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values
            raise RuntimeError(f"Vertex AI returned empty embedding list for text snippet.")
        except Exception as e:
            logger.error("Vertex AI embed_content failed for model '%s': %s", self.model_name, e)
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Failed to generate live Vertex AI embedding with model '{self.model_name}': {e}. "
                    f"Fallback embeddings are strictly disallowed in runs."
                ) from e
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str, dim: int = 768) -> List[float]:
        import hashlib
        vec = [0.0] * dim
        tokens = text.lower().split()
        if not tokens:
            return vec
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            val = ((h >> 8) % 1000) / 1000.0
            vec[idx] += val
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100):
        self.chunks = chunks
        unindexed = [c for c in self.chunks if c.embedding is None]
        if not unindexed:
            return

        if not self.client:
            for c in unindexed:
                c.embedding = self.get_embedding(c.content)
            return

        # Process in batches for fast network throughput
        for i in range(0, len(unindexed), batch_size):
            batch = unindexed[i : i + batch_size]
            contents = [c.content if c.content.strip() else " " for c in batch]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    res = self.client.models.embed_content(
                        model=self.model_name,
                        contents=contents
                    )
                    if res.embeddings and len(res.embeddings) == len(batch):
                        for c, emb in zip(batch, res.embeddings):
                            c.embedding = emb.values
                    else:
                        for c in batch:
                            c.embedding = self.get_embedding(c.content)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries - 1:
                        backoff = (2 ** attempt) * 2 + 1
                        logger.warning(f"Embedding rate limited (429). Retrying batch in {backoff}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(backoff)
                        continue
                    if attempt == max_retries - 1:
                        for c in batch:
                            c.embedding = self.get_embedding(c.content)
                        break

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.chunks:
            return []

        query_emb = self.get_embedding(query)
        scored: List[RetrievalResult] = []

        for chunk in self.chunks:
            sim = cosine_similarity(query_emb, chunk.embedding)
            scored.append(RetrievalResult(chunk=chunk, score=sim, rank=0))

        # Sort descending by similarity score
        scored.sort(key=lambda r: r.score, reverse=True)
        
        top_results = scored[:top_k]
        for rank_idx, item in enumerate(top_results, start=1):
            item.rank = rank_idx

        return top_results

