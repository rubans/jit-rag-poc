import logging
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from src.common.models import RetrievalResult

logger = logging.getLogger(__name__)


class FlashGenerator:
    """
    Synthesizes grounded answers using Gemini Flash with retrieved markdown context.
    """
    def __init__(
        self,
        project_id: Optional[str] = None,
        model_name: str = "gemini-3.8-flash",
        allow_fallback: bool = False
    ):
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
        self.model_name = model_name
        self.allow_fallback = allow_fallback
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from google import genai
            location = "global" if "3." in self.model_name else "us-central1"
            self.client = genai.Client(vertexai=True, project=self.project_id, location=location)
        except Exception as e:
            if not self.allow_fallback:
                raise RuntimeError(
                    f"Vertex AI Client initialization failed for generator in project '{self.project_id}': {e}"
                ) from e
            self.client = None

    def generate_answer(self, query: str, results: List[RetrievalResult]) -> Tuple[str, Dict[str, Any]]:
        context_blocks = []
        for idx, r in enumerate(results, 1):
            page_info = f" (Page {r.chunk.page_number})" if r.chunk.page_number else ""
            context_blocks.append(f"[Context {idx}{page_info}]\n{r.chunk.content}")

        context_str = "\n\n".join(context_blocks)
        prompt = (
            "You are an enterprise research assistant. Answer the user question accurately and concisely "
            "based ONLY on the provided document context. If the answer cannot be determined from the "
            "context, state that the information is unavailable.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        metadata = {
            "input_tokens": len(prompt.split()) * 2,
            "output_tokens": 50,
            "model": self.model_name
        }

        if not self.client and not self.allow_fallback:
            raise RuntimeError(
                f"Vertex AI client is not initialized for generator. Cannot generate live answer for model '{self.model_name}'."
            )

        if self.client:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt
                    )
                    answer = response.text or ""
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        metadata["input_tokens"] = getattr(response.usage_metadata, "prompt_token_count", metadata["input_tokens"])
                        metadata["output_tokens"] = getattr(response.usage_metadata, "candidates_token_count", metadata["output_tokens"])
                    return answer.strip(), metadata
                except Exception as e:
                    err_msg = str(e)
                    if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < max_retries - 1:
                        backoff = (2 ** attempt) * 2 + 1
                        logger.warning(f"Rate limited (429) on {self.model_name}. Backing off for {backoff}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(backoff)
                        continue
                    if not self.allow_fallback:
                        raise RuntimeError(
                            f"Failed to generate answer via Vertex AI model '{self.model_name}': {e}"
                        ) from e

        # Fallback generator for offline unit test mode only
        if results:
            first_chunk_text = results[0].chunk.content[:200]
            answer = f"Based on the documentation: {first_chunk_text}..."
        else:
            answer = "Information unavailable in the provided document context."
            
        return answer, metadata

