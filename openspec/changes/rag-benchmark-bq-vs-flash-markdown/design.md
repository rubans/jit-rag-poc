## Context

See `proposal.md` for detailed background and the core rationale on **Why JIT-RAG is required over Traditional RAG**:
* **Traditional RAG** relies on static, pre-indexed offline knowledge bases where high upfront ingestion costs ($1.00–$5.00/doc) and 24/7 dedicated vector search clusters ($500+/mo) are amortized over hundreds of thousands of queries across months.
* **Just-In-Time RAG (JIT-RAG)** addresses the real-world operational requirement of ad-hoc, session-bound documents (e.g. newly uploaded 50-to-200-page earnings filings, loan packages, legal contracts) where an analyst asks only 3 to 15 targeted questions. Under this dynamic paradigm, cold-start ingestion latency must be under 30s (or <1s for direct streaming), upfront cost cannot exceed pennies per document, run isolation (`runs/<run_id>`) must prevent multi-tenant contamination, and zero idle infrastructure can be tolerated.

This framework provides a reproducible, automated benchmark in Python contrasting four distinct architectures for **Just-In-Time Retrieval-Augmented Generation (JIT-RAG)**:
1. **Method 1 (BigQuery Cloud-Native Zero-Copy JIT-RAG with Autonomous Embeddings & AI.SEARCH)**: In-warehouse document analysis and retrieval running on BigQuery's data-centric AI capabilities (Object Tables over GCS, `ML.PROCESS_DOCUMENT`, Autonomous Embedding Generation via `AI.EMBED` with `STORED OPTIONS(asynchronous = TRUE)`, IVF `VECTOR_INDEX`, semantic search via `AI.SEARCH`, and Vertex AI Gemini answer generation).
2. **Method 2 (Multimodal Gemini Flash Markdown JIT-RAG)**: On-demand client-orchestrated pipeline splitting a PDF into individual pages, converting each page to Markdown with Gemini Flash (`gemini-3.5-flash-lite`) concurrently via `ThreadPoolExecutor`, chunking the markdown text, indexing chunks in a local vector store (`text-embedding-004`), and synthesizing answers with `gemini-3.8-flash`.
3. **Method 3 (PyMuPDF4LLM Local CPU Markdown JIT-RAG)**: Ultra-fast on-demand local CPU pipeline using `pymupdf4llm` to convert the PDF directly to structured Markdown in ~1.37s with **$0.00 cloud parsing cost**, then reusing the exact same structure-aware chunker, dense vector indexer, and Gemini generator.
4. **Method 4 (Direct Long-Context Gemini Flash Zero-Embedding JIT-RAG)**: Zero-embedding cloud streaming paradigm passing the PDF directly from Google Cloud Storage (`gs://...` via `types.Part.from_uri`) into Gemini Flash's 1M+ token context window on every query, completely eliminating chunking and vector indexing.

All four pipelines are evaluated against a standard **50-page complex PDF** using the **DeepEval** framework alongside dedicated latency profilers, cost accounting rate cards, and position-stratified attentional degradation ("lost in the middle") metrics. In a JIT-RAG setting, each run operates inside an isolated sandbox (`runs/<run_id>`) with automated cache clearing to ensure dynamic, session-based workloads never suffer from cross-document cache pollution.

## Goals / Non-Goals

**Goals:**
- Provide clean, modular implementations of all four pipelines adhering to a unified interface (`BaseRAGPipeline`).
- Standardize on **DeepEval** (`deepeval`) for automated, verifiable evaluation across 6 RAG dimensions (`Faithfulness`, `AnswerRelevancy`, `ContextualPrecision`, `ContextualRecall`, `ContextualRelevancy`, and `DocumentCompleteness`).
- Quantify **Position-Stratified Attentional Degradation ("Lost in the Middle")** comparing uniform chunked retrieval against direct long-context window attention drops across early (p1-15), middle (p16-35), and late (p36-50) document sections.
- Calculate exact **Cost Break-Even Crossover Curves** determining the query volume where upfront chunked indexing becomes cheaper than zero-embedding context stuffing.
- Evaluate across a representative **50-page complex PDF** containing dense tables, financial reports, and multi-column text with a 25-question golden evaluation dataset.
- Profile granular **Latency** (per-page cold ingestion write, background queue readiness, total E2E, retrieval, generation) and calculate exact **Cost** (tokens, BQ bytes billed, Document AI pages).
- Support dynamic configuration via `.env` files and automated cache purging per run.
- Generate side-by-side Markdown and JSON comparison reports across all 4 methods.

**Non-Goals:**
- Building a multi-tenant web application or production user management.
- Static offline pre-indexing of non-ephemeral corporate intranets (focus is strictly on on-demand Just-In-Time RAG).
- Fine-tuning foundational models (we evaluate off-the-shelf Vertex AI and local open-source utilities).

## Decisions

### Decision 1: Evaluation Framework — DeepEval (`deepeval`)
- **Choice**: Use the open-source DeepEval framework by Confident AI, creating `LLMTestCase` objects evaluated by native metric classes:
  - `FaithfulnessMetric(threshold=0.7)`: Detects hallucinations against context.
  - `AnswerRelevancyMetric(threshold=0.7)`: Measures how directly answers address user questions.
  - `ContextualPrecisionMetric(threshold=0.7)`: Validates ranking quality of retrieved passages.
  - `ContextualRecallMetric(threshold=0.7)`: Measures completeness of retrieved facts against ground truth.
  - `ContextualRelevancyMetric(threshold=0.7)`: Validates context signal-to-noise ratio.
- **Rationale**: DeepEval provides production-grade, standardized RAG metric implementations with native pytest integration and clear mathematical scoring.

### Decision 2: Method 1 Architecture — BigQuery Zero-Copy JIT-RAG with Autonomous Embeddings & AI.SEARCH
- **Choice**: Follow Google's reference architecture ([`GoogleCloudPlatform/document-analytics-on-bigquery`](https://github.com/GoogleCloudPlatform/document-analytics-on-bigquery)):
  1. Store raw PDF in GCS (`gs://.../*.pdf`); create BigQuery **Object Table**.
  2. **In-Warehouse Parsing**: The preferred architectural approach is BigQuery's native **`AI.PARSE_DOCUMENT`** SQL function. However, **`AI.PARSE_DOCUMENT` is not yet functional in preview** despite being announced as available earlier in the year (live execution encounters an internal BigQuery platform engine mismatch: the frontend SQL parser accepts at most 2 arguments while the backend execution runtime fails with `400 Invalid number of parameters: expected 3, got 2`). Therefore, the functional production path is **`ML.PROCESS_DOCUMENT`** with a Document AI Layout Parser remote model (`CLOUD_AI_DOCUMENT_v1`) or cloud-native extraction into structured rows (`chunk_size: 250`, `include_ancestor_headings: true`).
  3. Extract structured chunks into relational rows (`chunk_id`, `doc_id`, `page_number`, `content`).
  4. Generate continuous embeddings via **Autonomous Embedding Generation** (`GENERATED ALWAYS AS (AI.EMBED(content, connection_id => '...', endpoint => 'text-embedding-005')) STORED OPTIONS( asynchronous = TRUE )`) or `ML.GENERATE_EMBEDDING`.
  5. Enforce **row-level 100% readiness check** before allowing search queries.
  6. Retrieve with **`AI.SEARCH(TABLE ..., 'content', query, top_k => k)`** or `VECTOR_SEARCH()`.
  7. Ground and synthesize responses using Gemini over retrieved context.
- **Rationale**: Implements Google's official Zero-Copy RAG paradigm where unstructured PDFs stay in Cloud Storage, chunking occurs inside BigQuery SQL, and vectors are autonomously maintained in the warehouse.

### Decision 3: Method 2 Architecture — Multimodal Gemini Flash Markdown JIT-RAG
- **Choice**:
  1. Client-side PDF page splitting using `pymupdf` (fitz) rendering 50 page images at 300 DPI.
  2. Call Gemini Flash API (`gemini-3.5-flash-lite`) concurrently via `ThreadPoolExecutor(max_workers=10)` with a structured prompt requesting GitHub-Flavored Markdown (GFM) preserving visual tables and headings.
  3. Header-aware chunking preserving table integrity.
  4. Vector embedding and retrieval index (`text-embedding-004`).
  5. Client-orchestrated prompt assembly and Gemini generation (`gemini-3.8-flash`).
  6. Automatic cache purging on invocation and unique run sandboxing under `runs/<run_id>`.
- **Rationale**: Delivers exceptional table and visual fidelity at low cost compared to enterprise Document AI.

### Decision 4: Method 3 Architecture — PyMuPDF4LLM Local CPU Markdown JIT-RAG
- **Choice**:
  1. Parse the 50-page PDF directly on local CPU using `pymupdf4llm.to_markdown(pdf_path, page_chunks=True)`.
  2. Save page-level markdown into `runs/<run_id>/markdown/` with automatic sandbox cleanup.
  3. Reuse Method 2's structure-aware chunker, vector indexer (`text-embedding-004`), and Gemini generator (`gemini-3.8-flash`).
- **Rationale**: Completely bypasses cloud vision billing ($0.00 parsing cost) and executes in ~1.1s on CPU. Allows comparing heuristic layout extraction against multimodal LLM vision under identical retrieval and evaluation conditions.

### Decision 5: Test Corpus & Golden Dataset
- **Choice**: Use a 50-page complex real-world document with dense tables and charts. Paired with `data/eval_dataset.json` containing 25 curated test cases:
  ```json
  {
    "id": "q1",
    "input": "What was the total revenue in fiscal year 2023?",
    "expected_output": "The total revenue was $85.4 billion.",
    "context_ground_truth": ["...exact excerpt from Table 4 on page 18..."]
  }
  ```
- **Rationale**: 50 pages is large enough to expose real ingestion throughput, rate limits, table layout failure modes, and retrieval recall degradation without prohibitive benchmark costs.

## Risks / Trade-offs

- **[Risk] Method 1 Proprietary Vendor Lock-in**: Method 1 deeply couples the data pipeline to Google Cloud Platform: proprietary BigQuery GoogleSQL dialects, Document AI processor models, BigQuery Object Tables, Cloud Resource Connections, and BigQuery Vector Search. Unlike Methods 2 and 3 (which use standard open Markdown formats and portable vector search libraries that easily migrate across multi-cloud environments, local edge servers, Snowflake, Databricks, or PostgreSQL/pgvector), Method 1 is strictly non-portable outside of Google Cloud. → **Mitigation**: Standardize retrieval and generation interfaces across all pipelines using common Pydantic schemas (`RetrievalResult`, `DocumentChunk`, `ExecutionMetrics`) so downstream applications remain decoupled from BigQuery's proprietary internals.
- **[Risk] BigQuery `AI.PARSE_DOCUMENT` Preview Instability**: BigQuery's native `AI.PARSE_DOCUMENT` function is currently not functional in preview despite being announced earlier in the year, due to an internal engine signature/runtime parameter mismatch (frontend expects <= 2 args, backend expects 3). → **Mitigation**: Architect the pipeline with an explicit preference for `AI.PARSE_DOCUMENT` once Google fixes the platform bug, while utilizing `ML.PROCESS_DOCUMENT` with Document AI remote models (`CLOUD_AI_DOCUMENT_v1`) or cloud-native extraction as the operational fallback.
- **[Risk] Autonomous Embedding Generation regional & feature availability** → **Mitigation**: Verify dataset and connection are in supported BigQuery regions (`US`), provide automated DDL statements, and retain offline simulation fallbacks for test runner environments.
- **[Risk] DeepEval evaluation token costs** → **Mitigation**: Use `gemini-1.5-flash` as the judge model within DeepEval to minimize evaluation cost while maintaining high scoring fidelity.
- **[Risk] Gemini Flash rate limits during 50-page batch conversion** → **Mitigation**: Implement bounded concurrency with `asyncio.Semaphore(10)` and exponential backoff retry.
- **[Risk] Cloud Resource connection permissions for BigQuery** → **Mitigation**: Provide an automated setup script and documentation for creating the GCP external connection and assigning `roles/aiplatform.user`.

## Migration Plan

Greenfield evaluation project; no data migration required.
