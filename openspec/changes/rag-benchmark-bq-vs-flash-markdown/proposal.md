## Why

### The Paradigm Shift: Why JIT-RAG Over Traditional RAG

Traditional Retrieval-Augmented Generation (Traditional RAG) architectures were designed around the assumption of a **static, pre-indexed enterprise knowledge base** (e.g. corporate wikis, customer support FAQs, fixed policy repositories). In Traditional RAG:
* Documents are ingested once via heavy, offline batch ETL pipelines running overnight or over several days.
* Ingestion costs ($1.00–$5.00 per document for layout parsing, OCR, and chunking) are amortized across thousands or millions of queries over months.
* High-availability dedicated vector search clusters (e.g. Pinecone, Milvus, Vertex AI Vector Search endpoints) run 24/7, incurring hundreds of dollars per month in idle infrastructure costs regardless of actual query volume.

However, enterprise knowledge workers increasingly operate in an **ad-hoc, dynamic document review** reality:
* **Ephemeral, Session-Bound Documents**: Financial analysts, legal teams, underwriting auditors, and procurement managers upload newly received 50-to-200-page PDFs (quarterly earnings reports, vendor contracts, merger due-diligence filings) and need to ask **3 to 15 targeted questions** within a single session.
* **The Ingestion Cost Trap**: If Traditional RAG ingestion ($1.50+ Document AI / OCR) is applied to an ephemeral document queried only 5 times, the effective cost is **$0.30+ per query**, rendering the solution economically unviable compared to direct LLM usage.
* **The Cold-Start Latency Hurdle**: In Traditional RAG, waiting 10 minutes for batch OCR and vector index compilation is unnoticeable. In dynamic enterprise sessions, **a human analyst or autonomous agent is waiting in real time**. Cold-start time until the first query can be answered must be under 30 seconds (or under 1 second with direct context streaming).
* **Multi-Tenant Security & Zero Contamination**: Ad-hoc documents often contain sensitive non-public material (MNPI, confidential deal terms) that must **never** pollute a global shared vector database or leak into other user sessions. JIT-RAG mandates ephemeral sandboxing (`runs/<run_id>`) with automated cache purging and complete lifecycle eviction upon session conclusion.
* **Zero Infrastructure Sprawl**: Maintaining dedicated vector databases, sync pipelines, and external embedding infrastructure for ephemeral documents creates massive operational overhead. JIT-RAG replaces this with serverless, on-demand paradigms (in-warehouse SQL, lightweight local CPU parsing, or zero-embedding direct context windows).

### The Four JIT-RAG Architectural Paradigms

To address these demands, four distinct JIT-RAG architectural patterns exist, each representing a unique balance between cold-start latency, ingestion cost, query cost, and retrieval accuracy:

1. **Method 1: Modern Cloud-Native BigQuery JIT-RAG (Zero-Copy Architecture)**: Following Google's reference architecture ([`GoogleCloudPlatform/document-analytics-on-bigquery`](https://github.com/GoogleCloudPlatform/document-analytics-on-bigquery)), this in-warehouse pipeline stores raw PDF documents in Google Cloud Storage, accesses them via BigQuery **Object Tables**, extracts semantic chunks in-place via **`ML.PROCESS_DOCUMENT`** (Document AI layout chunker), automatically maintains dense vector representations via **Autonomous Embedding Generation** (`AI.EMBED` with `text-embedding-005` in `STORED OPTIONS(asynchronous = TRUE)`), enforces row-level 100% completion verification, and retrieves context via **`AI.SEARCH`**.
2. **Method 2: Multimodal Gemini Flash Markdown JIT-RAG**: A client-orchestrated on-demand workflow that splits PDFs into individual pages, renders page images, invokes multimodal Gemini Flash (`gemini-3.5-flash-lite` / `gemini-3.8-flash`) per page to extract structured GitHub-Flavored Markdown (`page_N.md`), chunks the markdown text, and indexes the chunks into a local in-memory vector store for subsequent retrieval.
3. **Method 3: PyMuPDF4LLM Local CPU Markdown JIT-RAG**: An ultra-low-latency, zero-cloud-cost on-demand workflow that extracts structured Markdown directly from PDF binary text and layout streams on local CPU via `pymupdf4llm` in **~1.37s** with **$0.00 cloud parsing cost**, eliminating both Document AI and Gemini Vision billing while reusing the same structure-aware chunking, dense vector indexing, and Gemini generation for controlled evaluation.
4. **Method 4: Direct Long-Context Gemini Flash JIT-RAG (Zero-Embedding Cloud Streaming)**: An on-demand zero-embedding paradigm where the raw PDF is streamed directly from Google Cloud Storage (`gs://...` via `types.Part.from_uri`) into Gemini Flash's 1M+ token multimodal context window on every query. Ingestion cost is **$0.0000** and cold-start latency is **<0.01s**, but queries cost ~25x–33x more per query and exhibit position-stratified attentional degradation ("lost in the middle").

This change establishes a rigorous comparative benchmarking harness evaluating all four JIT-RAG pipelines against a standard 50-page complex PDF document using **DeepEval**, position-stratified degradation metrics, and granular operational cost and latency profilers.

## What Changes

- Frame the benchmarking harness explicitly around **Just-In-Time RAG (JIT-RAG)**:
  - Isolate each run under ephemeral, unique run sandboxes (`runs/<run_id>/`).
  - Auto-purge stale or previous run caches on each invocation without user prompts.
  - Measure per-page cold ingestion latency, background embedding readiness, and per-document ingestion cost.
  - Calculate query cost break-even crossover points comparing upfront indexing vs. direct context stuffing.
- Implement **Method 1 (Modern Cloud-Native BigQuery JIT-RAG / Zero-Copy Architecture)**:
  - Ingest raw PDF documents via BigQuery Object Tables referencing Cloud Storage.
  - In-warehouse document parsing and semantic chunking using BigQuery **`AI.PARSE_DOCUMENT`** (and `ML.PROCESS_DOCUMENT`) with `chunking_config` (`chunk_size: 250`, `include_ancestor_headings: true`).
  - Autonomous embedding generation via managed `AI.EMBED` column using Vertex AI `text-embedding-005`.
  - Row-level readiness polling verifying 100% embedding completion (`status = ''`) before retrieval.
  - Semantic search execution using BigQuery `AI.SEARCH(TABLE ..., 'content', query, top_k => k)`.
  - Answer synthesis using Gemini over retrieved `AI.SEARCH` context.
- Implement **Method 2 (Multimodal Gemini Flash Markdown JIT-RAG)**:
  - Client-side PDF page-by-page splitting and image rendering.
  - Multimodal page-to-markdown extraction using Gemini Flash (`gemini-3.5-flash-lite`) preserving tables and headings with parallel worker execution (`ThreadPoolExecutor`).
  - Header- and table-aware markdown chunking.
  - Vector embedding generation and similarity retrieval.
  - Client-side prompt compilation and generation with Gemini (`gemini-3.8-flash`).
- Implement **Method 3 (PyMuPDF4LLM Local CPU Markdown JIT-RAG)**:
  - Local CPU PDF-to-Markdown conversion using `pymupdf4llm` without cloud vision APIs ($0.00 parsing cost, ~1.1s execution).
  - Page-by-page markdown output generation matching Method 2's structure.
  - Reuse Method 2's structure-aware chunker, dense vector indexer, and Gemini generator for direct comparison.
- Implement **DeepEval RAG Benchmarking Suite**:
  - Benchmark all three methods on a standardized **50-page complex PDF** containing tables, financial figures, and multi-column text.
  - Curated golden evaluation dataset (questions, expected answers, and ground-truth contexts).
  - Evaluate RAG quality using **DeepEval** (`deepeval`) metrics: `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, and `ContextualRelevancyMetric`.
  - Instrument and profile operational metrics: Client write latency, background queue 100% readiness latency, total E2E ingestion latency, query P50/P90/P99 latency, and granular cost accounting.
  - Generate automated comparative Markdown and JSON benchmark reports.

## Capabilities

### New Capabilities
- `rag-pipeline-bigquery`: Modern BigQuery cloud-native pipeline using Object Tables, `ML.PROCESS_DOCUMENT` chunking, autonomous `AI.EMBED`, and `AI.SEARCH`.
- `rag-pipeline-flash-markdown`: Multimodal pipeline using client-side PDF page splitting, Gemini Flash multimodal markdown extraction per page, markdown chunking, and vector retrieval.
- `rag-pipeline-pymupdf4llm`: Local CPU pipeline using `pymupdf4llm` for ultra-fast zero-cost markdown extraction with vector retrieval.
- `rag-evaluation-benchmarking`: Comprehensive evaluation framework using DeepEval and latency/cost profilers to benchmark all three JIT-RAG methods on a 50-page PDF.

### Modified Capabilities
<!-- None: Greenfield project -->

## Impact

- **Dependencies**: Python 3.11+, `deepeval`, `google-genai`, `google-cloud-bigquery`, `google-cloud-storage`, `pymupdf`, `pymupdf4llm`, `pydantic`.
- **Infrastructure**: BigQuery dataset with Vertex AI Cloud Resource connection (`vertex-connection`), Google Cloud Storage bucket, Document AI processor.
- **Artifacts**: Standardized 50-page test PDF, golden QA dataset, and 3-way comparative benchmark results.
