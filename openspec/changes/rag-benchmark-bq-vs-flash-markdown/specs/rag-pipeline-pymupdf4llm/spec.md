## Purpose

Provides a local CPU-based Just-In-Time RAG (JIT-RAG) pipeline utilizing `pymupdf4llm` for ultra-fast, zero-cloud-cost PDF-to-Markdown extraction, with structure-aware chunking, dense vector indexing, and Gemini generation matching Method 2.

## ADDED Requirements

### Requirement: Local CPU PDF to Markdown Extraction via PyMuPDF4LLM
The system SHALL extract formatted GitHub-Flavored Markdown from multi-page PDF documents entirely on local CPU using `pymupdf4llm.to_markdown()` with page chunks enabled, without rendering intermediate images or invoking cloud vision APIs.

#### Scenario: 50-page PDF CPU conversion
- **WHEN** a 50-page PDF document is submitted to the Method 3 ingestion pipeline
- **THEN** the system extracts all 50 pages into structured Markdown in under 5 seconds with $0.00 cloud parsing cost and stores individual page markdown files in an isolated run directory (`runs/<run_id>/markdown/page_001.md` ... `page_050.md`).

### Requirement: Automated Run Sandboxing and Cache Purging
The system SHALL assign each JIT-RAG execution a unique `run_id` and automatically purge prior run artifacts on invocation to guarantee clean, unpolluted evaluation across runs.

#### Scenario: Isolated JIT-RAG run
- **WHEN** a new ingestion run is initiated
- **THEN** previous run caches are deleted automatically and new outputs are isolated under `runs/<run_id>/`.

### Requirement: Shared Structure-Aware Markdown Chunking
The system SHALL parse the generated page markdown files into semantic chunks using the shared `MarkdownChunker` respecting section headers (`#`, `##`, `###`) and preserving table grids intact.

#### Scenario: Chunk extraction from PyMuPDF4LLM markdown
- **WHEN** the extracted markdown pages are passed to the chunker
- **THEN** discrete semantic chunks are generated with source page numbers and preserved tabular structures.

### Requirement: Shared Dense Vector Indexing and Semantic Retrieval
The system SHALL generate dense vector embeddings for the markdown chunks using `text-embedding-004` and index them into an in-memory cosine similarity retriever.

#### Scenario: Query retrieval over PyMuPDF4LLM chunks
- **WHEN** a benchmark query is received
- **THEN** the retriever generates the query embedding and returns the top-k ranked chunks with cosine similarity scores and page numbers.

### Requirement: Grounded Answer Synthesis
The system SHALL format the retrieved chunks into an augmented context prompt and invoke the configured generator model (`gemini-3.8-flash`) to produce grounded answers for DeepEval benchmarking.

#### Scenario: Response generation
- **WHEN** top-k chunks are retrieved for a benchmark query
- **THEN** the system calls Gemini and returns the synthesized response alongside the retrieved context list.
