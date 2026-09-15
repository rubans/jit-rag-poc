## Purpose

Provides a traditional, client-orchestrated RAG pipeline serving as the baseline, using PDF page splitting, per-page multimodal Gemini Flash markdown extraction, markdown chunking, and vector retrieval.

## ADDED Requirements

### Requirement: Client-Side PDF Page Splitting and Image Rendering
The system SHALL split multi-page PDF documents into individual page objects and render each page into high-resolution images (300 DPI PNG/JPEG) via client-side PDF processing (`pymupdf`).

#### Scenario: 50-page PDF page splitting
- **WHEN** a 50-page PDF document is submitted to the legacy ingestion pipeline
- **THEN** the system splits the file into exactly 50 discrete page images stored in a local working cache directory with ordered page index metadata

### Requirement: Per-Page Multimodal Markdown Conversion via Gemini Flash
The system SHALL iterate through the rendered page images, invoking the Gemini Flash multimodal API per page to extract visual text, headers, and tables into discrete GitHub-Flavored Markdown files (`page_001.md` through `page_050.md`).

#### Scenario: Per-page markdown generation
- **WHEN** page images are sent to Gemini Flash with layout extraction prompts
- **THEN** the system saves an individual `.md` file for each page containing formatted tables, headers, and body text while capturing token usage for cost accounting

### Requirement: Structure-Aware Markdown Chunking
The system SHALL parse the generated page markdown files into semantic chunks using header demarcations (`#`, `##`, `###`) and table block preservation with configurable chunk token limits.

#### Scenario: Markdown chunk creation
- **WHEN** the markdown pages are processed by the chunker
- **THEN** chunks are generated respecting section boundaries and keeping table structures intact without mid-row truncation

### Requirement: Vector Store Indexing and Semantic Retrieval
The system SHALL generate dense vector embeddings for the markdown chunks and index them into a vector retriever supporting top-k cosine similarity queries.

#### Scenario: Query retrieval over markdown index
- **WHEN** a user query is received
- **THEN** the retriever converts the query to an embedding, searches the markdown chunk index, and returns the top-k most relevant chunks along with source page numbers and relevance scores

### Requirement: Client-Side Generation with Retrieved Context
The system SHALL format retrieved markdown context into an augmented prompt, invoke Gemini to generate a response, and output standardized response payloads for DeepEval testing.

#### Scenario: Response generation
- **WHEN** top-k chunks are retrieved for a benchmark query
- **THEN** the system synthesizes the prompt, calls Gemini, and returns the generated answer alongside the retrieved context list
