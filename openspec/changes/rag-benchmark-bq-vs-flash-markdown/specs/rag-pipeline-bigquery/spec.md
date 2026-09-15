## Purpose

Provides an end-to-end cloud-native RAG pipeline utilizing BigQuery's latest data-centric AI features: Object Tables, ML.PROCESS_DOCUMENT, Autonomous Embedding Generation (AI.EMBED), native vector indexes, AI.SEARCH, and SQL generation via ML.GENERATE_TEXT.

## ADDED Requirements

### Requirement: Cloud Storage Object Table Ingestion
The system SHALL configure and query BigQuery Object Tables referencing PDF documents stored in Google Cloud Storage buckets via an authorized Cloud Resource external connection.

#### Scenario: Registering PDF files in BigQuery Object Tables
- **WHEN** raw PDF documents (including the 50-page benchmark document) are uploaded to a designated Cloud Storage URI
- **THEN** the system registers or queries the external Object Table exposing URI, metadata, and byte content directly within BigQuery without copying raw bytes into internal tables

### Requirement: In-Database Document Parsing
The system SHALL support in-warehouse document parsing. The architectural preference is to utilize BigQuery's native `AI.PARSE_DOCUMENT` function directly against Object Tables. However, because `AI.PARSE_DOCUMENT` is not yet functional in preview (due to an internal BigQuery platform signature mismatch where the compiler accepts at most 2 arguments while runtime demands 3), the system SHALL invoke the `ML.PROCESS_DOCUMENT` function in BigQuery against Object Tables using a Document AI processor connection (or cloud-native layout extraction) to extract layout, text blocks, and structured tables into SQL-queryable columns.

#### Scenario: Preferred native document parsing via AI.PARSE_DOCUMENT
- **WHEN** `AI.PARSE_DOCUMENT` becomes available and functional in Google Cloud BigQuery preview
- **THEN** the system invokes `AI.PARSE_DOCUMENT(uri, options => JSON '{"chunk_size": 250}')` directly over Object Tables without requiring external processor management

#### Scenario: Production automated PDF layout and text extraction via ML.PROCESS_DOCUMENT
- **WHEN** `ML.PROCESS_DOCUMENT` executes on the document Object Table records
- **THEN** the system generates structured JSON records containing parsed page text, block hierarchies, and tabular grids preserved for chunking

### Requirement: Autonomous In-Database Embedding Generation via AI.EMBED
The system SHALL generate dense vector embeddings for extracted document chunks directly within BigQuery using autonomous generated columns (`GENERATED ALWAYS AS (AI.EMBED(...)) STORED OPTIONS( asynchronous = TRUE )`) referencing a Vertex AI connection and embedding endpoint (such as `text-embedding-005` or `text-embedding-004`).

#### Scenario: Managed asynchronous vector embedding generation
- **WHEN** chunks are inserted or modified in the BigQuery chunks table
- **THEN** BigQuery autonomously and asynchronously generates and keeps in sync the `content_embedding STRUCT<result ARRAY<FLOAT64>, status STRING>` column without requiring manual batch ETL query orchestration

### Requirement: Data-Centric Vector Indexing and AI.SEARCH Retrieval
The system SHALL create an Inverted File (IVF) vector index on the autonomous embedding column (`content_embedding.result`) and execute top-k semantic retrieval using `AI.SEARCH`.

#### Scenario: Semantic retrieval via AI.SEARCH
- **WHEN** a natural language query is passed to `AI.SEARCH(TABLE ..., 'content', query, top_k => k)`
- **THEN** BigQuery automatically utilizes the embedding model bound to the column to embed the query and executes the nearest neighbor search, returning top-k matched chunks and cosine distance without requiring manual query vector generation

### Requirement: In-Database Generation via ML.GENERATE_TEXT / ML.GENERATE_CONTENT
The system SHALL support generating grounded final answers directly within SQL by passing context chunks retrieved via `AI.SEARCH` to a Gemini remote model via `ML.GENERATE_TEXT` or `ML.GENERATE_CONTENT`.

#### Scenario: SQL-driven end-to-end RAG response
- **WHEN** a user query executes the complete RAG query statement
- **THEN** BigQuery retrieves the top-k chunks via `AI.SEARCH`, binds them into the prompt template, invokes the Gemini remote model, and returns the grounded answer string alongside metadata

### Requirement: Architecture Trade-off & Vendor Lock-in Accounting
The system SHALL explicitly account for the architectural trade-off of Method 1: while it provides managed in-warehouse security and zero-client infrastructure, it incurs complete vendor lock-in to Google Cloud Platform (BigQuery SQL dialect, Document AI processor models, Vertex AI Cloud Resource Connections, and BigQuery Vector Search), making it non-portable to multi-cloud or open-source database environments.
