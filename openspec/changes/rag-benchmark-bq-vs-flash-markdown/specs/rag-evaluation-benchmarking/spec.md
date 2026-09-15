## Purpose

Provides a comprehensive evaluation framework using DeepEval to benchmark modern BigQuery RAG against the legacy Gemini Flash markdown pipeline on a 50-page complex PDF across cost, latency, and retrieval/generation accuracy.

## ADDED Requirements

### Requirement: 50-Page PDF Benchmark Corpus and Golden Dataset
The system SHALL establish a standardized 50-page complex PDF test document containing tables, financial data, and multi-column text, paired with a curated golden dataset of test cases specifying user input, expected output, and ground-truth contexts.

#### Scenario: Benchmark dataset loading
- **WHEN** the benchmark execution suite is initialized
- **THEN** the test harness loads the 50-page PDF document and validates that the golden dataset contains at least 25 query test cases with verified ground-truth references

### Requirement: DeepEval RAG Quality Metrics Evaluation
The system SHALL integrate the DeepEval framework (`deepeval`) to evaluate both pipelines across five standardized RAG quality metrics: Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall, and Contextual Relevancy.

#### Scenario: DeepEval metric scoring
- **WHEN** evaluation runs execute across test cases for both pipelines
- **THEN** DeepEval calculates and outputs scores (0.0 to 1.0) for:
  - `FaithfulnessMetric` (measures factual adherence to retrieved context without hallucinations)
  - `AnswerRelevancyMetric` (measures how directly the generated answer addresses the question)
  - `ContextualPrecisionMetric` (evaluates whether the most relevant chunks are ranked at top positions)
  - `ContextualRecallMetric` (evaluates whether all ground-truth facts are captured in retrieved context)
  - `ContextualRelevancyMetric` (measures the proportion of retrieved text relevant to the query)

### Requirement: Granular Latency Profiling
The system SHALL capture high-resolution execution timestamps (`perf_counter`) for each processing phase: document ingestion/splitting, conversion/parsing, embedding generation, vector indexing, retrieval execution, and generation.

#### Scenario: Latency profiling report
- **WHEN** benchmark queries execute against both pipelines
- **THEN** the system logs and calculates P50, P90, and P99 latency statistics in milliseconds for each phase and for end-to-end response time

### Requirement: Granular Cost and Resource Accounting
The system SHALL compute total and itemized dollar costs for processing the 50-page document and executing queries across both architectures based on a configurable GCP pricing rate card.

#### Scenario: Cost comparison computation
- **WHEN** both pipelines complete ingestion and query evaluation
- **THEN** the system produces an itemized cost ledger accounting for:
  - BigQuery: Bytes billed for queries and vector search, slot consumption, and Document AI `ML.PROCESS_DOCUMENT` charges.
  - Legacy Pipeline: Gemini Flash input/output token counts for 50 pages of markdown conversion and generation, plus `text-embedding-004` API calls.
  - Final total cost per 50-page document ingestion and average cost per user query.

### Requirement: Comparative Summary Reporting
The system SHALL output side-by-side comparison tables in Markdown format and structured JSON files presenting DeepEval scores, latency percentiles, and cost breakdowns contrasting Method 1 and Method 2.

#### Scenario: Benchmark report generation
- **WHEN** the evaluation suite completes all benchmark runs
- **THEN** the system generates a formatted Markdown report table highlighting key trade-offs and deltas between the two approaches
