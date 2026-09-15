---
name: jit-rag-benchmark
description: >-
  Runs, analyzes, and compares Just-In-Time RAG (JIT-RAG) architectures across Method 1 (BigQuery Native Zero-Copy),
  Method 2 (Gemini Flash Vision MD), Method 3 (PyMuPDF4LLM Local CPU), and Method 4 (Direct GCS Long-Context Flash).
  Supports running live benchmarks on GCP Vertex AI or generating offline comparative reports (Markdown, CSV, JSON)
  with DeepEval quality metrics, break-even crossover curves, position-stratified recall, and N/A handling.
license: Apache-2.0
metadata:
  version: "1.0"
---

# Just-In-Time RAG (JIT-RAG) Benchmark Skill

This skill allows the agent to execute, analyze, and generate comparative reports across 4 Just-In-Time RAG architectures on enterprise documents.

---

## 1. Quick Modes of Operation

### Mode A: Offline Comparison Analysis (No GCP Credentials Needed)
Use this mode when benchmark JSON results already exist (e.g. `output/benchmark_report.json`) or when the user asks to analyze, re-rank, or export the comparative tables to Markdown or CSV.

```bash
# Re-generate Markdown, CSV, and JSON comparative reports
python src/main.py --compare output/benchmark_report.json

# Or with custom input/output paths:
python src/eval/reporter.py --input output/benchmark_report.json --output-md output/benchmark_report.md --output-csv output/benchmark_report.csv
```

### Mode B: Live Pipeline Execution (GCP Vertex AI)
Use this mode when running live ingestion and evaluation against GCP project `vertex-ai-poc-416620` (or the configured `GCP_PROJECT_ID` in `.env`).

```bash
# Run all 4 methods sequentially on 50-page PDF
python src/main.py --method all

# Run individual methods using canonical names (or numeric shortcodes / aliases):
python src/main.py --method method1      # Method 1: BigQuery Native (aliases: bigquery, bq, bq-zero-copy, 1)
python src/main.py --method method2      # Method 2: Gemini Flash MD Vision (aliases: flash, flash-vision-md, 2)
python src/main.py --method method3      # Method 3: PyMuPDF4LLM Local CPU (aliases: pymupdf, pymupdf-cpu, 3)
python src/main.py --method method4      # Method 4: Direct Long-Context Flash (aliases: direct, direct-flash, 4)

# Custom test slice:
python src/main.py --method method4 --max-items 5 --pdf data/documents/sample_50page.pdf
```

---

## 2. The 4 Benchmark Architectures

1. **Method 1: BigQuery Native (Zero-Copy)**
   - **Ingestion**: Google Cloud Document AI Layout Parser -> Cloud Storage -> BigQuery External Table.
   - **Vector Indexing**: Autonomous background `AI.EMBED` on `text-embedding-005`.
   - **Retrieval**: SQL `AI.SEARCH` (Cosine distance).
   - **Synthesis**: Vertex AI `gemini-3.8-flash`.

2. **Method 2: Multimodal Gemini Flash Markdown**
   - **Ingestion**: PDF page slicing to images -> `gemini-3.5-flash-lite` vision OCR.
   - **Vector Indexing**: Local in-memory Cosine Index on `text-embedding-004`.
   - **Retrieval**: Semantic Top-K chunk retrieval.
   - **Synthesis**: Vertex AI `gemini-3.8-flash`.

3. **Method 3: PyMuPDF4LLM (Local CPU)**
   - **Ingestion**: Local CPU layout parsing via `pymupdf4llm` ($0.00 cloud parsing cost, ~1.37s client write).
   - **Vector Indexing**: Local in-memory Cosine Index on `text-embedding-004`.
   - **Retrieval**: Semantic Top-K chunk retrieval.
   - **Synthesis**: Vertex AI `gemini-3.8-flash`.

4. **Method 4: Direct GCS Long-Context Flash (Zero-Embedding)**
   - **Ingestion**: Zero preprocessing; direct streaming from Google Cloud Storage (`gs://...`).
   - **Vector Indexing**: **Zero Embeddings**; no vector database.
   - **Retrieval**: Full-context window injection (~26k tokens).
   - **Synthesis**: Vertex AI `gemini-3.8-flash`.

---

## 3. Evaluation & Reporting Rules

When presenting or compiling comparison metrics:
1. **Ingestion Latency Metric Decomposition**:
   - Always split ingestion into distinct architectural dimensions to avoid conflating client-side parsing with cloud staging:
     * **Document Parsing & Extraction Time**: Time spent converting the PDF into text/tables before embedding (Method 1: `Managed in Cloud (DocAI)`, Method 2: `112.74s` Vision OCR, Method 3: `1.41s` Local CPU, Method 4: `N/A` Zero ETL).
     * **Client Upload & Staging Time**: Synchronous time the client is blocked staging files to cloud storage before processing (Method 1: `2.87s` GCS+BQ, Methods 2 & 3: `N/A` Local execution, Method 4: `N/A` Direct GCS URI).
     * **Embedding Generation Time**: Vector index computation (Method 4: `N/A` Zero-Embedding).
     * **Total Cold-Start Ingest Time**: Total wall-clock time until ready for retrieval.
2. **Explicit `N/A` Handling for Architectural Non-Applicability**:
   - Never use `0.00s`, `0.00`, or `0.0%` for pipeline phases or metrics that do not exist or were not evaluated:
     * Method 4 bypasses parsing, staging, and embeddings: mark as `N/A`.
     * Methods 2 & 3 do not stage to cloud storage: mark as `N/A`.
     * Only output `0.0%` if a metric was actually measured and scored zero (such as Method 1/2/3 attentional degradation).
3. **Position-Stratified Recall ("Lost in the Middle")**:
   - Stratify test queries into Early (Pages 1–15), Middle (Pages 16–35), and Late (Pages 36–50).
   - Calculate Attentional Degradation: `(Mean(Early, Late) - Middle) / Mean(Early, Late) * 100%`.
4. **Break-Even Crossover Formula**:
   - Calculate crossover query count:
     $$N_{\text{crossover}} = \frac{\text{IngestionCost}_A - \text{IngestionCost}_B}{\text{QueryCost}_B - \text{QueryCost}_A}$$
5. **Artifacts Output**:
   - Markdown: `output/benchmark_report.md`
   - CSV: `output/benchmark_report.csv`
   - JSON: `output/benchmark_report.json`
   - Diagrams: `docs/diagrams/jit_rag_4_methods_architecture.svg` and `.html`

---

## 4. Agent Guidelines for Invoking this Skill

- If the user asks **"compare the RAG methods"** or **"show benchmark results"**:
  Run `python src/main.py --compare output/benchmark_report_50page.json` (or `output/benchmark_report.json`) to verify the latest metrics, then present the side-by-side comparison tables.
- If the user asks **"evaluate Method 4"** or **"test on a new document"**:
  Run `python src/main.py --method method4 --pdf <path> --dataset <path>`.
- If the user asks for **spreadsheet data**:
  Point them to `output/benchmark_report_50page.csv` (or `output/benchmark_report.csv`) or read from it directly.
- **Safety & Cost Constraint**:
  Do NOT run Method 1 (`ML.PROCESS_DOCUMENT` via Document AI) on massive documents like `bp_annual_report_2023.pdf` (392 pages @ $30/1k pages = ~$12/run). Strictly run Method 1 on `sample_50page.pdf` unless explicit user confirmation is provided.

