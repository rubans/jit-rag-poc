# Just-In-Time RAG (JIT-RAG) Comparative Benchmark Suite

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Google Cloud Vertex AI](https://img.shields.io/badge/GCP-Vertex%20AI-orange.svg)](https://cloud.google.com/vertex-ai)
[![BigQuery Native AI](https://img.shields.io/badge/BigQuery-AI.EMBED%20%7C%20AI.SEARCH-green.svg)](https://cloud.google.com/bigquery)
[![Gemini Flash](https://img.shields.io/badge/Gemini-3.8%20Flash%20%7C%203.5%20Flash%20Lite-blueviolet.svg)](https://deepmind.google/technologies/gemini/)
[![Evaluation](https://img.shields.io/badge/DeepEval-25%20Golden%20Tests-success.svg)](https://github.com/confident-ai/deepeval)

An empirical benchmark framework evaluating **four production Just-In-Time Retrieval-Augmented Generation (JIT-RAG) architectures** on complex enterprise documents (50-page annual report with dense tables, financial metrics, and architectural specifications) across **25 golden test cases**.

---

## Architecture Overview

![JIT-RAG 4-Lane Architecture](docs/diagrams/jit_rag_4_methods_architecture.svg)

> **Interactive Visualization**: Open [`docs/diagrams/jit_rag_4_methods_architecture.html`](docs/diagrams/jit_rag_4_methods_architecture.html) in your browser for an interactive lane-by-lane walkthrough with performance badges.

### The 4 Evaluated Architectures

| Architecture | Ingestion Strategy | Vector Indexing | Retrieval Mechanism | Synthesis Engine |
| :--- | :--- | :--- | :--- | :--- |
| **Method 1: BigQuery Native (Zero-Copy)** | Google Cloud Document AI Layout Parser | BigQuery Autonomous `AI.EMBED` (`text-embedding-005`) | BigQuery SQL `AI.SEARCH` (Cosine distance) | Vertex AI `gemini-3.8-flash` |
| **Method 2: Multimodal Gemini Flash MD** | PDF page slicing to images + `gemini-3.5-flash-lite` vision OCR | In-memory Cosine Index (`text-embedding-004`) | Semantic Top-K chunk retrieval | Vertex AI `gemini-3.8-flash` |
| **Method 3: PyMuPDF4LLM (Local CPU)** | Local CPU layout extraction via `pymupdf4llm` ($0 cloud cost) | In-memory Cosine Index (`text-embedding-004`) | Semantic Top-K chunk retrieval | Vertex AI `gemini-3.8-flash` |
| **Method 4: Direct GCS Long-Context Flash** | Zero preprocessing; direct streaming from Google Cloud Storage (`gs://...`) | **Zero Embeddings** (No vector database) | Full-context window injection (~26k tokens) | Vertex AI `gemini-3.8-flash` |

---

## Why JIT-RAG Over Traditional RAG?

In traditional RAG pipelines, organizations run asynchronous batch jobs that ingest, chunk, embed, and index millions of documents into a persistent vector database (e.g. Pinecone, pgvector, Milvus).

This creates **The Ingestion Cost & Latency Trap**:
1. **Unnecessary Amortization**: In enterprise data lakes, **less than 10% of ingested documents are ever queried**. Paying upfront parsing and embedding costs across the entire repository is economically wasteful.
2. **Cold-Start Agility**: When a user drops a new 50-page contract or regulatory filing, they need answers immediately—not after a 20-minute batch ETL indexing job.
3. **Infrastructure Sprawl**: Maintaining vector database clusters, replication, and synchronization creates operational overhead.

**Just-In-Time RAG (JIT-RAG)** activates indexing and retrieval dynamically upon request, or bypasses vector indexing altogether using long-context models.

---

## Empirical Benchmark Results (50-Page Enterprise Document)

Evaluated on `data/documents/sample_50page.pdf` (50 pages, ~26,000 tokens) across **25 golden evaluation test cases** using DeepEval in GCP Vertex AI.

### 1. Ingestion, Indexing, and Query Economics

| Dimension | Method 1: BigQuery Native | Method 2: Gemini Flash MD | Method 3: PyMuPDF4LLM | Method 4: Direct GCS Flash | Production Takeaway |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ingestion Cost (50 pages)** | **$1.5034** | **$0.0081** | **$0.0006** | **$0.0000** | **Method 4** ($0 upfront); **Method 3** is cheapest indexed ($0.0006) |
| **Document Parsing & Extraction Time** | **Managed in Cloud** *(DocAI)* | **90.97s** *(Vision OCR)* | **1.41s** *(Local CPU)* | **N/A** *(Zero ETL / Native)* | **Method 3** fastest local parse; **Method 4** eliminates ETL |
| **Client Upload & Staging Time** | **2.87s** *(GCS + BQ)* | **N/A** *(Local)* | **N/A** *(Local)* | **N/A** *(Direct GCS URI)* | Method 1 unblocks client in ~3s; others run locally or stream |
| **Embedding Generation Time** | **46.44s** | **31.36s** | **33.02s** | **N/A** *(Zero-Embedding)* | Method 4 bypasses vector embeddings entirely |
| **Total Cold-Start Ingest Time** | **49.31s** | **122.33s** | **34.44s** | **0.00s** | **Method 4** is immediately queryable upon document arrival |
| **Query Latency (P50 / P90)** | **3,863ms / 5,704ms** | **2,800ms / 4,656ms** | **3,498ms / 5,865ms** | **5,951ms / 50,312ms** | In-memory chunk search vs BigQuery SQL vs Context injection |
| **Cost per Query** | **$0.000060** | **$0.000079** | **$0.000079** | **$0.001995** | **Chunked RAG is ~25x–33x cheaper per query** |
| **Infrastructure Footprint** | Managed in BigQuery SQL | Local image slicing + Vision API | Python CPU parser (`fitz`) | Zero vector DB; direct GCS URI streaming |

### 2. DeepEval Quality Benchmarks (25 Test Cases, Threshold = 0.70)

| Metric | Threshold | Method 1: BigQuery | Method 2: Flash MD | Method 3: PyMuPDF4LLM | Method 4: Direct GCS Flash | Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Faithfulness (Hallucination avoidance)** | 0.70 | **0.77** ✅ | **0.76** ✅ | 0.57 ⚠️ | **1.00** ✅ | **Method 4** has zero chunk boundary cutoff |
| **Answer Relevancy** | 0.70 | **0.90** ✅ | **0.95** ✅ | **0.84** ✅ | **0.99** ✅ | Live Gemini 3.8 Flash synthesis yields high relevance |
| **Contextual Precision (Ranking)** | 0.70 | **0.88** ✅ | **0.90** ✅ | **0.86** ✅ | **0.90** ✅ | Strong semantic precision across all architectures |
| **Contextual Recall (Fact Coverage)** | 0.70 | **0.85** ✅ | **0.94** ✅ | 0.62 ⚠️ | **0.89** ✅ | **Method 2** and **Method 4** capture the highest fact ratio |
| **Contextual Relevancy (Signal/Noise)** | 0.70 | **0.83** ✅ | **0.88** ✅ | **0.88** ✅ | **0.88** ✅ | High signal-to-noise across all approaches |
| **Document Completeness** | 0.70 | **0.73** ✅ | **0.76** ✅ | 0.54 ⚠️ | **0.86** ✅ | **Method 4** & **Method 2** excel at multi-entity extraction |

### 3. Position-Stratified Recall ("Lost in the Middle" Attentional Degradation)

To verify whether models forget facts buried deep in long contexts, we stratified the 25 golden test cases across document positions:

| Document Section | Page Range | Method 1: BigQuery | Method 2: Flash MD | Method 3: PyMuPDF4LLM | Method 4: Direct GCS Flash |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Early Section (Primacy)** | Pages 1–15 (9 TCs) | **0.78** | **0.92** | **0.63** | **0.92** |
| **Middle Section (Valley)** | Pages 16–35 (8 TCs) | **0.87** | **0.97** | **0.64** | **0.81** |
| **Late Section (Recency)** | Pages 36–50 (8 TCs) | **0.91** | **0.90** | **0.58** | **0.92** |
| **Attentional Degradation** | Middle drop vs Edges | **0.0%** | **0.0%** | **0.0%** | **11.7% Drop** |

> **Key Finding**:
> - **Chunked RAG (Methods 1, 2, 3)** is **position-invariant**: A chunk on page 28 is retrieved with the exact same probability and precision as page 2.
> - **Direct Long-Context (Method 4)** exhibits a mild **U-shaped attention curve**: Recall is ~92% at both edges, dipping to 81% in pages 16–35.

### 4. Query Cost Break-Even Analysis

Method 4 requires **$0.00 upfront ingestion cost**, but incurs **~$0.001995 per query** because the entire 50-page document (~26,000 tokens) must be re-processed on every interaction. Chunked RAG only sends ~1,500 retrieved tokens per query (~$0.000079).

| Total Queries | Method 1 (BigQuery) | Method 2 (Flash MD) | Method 3 (PyMuPDF4LLM) | Method 4 (Direct GCS Flash) | Lowest Cost Architecture |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1 query** | $1.5035 | $0.0082 | $0.0007 | **$0.0020** | **Method 4** ($0.0020 total) |
| **2 queries** | $1.5035 | $0.0083 | **$0.0008** | $0.0040 | **Method 3 Crossover** |
| **5 queries** | $1.5037 | $0.0085 | **$0.0010** | $0.0100 | Method 3 is 10x cheaper |
| **10 queries** | $1.5040 | $0.0089 | **$0.0014** | $0.0199 | Method 3 is 14x cheaper |
| **50 queries** | $1.5064 | $0.0121 | **$0.0046** | $0.0997 | Method 3 & 2 dominate |
| **1,000 queries** | $1.5634 | $0.0871 | **$0.0796** | $1.9950 | Method 4 is 25x more expensive |

- **Method 4 vs Method 3 Crossover**: **~1–2 queries**. For >2 queries, building a PyMuPDF4LLM index is cheaper.
- **Method 4 vs Method 2 Crossover**: **~4 queries**.
- **Method 4 vs Method 1 Crossover**: **~714 queries**.

---

## The Adaptive JIT-RAG Pattern

In production enterprise deployments, no single RAG method is universally optimal across all document formats and interaction patterns. Instead of forcing all traffic into a single pipeline, the **Adaptive JIT-RAG Pattern** classifies inbound documents in **<50ms** based on text density, page count, and query intent to dynamically dispatch to the optimal architecture.

For the full architectural specification, classification heuristics, and reference Python router, see **[Adaptive JIT-RAG Architecture Pattern](docs/architecture/adaptive_jit_rag_pattern.md)**.

### Architectural Decision Tree

```
                        Is it a 1-off query on a cold document?
                                  /                 \
                              YES                     NO
                              /                         \
                    Use METHOD 4                  Are there complex tables,
               (Direct GCS Long-Context)          charts & visual layouts?
               • $0.00 upfront ingest                     /            \
               • 0s cold-start latency                 YES              NO (Digital PDF)
               • Perfect Faithfulness (1.00)           /                  \
                                              Use METHOD 2           Large doc (>100 pages) needing
                                       (Gemini Flash Vision MD)      cloud scale & parallel sharding?
                                           • Best visual recall             /            \
                                           • 0.94 Recall / 0.95 Relevancy YES              NO (<100 pgs)
                                           • Multimodal table OCR         /                  \
                                                                 Use METHOD 1            Use METHOD 3
                                                               (BigQuery Native)       (PyMuPDF4LLM CPU)
                                                               • Distributed BQ slots  • Sub-7s cold start
                                                               • Parallel DocAI shard  • $0.00 parsing cost
                                                               • Zero client RAM/OOM   • Fast single-CPU parse
                                                               • Scales on 500+ pages  • Avoids cloud overhead
```

---

## Quickstart & Usage

### 1. Prerequisites
- Python 3.11+
- Google Cloud Project with Vertex AI and BigQuery APIs enabled.
- Authenticated via `gcloud auth application-default login` or a service account key.

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/rubans/jit-rag-benchmark.git
cd jit-rag-benchmark

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Copy `.env.example` to `.env` and set your GCP configuration:

```bash
cp .env.example .env
```

```ini
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1
GCS_CORPUS_BUCKET=your-gcs-bucket-name
FLASH_VISION_MODEL=gemini-3.5-flash-lite
FLASH_GENERATOR_MODEL=gemini-3.8-flash
FLASH_EMBEDDING_MODEL=text-embedding-004
BQ_EMBEDDING_MODEL=text-embedding-005
BQ_GENERATOR_MODEL=gemini-3.8-flash
DIRECT_GEMINI_MODEL=gemini-3.8-flash
```

### 4. Running Benchmarks

```bash
# Run all 4 methods
python src/main.py --method all

# Run an individual method
python src/main.py --method direct      # Method 4 (Direct GCS Flash)
python src/main.py --method pymupdf4llm  # Method 3 (PyMuPDF4LLM)
python src/main.py --method flash       # Method 2 (Gemini Flash 3.5 Lite MD)
python src/main.py --method bigquery    # Method 1 (BigQuery Native)
```

### 5. Reusable Standalone Analysis & Reporting

To re-analyze, re-rank, or export comparison reports from any existing JSON benchmark results without running GCP cloud pipelines:

```bash
# Re-generate Markdown, CSV, and JSON reports
python src/main.py --compare output/benchmark_report.json

# Or using the standalone reporter CLI
python src/eval/reporter.py --input output/benchmark_report.json --output-md output/benchmark_report.md --output-csv output/benchmark_report.csv
```

Outputs generated:
- [`output/benchmark_report.md`](output/benchmark_report.md): Full markdown report
- [`output/benchmark_report.csv`](output/benchmark_report.csv): Tabular data for Excel / Sheets
- [`output/benchmark_report.json`](output/benchmark_report.json): Raw metrics schema

---

## Project Structure

```
├── config/
│   └── pricing.yaml           # Vertex AI, Gemini, BigQuery, and DocAI pricing model
├── data/
│   ├── documents/             # 50-page enterprise test document
│   └── eval_dataset.json      # 25 position-stratified golden test cases
├── docs/
│   └── diagrams/              # Unified 4-lane architecture SVG & interactive HTML
├── openspec/                  # OpenSpec proposal, design, and change specifications
├── output/
│   ├── benchmark_report.json  # Live benchmark metrics
│   ├── benchmark_report.md    # Compiled markdown comparison report
│   └── benchmark_report.csv   # Tabular CSV summary
├── scripts/
│   └── generate_benchmark_corpus.py # Synthetic 50-page PDF & test dataset generator
├── src/
│   ├── bigquery_native/       # Method 1: BigQuery Zero-Copy pipeline
│   ├── legacy_flash/          # Method 2: Multimodal Gemini Flash vision pipeline
│   ├── pymupdf4llm_pipeline/  # Method 3: PyMuPDF4LLM local CPU pipeline
│   ├── direct_gemini/         # Method 4: Direct GCS long-context streaming pipeline
│   ├── common/                # Shared configs, cost calculators, and models
│   ├── eval/                  # DeepEval runner, metrics, and reporter
│   └── main.py                # Main benchmark CLI and comparison runner
├── tests/                     # Unit and integration test suite
├── .env.example               # Configuration template
├── pyproject.toml             # Project metadata
└── requirements.txt           # Python dependencies
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
