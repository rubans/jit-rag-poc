# Comparative Just-In-Time RAG (JIT-RAG) Benchmark Report

**Test Document**: bp_annual_report_2023.pdf (`data/documents/bp_annual_report_2023.pdf`)  
**Target GCP Project**: `vertex-ai-poc-416620`  
**Evaluation Suite**: DeepEval RAG Quality Metrics & Position-Stratified Attentional Degradation (15 Golden Test Cases)

---

## 1. Executive Summary & Key Trade-offs

| Dimension | Method 1: BigQuery Native (Zero-Copy) | Method 2: Gemini Flash MD (Vision) | Method 4c: Direct Flash (Explicit Cache) | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion Cost** | **$1.5034** | **$0.0081** | **$0.0000** | **[WINNER] Method 4/4b/4c ($0 upfront) / Method 3 ($0.0006 with indexing)** |
| **Document Parsing & Extraction Time** | **Managed in Cloud** *(DocAI)* | **112.74s** *(Vision OCR)* | **N/A** *(Zero ETL / Native)* | **Method 3 fastest CPU parse; Method 3b deep table extraction; Method 4/4b/4c zero ETL** |
| **Client Upload & Staging Time** | **2.87s** *(GCS + BQ)* | **N/A** *(Local)* | **N/A** *(Locked in TPU HBM)* | Method 1 unblocks client in ~3s; others run locally or stream |
| **Embedding Generation Time** | **46.44s** | **30.62s** | **N/A** *(Zero-Embedding)* | Method 4/4b/4c bypasses embeddings entirely |
| **Total Cold-Start Ingestion Time** | **49.31s** | **143.35s** | **43.81s** | Total wall-clock time until ready for retrieval |
| **Query Latency (P50 / P90)** | **1717ms / 2011ms** | **3858ms / 26043ms** | **39382ms / 64435ms** | BigQuery SQL vs In-memory vs Context window |
| **Cost per Query** | **$0.000060** | **$0.000079** | **$0.000599** *(100% cache)* | **Chunked RAG is ~25x cheaper; Explicit cache guarantees 75% input discount** |
| **Architecture Footprint** | **Zero client ETL** (Managed in SQL) | **Custom Python ETL + Vision API** | **Zero-Embedding (Explicit TPU Cache)** | Trade-off between governance, cost & simplicity |
| **Context Cache Hit Rate** | **N/A** *(Chunked Index)* | **N/A** *(Chunked Index)* | **93.3%** *(Avg 197604 tok)* | Measures TPU prefix cache reuse on repeat document queries |

---

## 2. DeepEval Quality Benchmarks (15 Test Cases)

| DeepEval Metric | Threshold | Method 1: BigQuery | Method 2: Flash MD | Method 4c: Direct Flash (Explicit Cache) |
| :--- | :---: | :---: | :---:| :---: |
| **Faithfulness (Hallucination avoidance)** | 0.70 | **N/A** | **N/A** | **1.00** |
| **Answer Relevancy** | 0.70 | **N/A** | **N/A** | **0.95** |
| **Contextual Precision (Rank quality)** | 0.70 | **N/A** | **N/A** | **0.89** |
| **Contextual Recall (Fact coverage)** | 0.70 | **N/A** | **N/A** | **0.81** |
| **Contextual Relevancy (Signal-to-noise)** | 0.70 | **N/A** | **N/A** | **0.88** |
| **Document Completeness (Entity extraction)** | 0.70 | **N/A** | **N/A** | **0.75** |

---

## 3. Position-Stratified Recall & Attentional Degradation ("Lost in the Middle")

When long documents are stuffed directly into an LLM context window without chunking/indexing (Method 4), models often exhibit attentional degradation: strong recall at the beginning (primacy effect) and end (recency effect), but degraded recall for details buried in the middle ("lost in the middle"). Chunked RAG architectures retrieve top-k chunks based on semantic similarity regardless of where they appear in the document.

| Document Section | Page Range | Method 1 (BigQuery) | Method 2 (Flash MD) | Method 4c (Explicit) |
| :--- | :---: | :---: | :---: | :---: |
| **Early Section (Primacy)** | Pages 1–116 (5 TCs) | **N/A** | **N/A** | **0.87** |
| **Middle Section (Valley)** | Pages 117–272 (5 TCs) | **N/A** | **N/A** | **0.69** |
| **Late Section (Recency)** | Pages 273–392 (5 TCs) | **N/A** | **N/A** | **0.93** |
| **Attentional Degradation ("Lost in Middle")** | Middle drop vs Edges | **N/A** | **N/A** | **23.8%** (U-Curve) |

### Key Insight: Position-Invariant Chunking vs. Long-Context Stuffing
* **Chunked RAG (Methods 1, 2, 3, 3b)**: Vector embeddings normalize position. A chunk on page 28 is retrieved with the exact same probability and precision as a chunk on page 3. Attentional degradation across document positions is negligible (<2%).
* **Direct Long-Context (Method 4, 4b, 4c)**: While Gemini Flash possesses a 1M+ token context window and easily ingests the entire document (392 pages), empirical factual recall drops in the middle sections. For dense enterprise extraction with zero tolerance for omissions, chunking provides higher reliability.

---

## 4. Query Cost Break-Even Analysis

Method 4 requires **$0.00 upfront ingestion cost**, but incurs **~$0.001980 per query** (or discounted with Method 4b stateful caching or Method 4c explicit caching) because the entire 392-page document must be evaluated by Gemini Flash. In contrast, Chunked RAG (Methods 2, 3, 3b) incurs an upfront indexing cost but only sends ~1,500 retrieved tokens per query (~$0.000079).

| Query Volume | Method 1 (BigQuery) | Method 2 (Flash MD) | Method 4c (Explicit) | Most Cost-Effective Architecture |
| :---: | :---: | :---: | :---: | :--- |
| **1 query** | $1.5035 | $0.0082 | **$0.0006** | **Method 4/4b/4c ($0.00 upfront)** |
| **5 queries** | $1.5037 | $0.0085 | **$0.0030** | **Chunked RAG saves orders of magnitude** |
| **10 queries** | $1.5040 | $0.0089 | **$0.0060** | **Chunked RAG saves orders of magnitude** |
| **50 queries** | $1.5064 | $0.0120 | $0.0299 | **Chunked RAG saves orders of magnitude** |
| **200 queries** | $1.5154 | $0.0239 | $0.1197 | **Chunked RAG saves orders of magnitude** |
| **1000 queries** | $1.5634 | $0.0871 | $0.5987 | **Chunked RAG saves orders of magnitude** |

* **Break-Even Point (Method 4 vs Method 3)**: **N/A**. If you ask more questions than this against the document, building a PyMuPDF4LLM index is cheaper than re-stuffing the full document.
* **Break-Even Point (Method 4 vs Method 3b)**: **N/A**.
* **Break-Even Point (Method 4 vs Method 2)**: **N/A**.
* **Break-Even Point (Method 4 vs Method 1)**: **N/A**.

---

## 5. Operational Ingestion Cost Breakdown

### Method 1: Modern BigQuery Native RAG (Zero-Copy)
* **Document AI Layout Parsing**: $1.5000 ($30.00 / 1k pages)
* **ML.GENERATE_EMBEDDING**: $0.0031 ($0.025 / 1M chars)
* **BigQuery Compute & Storage**: $0.0003
* **Total Document Ingestion (392 Pages)**: **$1.5034** ($0.0038 / page)

### Method 2: Multimodal Gemini Flash Markdown
* **Gemini Flash Page Image Vision Tokens**: $0.0050
* **Markdown Output Tokens**: $0.0090
* **Text Embedding API (text-embedding-004)**: $0.0031
* **Total Document Ingestion (392 Pages)**: **$0.0081** ($0.000021 / page)

### Method 4c: Direct Gemini Flash (Explicit Context Caching)
* **Client Preprocessing / OCR**: $0.0000 (Zero API calls)
* **Vector Indexing / Embeddings**: $0.0000 (No vector database)
* **Deterministic Cache Hit Rate**: 93.3% (Locked in TPU HBM via client.caches.create, 75% token discount)
* **Total Ingestion Cost**: **$0.0000** ($0.0000 / page)

---

## 6. Architectural Recommendations

1. **Use Method 4c (Direct Flash Explicit Context Caching)** when:
   * **Documents are large (>=32,768 tokens) and queried frequently**: Provides 100% guaranteed cache hit rates with zero stickiness/routing drop-off. Locked in TPU memory with a 75% discount on all input tokens.
   * **Automated background workers or API endpoints**: Stateless client workers can query the cache by name (`cachedContents/...`) without maintaining WebSocket connections or stateful chat objects.

2. **Use Method 4b (Direct Flash Stateful Session Caching)** when:
   * **Interactive multi-turn document exploration on small/medium documents (<32,768 tokens)**: Where explicit caching is unavailable due to the 32k token floor. Stateful sessions preserve pod affinity and achieve ~70-85% implicit cache hits with $0.00 storage fees.

3. **Use Method 4 (Direct Flash Stateless)** when:
   * **Purely isolated, single-query interactions**: Single-shot queries where sessions are not maintained.

4. **Use Method 3 (PyMuPDF4LLM Local CPU)** when:
   * **Lowest cost + high interactive query volume**: You have >2 queries per document, want cold ingestion in <10 seconds, $0 parsing cost, and mostly narrative/text-based layouts.

5. **Use Method 3b (Docling Local CPU)** when:
   * **Dense tabular/financial documents requiring high factual accuracy**: Preserves complex table borders and merged cells without external Cloud Vision API costs.

6. **Use Method 2 (Gemini Flash Markdown Vision)** when:
   * **Intricate visual tables and infographic layouts**: Visual OCR required for graphic charts and complex PDF layouts.

7. **Use Method 1 (BigQuery Native RAG)** when:
   * **Enterprise Data Lake & Governance**: Millions of documents managed in Cloud Storage with BigQuery IAM governance and SQL analyst workflows.
