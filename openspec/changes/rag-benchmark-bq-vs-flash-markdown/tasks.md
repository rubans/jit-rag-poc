## 1. Project Setup & Benchmark Corpus

- [x] 1.1 Create Python virtual environment and install core dependencies (`deepeval`, `google-genai`, `google-cloud-bigquery`, `google-cloud-storage`, `pymupdf`, `pydantic`, `pytest`) and verify clean imports.
- [x] 1.2 Define pricing rate card (`config/pricing.yaml`) capturing BigQuery storage/query rates, Document AI per-page cost, and Gemini Flash/embedding token rates, verifying configuration loading tests pass.
- [x] 1.3 Add 50-page complex PDF benchmark document to `data/documents/sample_50page.pdf` and verify page count with `pymupdf`.
- [x] 1.4 Construct golden evaluation dataset in `data/eval_dataset.json` with 25+ curated test cases (`input`, `expected_output`, `context_ground_truth`), verifying schema validation test passes.

## 2. Method 2: Legacy Client-Side Markdown Pipeline

- [x] 2.1 Implement PDF page splitter and 300-DPI renderer in `src/legacy_flash/pdf_splitter.py` generating 50 discrete page images, verifying image generation test passes.
- [x] 2.2 Implement multimodal Gemini Flash page-to-markdown extractor in `src/legacy_flash/page_converter.py` outputting `page_001.md` through `page_050.md` preserving tables, verifying table formatting test passes.
- [x] 2.3 Implement structure-aware markdown chunker in `src/legacy_flash/markdown_chunker.py` respecting `#`/`##` headers and table boundaries, verifying chunk boundary test passes.
- [x] 2.4 Implement embedding generator and vector index retriever in `src/legacy_flash/retriever.py`, verifying top-k retrieval returns ranked chunks with page references.
- [x] 2.5 Implement client-side generation wrapper in `src/legacy_flash/generator.py` formatting context and invoking Gemini Flash, verifying answer synthesis.

## 3. Method 1: Modern BigQuery Cloud-Native Pipeline

- [x] 3.1 Implement BigQuery Object Table provisioning script in `src/bigquery_native/object_table.py` referencing the 50-page PDF in GCS, verifying table metadata query executes.
- [x] 3.2 Implement `AI.PARSE_DOCUMENT` / `ML.PROCESS_DOCUMENT` extraction in `src/bigquery_native/document_parser.py` extracting structured layout, text, and tables with semantic chunking options, verifying parsed records exist in BQ.
- [x] 3.3 Implement autonomous embedding generation (`AI.EMBED` column) in `src/bigquery_native/embedder.py` generating dense vectors with Vertex AI `text-embedding-005` in region `US`, verifying readiness check gates retrieval.
- [x] 3.4 Implement `AI.SEARCH` vector retrieval in `src/bigquery_native/retriever.py`, verifying top-k vector search results without mock fallbacks.
- [x] 3.5 Implement `ML.GENERATE_TEXT` / `ML.GENERATE_CONTENT` query in `src/bigquery_native/generator.py` generating grounded answers directly in SQL, verifying SQL RAG response output.

## 4. DeepEval Evaluation Suite & Operational Profiling

- [x] 4.1 Implement latency profiler in `src/eval/profiler.py` instrumenting each phase (client ingestion write, background 100% ready, total E2E, retrieval, generation), verifying latency metrics recording.
- [x] 4.2 Implement granular cost calculator in `src/eval/cost_calculator.py` tracking token usage, BQ bytes billed, and Document AI charges, verifying cost computation tests pass.
- [x] 4.3 Implement DeepEval test harness in `src/eval/deepeval_runner.py` executing `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, and `ContextualRelevancyMetric` across the 25+ test cases, verifying DeepEval evaluation run.
- [x] 4.4 Implement comparative report generator in `src/eval/reporter.py` compiling side-by-side Markdown and JSON comparison summaries, verifying report formatting.

## 5. End-to-End Benchmark Execution

- [x] 5.1 Implement unified CLI in `src/main.py` running the complete benchmark across both methods against the 50-page PDF and outputting final DeepEval scores, latency, and cost comparison tables, verifying end-to-end execution.

## 6. Method 3: PyMuPDF4LLM Local CPU Just-In-Time RAG Pipeline

- [ ] 6.1 Implement `PyMuPDF4LLMRAGPipeline` in `src/pymupdf4llm_pipeline/pipeline.py` extracting PDF to Markdown via `pymupdf4llm.to_markdown(..., page_chunks=True)` on local CPU, writing page markdown to isolated `runs/<run_id>`, and reusing `MarkdownChunker`, `FlashMarkdownRetriever`, and `FlashGenerator`.
- [ ] 6.2 Write unit tests in `tests/test_pymupdf4llm_pipeline.py` verifying CPU conversion speed, chunk generation, and retrieval execution.
- [ ] 6.3 Update `src/main.py` and `src/eval/reporter.py` to support `--method pymupdf4llm` and produce 3-way comparative reports (BigQuery vs. Flash Vision vs. PyMuPDF4LLM).
- [ ] 6.4 Execute live benchmark evaluation for Method 3 across all 25 golden test cases and compare DeepEval quality, cost, and latency against Methods 1 & 2.
