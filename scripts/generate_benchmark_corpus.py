"""
Generates a representative 50-page complex enterprise annual report PDF
containing tables, headers, multi-column financial metrics, and creates
the paired 25-question golden evaluation dataset.
"""

from pathlib import Path
import json
import fitz  # PyMuPDF


def generate_50page_pdf(output_pdf_path: str) -> None:
    doc = fitz.open()
    Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)

    # Content definition for each of the 50 pages
    pages_data = [
        {
            "page": 1,
            "title": "VERTEX GLOBAL ENTERPRISE CORP.",
            "subtitle": "Annual Report & Financial Review (Fiscal Year 2024)",
            "body": "This document contains audited consolidated financial results, segment operational performance, "
                    "strategic technical architectures, risk disclosures, and forward-looking statements for Vertex Global Enterprise Corp. "
                    "for the fiscal year ended December 31, 2024.",
            "table": [
                ["Document Type", "Annual Report 10-K Equivalent"],
                ["Filing Date", "February 15, 2025"],
                ["Jurisdiction", "Delaware, United States"],
                ["Primary Exchange", "NASDAQ: VTEX"],
            ]
        },
        {
            "page": 2,
            "title": "Table of Contents & Guide",
            "subtitle": "Overview of Report Sections",
            "body": "Part I: Executive Summary (Pages 3-5)\n"
                    "Part II: Segment Performance & Product Engineering (Pages 6-15)\n"
                    "Part III: Consolidated Financial Statements (Pages 16-25)\n"
                    "Part IV: Technical Infrastructure & AI Modernization (Pages 26-35)\n"
                    "Part V: Risk Factors & Market Analysis (Pages 36-45)\n"
                    "Part VI: Corporate Governance & Sustainability (Pages 46-50)\n",
            "table": None
        },
        {
            "page": 3,
            "title": "Executive Summary: Chairman's Letter",
            "subtitle": "Accelerating Sustainable Enterprise Growth",
            "body": "In 2024, Vertex Global Enterprise delivered record financial results. Total consolidated revenue "
                    "grew 18.5% year-over-year to $84.2 billion, driven by surging demand for autonomous cloud data platforms "
                    "and enterprise AI workloads. Our operating margin expanded by 240 basis points to 28.2%, reflecting strong operating leverage.",
            "table": [
                ["Financial Metric", "FY 2023", "FY 2024", "YoY Change"],
                ["Consolidated Revenue", "$71.05 B", "$84.20 B", "+18.5%"],
                ["Operating Income", "$18.33 B", "$23.74 B", "+29.5%"],
                ["Net Income", "$14.21 B", "$18.69 B", "+31.5%"],
                ["Diluted EPS", "$4.12", "$5.48", "+33.0%"],
            ]
        },
        {
            "page": 4,
            "title": "Strategic Milestones of 2024",
            "subtitle": "Cloud and Modernization Progress",
            "body": "Key strategic achievements in 2024 included the migration of 92% of core analytical workloads to "
                    "serverless BigQuery architectures, achieving a 42% reduction in query idle cost. Additionally, "
                    "the enterprise launched its next-generation generative AI suite across 12,000 corporate accounts.",
            "table": [
                ["Strategic Initiative", "Target Metric", "Actual 2024 Result", "Status"],
                ["Cloud Migration", "85% workload migration", "92% core analytics migrated", "Exceeded"],
                ["Analytical Cost Optimization", "30% idle reduction", "42% idle compute saved", "Exceeded"],
                ["Enterprise GenAI Adoption", "10,000 corporate seats", "12,450 corporate seats", "Exceeded"],
                ["Global Data Center PUE", "PUE < 1.18", "PUE 1.14 achieved", "Achieved"],
            ]
        },
        {
            "page": 5,
            "title": "Capital Allocation and Shareholder Returns",
            "subtitle": "Disciplined Capital Deployment",
            "body": "The Board of Directors approved a $15.0 billion expansion of our share repurchase authorization. "
                    "During 2024, we returned $12.4 billion to shareholders via $9.2 billion in common share repurchases "
                    "and $3.2 billion in quarterly cash dividends ($0.94 per share annualized).",
            "table": [
                ["Capital Use", "Amount (in Millions)", "Percentage of Cash Flow"],
                ["Share Repurchases", "$9,200 M", "48.2%"],
                ["Cash Dividends", "$3,200 M", "16.8%"],
                ["R&D Investments", "$4,800 M", "25.1%"],
                ["Strategic M&A", "$1,900 M", "9.9%"],
            ]
        }
    ]

    # Pages 6-15: Segment Operations
    for p in range(6, 16):
        pages_data.append({
            "page": p,
            "title": f"Business Segment Operations: Unit {p-5}",
            "subtitle": f"Operational Metrics for Regional Division {p-5}",
            "body": f"Operational performance for Regional Division {p-5} showed consistent customer expansion. "
                    f"Active enterprise subscriptions in this division reached {15000 + p * 820:,} organizations, "
                    f"with Net Promoter Scores (NPS) averaging {64 + (p % 8)} across mid-market and global accounts. "
                    f"Infrastructure availability SLA was maintained at 99.995% across all availability zones.",
            "table": [
                ["Operational KPI", "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
                ["Active Subscriptions", f"{12000 + p*500}", f"{13000 + p*600}", f"{14000 + p*700}", f"{15000 + p*820}"],
                ["Customer Churn Rate", "1.4%", "1.2%", "1.1%", "0.9%"],
                ["Average Contract Value", "$124,000", "$131,000", "$138,000", "$145,000"],
                ["Support SLA Compliance", "99.8%", "99.9%", "99.9%", "99.95%"],
            ]
        })

    # Pages 16-25: Consolidated Financials
    for p in range(16, 26):
        pages_data.append({
            "page": p,
            "title": f"Consolidated Financial Statements: Note {p-15}",
            "subtitle": f"Financial Analysis & Asset Balance (Schedule {p-15})",
            "body": f"Financial disclosure Note {p-15} presents detailed breakdowns of capital assets, debt obligations, "
                    f"and liquidity reserves. As of December 31, 2024, total corporate liquidity stood at ${24.5 + (p*0.3):.2f} billion, "
                    f"consisting of cash, cash equivalents, and short-term US Treasury bills with weighted average maturity of 4.2 months.",
            "table": [
                ["Balance Sheet Category", "Carrying Value ($M)", "Fair Value ($M)", "Effective Yield"],
                ["Cash and Equivalents", f"${8500 + p*100}", f"${8500 + p*100}", "5.12%"],
                ["Short-Term US Treasuries", f"${12000 + p*150}", f"${12050 + p*150}", "5.28%"],
                ["Corporate Commercial Paper", f"${3000 + p*50}", f"${3010 + p*50}", "5.45%"],
                ["Total Liquid Reserves", f"${23500 + p*300}", f"${23560 + p*300}", "5.24%"],
            ]
        })

    # Pages 26-35: Technical Architecture & AI
    for p in range(26, 36):
        pages_data.append({
            "page": p,
            "title": f"Technical Infrastructure & Architecture: Module {p-25}",
            "subtitle": f"System Architecture & Data Pipeline Specification {p-25}",
            "body": f"Architecture Module {p-25} details the enterprise data fabric. The platform processes {45 + p*2} billion "
                    f"daily analytical events using BigQuery storage and Google Cloud Storage object tables. Vector search indices "
                    f"utilize 768-dimensional embeddings generated via text-embedding-004, supporting cosine distance queries with "
                    f"sub-100 millisecond response times across partitioned shards.",
            "table": [
                ["Pipeline Component", "Technology / Specification", "Throughput Limit", "Target Latency"],
                ["Raw Storage", "GCS Multi-Region Object Store", "100 TB / day", "< 50 ms"],
                ["Table Engine", "BigQuery External Object Table", "1M reads / sec", "< 25 ms"],
                ["Embedding Engine", "Vertex AI text-embedding-004", "50,000 req / min", "< 80 ms"],
                ["Vector Indexing", "BigQuery IVF (Inverted File)", "10M vectors", "< 150 ms"],
            ]
        })

    # Pages 36-45: Risk Factors
    for p in range(36, 46):
        pages_data.append({
            "page": p,
            "title": f"Risk Factors & Regulatory Compliance: Category {p-35}",
            "subtitle": f"Risk Management Matrix & Mitigation Protocols {p-35}",
            "body": f"Section {p-35} itemizes operational and regulatory risks. Cyber defense telemetry reported zero critical "
                    f"security breaches in 2024. SOC2 Type II, ISO 27001, and HIPAA compliance audits were passed with {100 - (p % 3)}% "
                    f"satisfaction scores. Data sovereignty mechanisms guarantee data isolation across 18 geographic jurisdictions.",
            "table": [
                ["Identified Risk", "Impact Severity", "Probability", "Mitigation Strategy"],
                ["Cross-Region Outage", "High", "Low", "Multi-region automatic failover with RPO < 1 min"],
                ["API Quota Exhaustion", "Medium", "Medium", "Dynamic backoff queue with exponential jitter"],
                ["Model Hallucination", "High", "Medium", "RAG Groundedness filter with DeepEval threshold > 0.70"],
                ["Data Drift in Vectors", "Medium", "Low", "Bi-weekly vector index re-clustering & audit"],
            ]
        })

    # Pages 46-50: Governance & Sustainability
    for p in range(46, 51):
        pages_data.append({
            "page": p,
            "title": f"Corporate Governance & ESG: Dimension {p-45}",
            "subtitle": f"Environmental, Social, and Governance Metrics {p-45}",
            "body": f"In 2024, our data center carbon footprint decreased by {14 + p % 4}% relative to baseline 2020 metrics. "
                    f"Renewable energy purchasing contracts covered {95 + (p % 5)}% of total electricity consumption across all "
                    f"wholly-owned enterprise compute clusters.",
            "table": [
                ["ESG Metric", "2022", "2023", "2024", "2025 Target"],
                ["Renewable Power Match", "82%", "89%", f"{95 + (p%5)}%", "100%"],
                ["Water Usage Effectiveness (WUE)", "1.42", "1.28", "1.15", "< 1.10"],
                ["Board Independence", "80%", "80%", "85%", "85%"],
                ["Supplier Code Compliance", "94%", "97%", "99.2%", "100%"],
            ]
        })

    for pdata in pages_data:
        page = doc.new_page(width=612, height=792)  # Standard Letter size
        
        # Header banner
        page.draw_rect(fitz.Rect(36, 36, 576, 75), color=(0.1, 0.2, 0.45), fill=(0.1, 0.2, 0.45))
        page.insert_text((46, 58), pdata["title"], fontsize=13, color=(1, 1, 1), fontname="helv")
        
        # Subtitle
        page.insert_text((36, 95), pdata["subtitle"], fontsize=11, color=(0.2, 0.2, 0.2), fontname="helv")
        
        # Body text
        text_rect = fitz.Rect(36, 115, 576, 230)
        page.insert_textbox(text_rect, pdata["body"], fontsize=10, fontname="helv", color=(0.1, 0.1, 0.1))
        
        # Table if present
        if pdata["table"]:
            y_start = 245
            row_height = 24
            table_rect = fitz.Rect(36, y_start, 576, y_start + len(pdata["table"]) * row_height)
            page.draw_rect(table_rect, color=(0.7, 0.7, 0.7))
            
            for r_idx, row in enumerate(pdata["table"]):
                row_y = y_start + r_idx * row_height
                # Header row fill
                if r_idx == 0:
                    page.draw_rect(fitz.Rect(36, row_y, 576, row_y + row_height), fill=(0.9, 0.93, 0.98), color=(0.7, 0.7, 0.7))
                else:
                    page.draw_rect(fitz.Rect(36, row_y, 576, row_y + row_height), color=(0.85, 0.85, 0.85))
                
                col_width = (576 - 36) / len(row)
                for c_idx, cell in enumerate(row):
                    cell_x = 42 + c_idx * col_width
                    page.insert_text((cell_x, row_y + 16), str(cell), fontsize=8.5, fontname="helv", color=(0, 0, 0))
        
        # Footer
        footer_text = f"Page {pdata['page']} of 50 | Vertex Global Enterprise Corp. FY2024 Report"
        page.insert_text((36, 765), footer_text, fontsize=8, color=(0.5, 0.5, 0.5), fontname="helv")
        page.draw_line(fitz.Point(36, 755), fitz.Point(576, 755), color=(0.8, 0.8, 0.8), width=0.5)

    doc.save(output_pdf_path)
    doc.close()
    print(f"Successfully generated 50-page PDF at: {output_pdf_path}")


def generate_golden_dataset(output_json_path: str) -> None:
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 25 realistic, high-fidelity test cases mapped to specific pages and tables
    test_cases = [
        {
            "id": "TC-01",
            "page": 3,
            "input": "What was the consolidated revenue of Vertex Global Enterprise in FY 2024 and what was the year-over-year growth rate?",
            "expected_output": "The consolidated revenue in FY 2024 was $84.20 billion, representing an 18.5% year-over-year increase from $71.05 billion in FY 2023.",
            "context_ground_truth": ["Consolidated Revenue FY 2023: $71.05 B, FY 2024: $84.20 B, YoY Change: +18.5%"]
        },
        {
            "id": "TC-02",
            "page": 3,
            "input": "What was the operating margin expansion and operating income for FY 2024?",
            "expected_output": "The operating margin expanded by 240 basis points to 28.2%, and operating income reached $23.74 billion, up 29.5% from $18.33 billion in FY 2023.",
            "context_ground_truth": ["Operating margin expanded by 240 basis points to 28.2%", "Operating Income: $18.33 B in FY23 to $23.74 B in FY24 (+29.5%)"]
        },
        {
            "id": "TC-03",
            "page": 4,
            "input": "What percentage of core analytical workloads were migrated to BigQuery in 2024, and what was the idle cost reduction achieved?",
            "expected_output": "92% of core analytical workloads were migrated to serverless BigQuery architectures, achieving a 42% reduction in query idle cost.",
            "context_ground_truth": ["migration of 92% of core analytical workloads to serverless BigQuery architectures, achieving a 42% reduction in query idle cost."]
        },
        {
            "id": "TC-04",
            "page": 4,
            "input": "What was the achieved Global Data Center Power Usage Effectiveness (PUE) in 2024?",
            "expected_output": "The achieved Global Data Center PUE in 2024 was 1.14, beating the target of PUE < 1.18.",
            "context_ground_truth": ["Global Data Center PUE Target Metric PUE < 1.18, Actual 2024 Result PUE 1.14 achieved."]
        },
        {
            "id": "TC-05",
            "page": 5,
            "input": "How much total capital was returned to shareholders in 2024 across share repurchases and dividends?",
            "expected_output": "Vertex returned a total of $12.4 billion to shareholders, comprising $9.2 billion in share repurchases and $3.2 billion in cash dividends.",
            "context_ground_truth": ["returned $12.4 billion to shareholders via $9.2 billion in common share repurchases and $3.2 billion in quarterly cash dividends"]
        },
        {
            "id": "TC-06",
            "page": 5,
            "input": "What was the total dollar amount and percentage of cash flow spent on R&D investments in 2024?",
            "expected_output": "R&D investments were $4,800 million ($4.8 billion), representing 25.1% of cash flow.",
            "context_ground_truth": ["R&D Investments: $4,800 M, 25.1% of Cash Flow"]
        },
        {
            "id": "TC-07",
            "page": 6,
            "input": "What was the customer churn rate reported in Q4 2024 for Regional Division 1?",
            "expected_output": "The customer churn rate in Q4 2024 for Regional Division 1 was 0.9%.",
            "context_ground_truth": ["Customer Churn Rate Q1: 1.4%, Q2: 1.2%, Q3: 1.1%, Q4 2024: 0.9%"]
        },
        {
            "id": "TC-08",
            "page": 8,
            "input": "What was the infrastructure availability SLA maintained across availability zones in Division 3?",
            "expected_output": "The infrastructure availability SLA was maintained at 99.995% across all availability zones.",
            "context_ground_truth": ["Infrastructure availability SLA was maintained at 99.995% across all availability zones."]
        },
        {
            "id": "TC-09",
            "page": 10,
            "input": "What was the average contract value in Q4 2024 for Business Segment Operations Unit 5?",
            "expected_output": "The average contract value in Q4 2024 was $145,000.",
            "context_ground_truth": ["Average Contract Value: $124,000 in Q1, $131,000 in Q2, $138,000 in Q3, $145,000 in Q4 2024"]
        },
        {
            "id": "TC-10",
            "page": 16,
            "input": "What was the total corporate liquidity and weighted average maturity of US Treasury bills as of December 31, 2024?",
            "expected_output": "Total corporate liquidity stood at $29.30 billion, consisting of cash, equivalents, and US Treasuries with a weighted average maturity of 4.2 months.",
            "context_ground_truth": ["liquidity reserves stood at $29.30 billion, consisting of cash, cash equivalents, and short-term US Treasury bills with weighted average maturity of 4.2 months."]
        },
        {
            "id": "TC-11",
            "page": 18,
            "input": "What was the effective yield reported for Short-Term US Treasuries under Note 3?",
            "expected_output": "The effective yield for Short-Term US Treasuries was 5.28%.",
            "context_ground_truth": ["Short-Term US Treasuries effective yield: 5.28%"]
        },
        {
            "id": "TC-12",
            "page": 20,
            "input": "What was the carrying value of Corporate Commercial Paper reported under Note 5?",
            "expected_output": "The carrying value of Corporate Commercial Paper under Note 5 was $4,000 million ($4.0 billion).",
            "context_ground_truth": ["Corporate Commercial Paper: Carrying Value $4000 M"]
        },
        {
            "id": "TC-13",
            "page": 26,
            "input": "How many daily analytical events does the enterprise data fabric process?",
            "expected_output": "The platform processes 97 billion daily analytical events using BigQuery storage and GCS object tables.",
            "context_ground_truth": ["The platform processes 97 billion daily analytical events using BigQuery storage and Google Cloud Storage object tables."]
        },
        {
            "id": "TC-14",
            "page": 26,
            "input": "What embedding model and vector dimension are used for vector search indices in Module 1?",
            "expected_output": "Vector search indices utilize 768-dimensional embeddings generated via text-embedding-004.",
            "context_ground_truth": ["Vector search indices utilize 768-dimensional embeddings generated via text-embedding-004"]
        },
        {
            "id": "TC-15",
            "page": 28,
            "input": "What is the target latency and throughput limit for the BigQuery External Object Table engine in Module 3?",
            "expected_output": "The table engine throughput limit is 1M reads / sec with a target latency of < 25 ms.",
            "context_ground_truth": ["Table Engine: BigQuery External Object Table, Throughput: 1M reads / sec, Target Latency: < 25 ms"]
        },
        {
            "id": "TC-16",
            "page": 30,
            "input": "What vector index type is specified in the technical infrastructure and what is its vector capacity limit?",
            "expected_output": "BigQuery IVF (Inverted File) vector index with a capacity of 10M vectors and target latency < 150 ms.",
            "context_ground_truth": ["Vector Indexing: BigQuery IVF (Inverted File), Throughput Limit: 10M vectors, Target Latency: < 150 ms"]
        },
        {
            "id": "TC-17",
            "page": 32,
            "input": "What is the request rate limit for the Vertex AI text-embedding-004 embedding engine?",
            "expected_output": "The throughput limit for the embedding engine is 50,000 requests per minute with target latency < 80 ms.",
            "context_ground_truth": ["Embedding Engine: Vertex AI text-embedding-004, Throughput Limit: 50,000 req / min, Target Latency: < 80 ms"]
        },
        {
            "id": "TC-18",
            "page": 36,
            "input": "What is the mitigation strategy and Recovery Point Objective (RPO) for cross-region outages?",
            "expected_output": "Multi-region automatic failover with an RPO of less than 1 minute.",
            "context_ground_truth": ["Cross-Region Outage: Multi-region automatic failover with RPO < 1 min"]
        },
        {
            "id": "TC-19",
            "page": 38,
            "input": "What framework and threshold score are specified to mitigate model hallucinations in the risk matrix?",
            "expected_output": "A RAG Groundedness filter using DeepEval with a threshold score > 0.70.",
            "context_ground_truth": ["Model Hallucination: RAG Groundedness filter with DeepEval threshold > 0.70"]
        },
        {
            "id": "TC-20",
            "page": 40,
            "input": "What is the frequency of vector index re-clustering audits to mitigate vector data drift?",
            "expected_output": "Bi-weekly vector index re-clustering and audit.",
            "context_ground_truth": ["Data Drift in Vectors: Bi-weekly vector index re-clustering & audit"]
        },
        {
            "id": "TC-21",
            "page": 42,
            "input": "Across how many geographic jurisdictions do data sovereignty mechanisms guarantee data isolation?",
            "expected_output": "Data sovereignty mechanisms guarantee data isolation across 18 geographic jurisdictions.",
            "context_ground_truth": ["Data sovereignty mechanisms guarantee data isolation across 18 geographic jurisdictions."]
        },
        {
            "id": "TC-22",
            "page": 46,
            "input": "What was the percentage reduction in data center carbon footprint in 2024 relative to the 2020 baseline?",
            "expected_output": "Data center carbon footprint decreased by 15% relative to the 2020 baseline.",
            "context_ground_truth": ["data center carbon footprint decreased by 15% relative to baseline 2020 metrics."]
        },
        {
            "id": "TC-23",
            "page": 48,
            "input": "What was the Water Usage Effectiveness (WUE) achieved in 2024 and what is the 2025 target?",
            "expected_output": "Water Usage Effectiveness (WUE) was 1.15 in 2024, with a 2025 target of < 1.10.",
            "context_ground_truth": ["Water Usage Effectiveness (WUE) 2022: 1.42, 2023: 1.28, 2024: 1.15, 2025 Target: < 1.10"]
        },
        {
            "id": "TC-24",
            "page": 49,
            "input": "What was the Board Independence percentage reported for 2024?",
            "expected_output": "The Board Independence percentage was 85% in 2024.",
            "context_ground_truth": ["Board Independence: 80% in 2022, 80% in 2023, 85% in 2024, 85% in 2025 Target"]
        },
        {
            "id": "TC-25",
            "page": 50,
            "input": "What was the Supplier Code Compliance rate reported for 2024?",
            "expected_output": "The Supplier Code Compliance rate was 99.2% in 2024.",
            "context_ground_truth": ["Supplier Code Compliance: 94% in 2022, 97% in 2023, 99.2% in 2024, 100% in 2025 Target"]
        }
    ]
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2)
    print(f"Successfully generated 25 test cases in: {output_json_path}")


if __name__ == "__main__":
    generate_50page_pdf("data/documents/sample_50page.pdf")
    generate_golden_dataset("data/eval_dataset_50page.json")

