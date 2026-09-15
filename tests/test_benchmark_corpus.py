import json
from pathlib import Path
import pymupdf


def test_sample_50page_pdf_exists_and_has_50_pages():
    pdf_path = Path("data/documents/sample_50page.pdf")
    assert pdf_path.exists(), "sample_50page.pdf does not exist"
    doc = pymupdf.open(str(pdf_path))
    assert len(doc) == 50, f"Expected 50 pages, found {len(doc)}"
    doc.close()


def test_bp_pdf_exists_and_has_392_pages():
    pdf_path = Path("data/documents/bp_annual_report_2023.pdf")
    assert pdf_path.exists(), "bp_annual_report_2023.pdf does not exist"
    doc = pymupdf.open(str(pdf_path))
    assert len(doc) == 392, f"Expected 392 pages, found {len(doc)}"
    doc.close()


def test_eval_dataset_50page_schema():
    dataset_path = Path("data/eval_dataset_50page.json")
    assert dataset_path.exists(), "eval_dataset_50page.json does not exist"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 25, f"Expected >=25 test cases, found {len(data)}"
    for item in data:
        assert "id" in item
        assert "input" in item
        assert "expected_output" in item
        assert "context_ground_truth" in item
        assert len(item["context_ground_truth"]) > 0


def test_eval_dataset_bp_schema():
    dataset_path = Path("data/eval_dataset_bp.json")
    assert dataset_path.exists(), "eval_dataset_bp.json does not exist"
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 15, f"Expected >=15 test cases, found {len(data)}"
    for item in data:
        assert "id" in item
        assert "page" in item
        assert "section" in item
        assert item["section"] in ["Early", "Middle", "Late"]
        assert "input" in item
        assert "expected_output" in item
        assert "context_ground_truth" in item
        assert len(item["context_ground_truth"]) > 0
