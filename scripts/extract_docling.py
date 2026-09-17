"""
scripts/extract_docling.py
Extracts structured Markdown from PDF using Docling's TableFormer and layout model in isolated environment.
"""

import sys
import json
import time
import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat


def extract_pdf_with_docling(pdf_path: str, output_dir: str) -> dict:
    pdf_path = Path(pdf_path).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[Docling] Initializing DocumentConverter (do_ocr=False, do_table_structure=True)...")
    t0 = time.perf_counter()
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False  # Digital text PDF; TableFormer runs on layout
    pipeline_options.do_table_structure = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    init_time = time.perf_counter() - t0
    print(f"[Docling] Initialized in {init_time:.2f}s")
    
    print(f"[Docling] Converting {pdf_path.name}...")
    t_conv_start = time.perf_counter()
    conv_res = converter.convert(str(pdf_path))
    parse_time = time.perf_counter() - t_conv_start
    print(f"[Docling] Converted in {parse_time:.2f}s")
    
    # Export full document markdown
    full_md = conv_res.document.export_to_markdown()
    full_md_path = out_path / "document.md"
    with open(full_md_path, "w", encoding="utf-8") as f:
        f.write(full_md)
        
    metadata = {
        "pdf_path": str(pdf_path),
        "output_dir": str(out_path),
        "init_time_s": init_time,
        "parse_time_s": parse_time,
        "markdown_path": str(full_md_path),
        "markdown_char_count": len(full_md),
    }
    
    meta_path = out_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[Docling] Saved extracted markdown to {full_md_path} ({len(full_md)} chars)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PDF to Markdown with Docling")
    parser.add_argument("--pdf", default="data/documents/sample_50page.pdf", help="Input PDF path")
    parser.add_argument("--output-dir", default="output/docling_extracted", help="Output directory")
    args = parser.parse_args()
    
    meta = extract_pdf_with_docling(args.pdf, args.output_dir)
    print(json.dumps(meta, indent=2))

