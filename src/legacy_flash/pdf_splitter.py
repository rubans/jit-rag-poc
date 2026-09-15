from pathlib import Path
from typing import List
import pymupdf


def split_pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 150,
    image_format: str = "jpg"
) -> List[str]:
    """
    Splits a PDF into individual page images rendered at specified DPI (default 150 DPI JPEG).
    Returns list of paths to the saved page images.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    image_paths: List[str] = []
    doc = pymupdf.open(str(pdf_file))
    
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        ext = image_format.lower().lstrip(".")
        img_filename = f"page_{page_idx + 1:03d}.{ext}"
        img_full_path = out_path / img_filename
        pix.save(str(img_full_path))
        image_paths.append(str(img_full_path))
        
    doc.close()
    return image_paths

