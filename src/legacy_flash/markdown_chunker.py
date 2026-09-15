import re
from typing import List
from pathlib import Path
from src.common.models import DocumentChunk


class MarkdownChunker:
    """
    Chunks Markdown documents while respecting structural boundaries
    such as section headers (#, ##, ###) and keeping table blocks intact.
    """
    def __init__(self, target_chunk_size: int = 1000, overlap: int = 150):
        self.target_chunk_size = target_chunk_size
        self.overlap = overlap

    def chunk_markdown_file(self, md_path: str, doc_id: str = "sample_50page") -> List[DocumentChunk]:
        path = Path(md_path)
        if not path.exists():
            return []
            
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        page_num = None
        # Try to infer page number from filename e.g. page_003.md
        match = re.search(r"page_(\d+)", path.stem)
        if match:
            page_num = int(match.group(1))

        return self.chunk_text(content, doc_id=doc_id, page_number=page_num)

    def chunk_text(self, text: str, doc_id: str, page_number: int = None) -> List[DocumentChunk]:
        if not text.strip():
            return []

        # Split by section headers while preserving them
        sections = re.split(r"(?m)(?=^#{1,3}\s+)", text)
        chunks: List[DocumentChunk] = []
        chunk_idx = 1

        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            # If section fits within target chunk size, keep it as a single chunk
            if len(sec_clean) <= self.target_chunk_size:
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}-p{page_number or 0}-c{chunk_idx}",
                    doc_id=doc_id,
                    page_number=page_number,
                    content=sec_clean,
                    metadata={"char_length": len(sec_clean)}
                ))
                chunk_idx += 1
            else:
                # Sub-chunk larger sections without breaking table lines
                lines = sec_clean.split("\n")
                current_chunk_lines = []
                current_length = 0

                for line in lines:
                    line_len = len(line) + 1
                    if current_length + line_len > self.target_chunk_size and current_chunk_lines:
                        chunk_text = "\n".join(current_chunk_lines).strip()
                        chunks.append(DocumentChunk(
                            chunk_id=f"{doc_id}-p{page_number or 0}-c{chunk_idx}",
                            doc_id=doc_id,
                            page_number=page_number,
                            content=chunk_text,
                            metadata={"char_length": len(chunk_text)}
                        ))
                        chunk_idx += 1
                        current_chunk_lines = []
                        current_length = 0

                    current_chunk_lines.append(line)
                    current_length += line_len

                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    chunks.append(DocumentChunk(
                        chunk_id=f"{doc_id}-p{page_number or 0}-c{chunk_idx}",
                        doc_id=doc_id,
                        page_number=page_number,
                        content=chunk_text,
                        metadata={"char_length": len(chunk_text)}
                    ))
                    chunk_idx += 1

        return chunks

