"""
PDF text extraction using PyMuPDF

Two-column layout strategy:
  page.get_text("blocks") returns text blocks with bounding box coordinates
  (x0, y0, x1, y1). Blocks are sorted by column (left before right) and then
  top-to-bottom within each column, preserving the natural reading order.
"""

from pathlib import Path

import pymupdf

from .config import MAX_CHARS, MIN_CHARS_THRESHOLD


def extract_text(pdf_path: Path) -> tuple[str, str]:
    """
    Extract text from a PDF file.

    Returns:
        (text, status)
        status: "ok" | "pdf_sem_texto" | "erro_leitura:<reason>"
    """
    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        return "", f"erro_leitura: {exc}"

    full_text_parts: list[str] = []

    for page in doc:
        page_text = _extract_page_text(page)
        full_text_parts.append(page_text)

    doc.close()

    full_text = "\n".join(full_text_parts).strip()

    if len(full_text) < MIN_CHARS_THRESHOLD:
        return full_text, "pdf_sem_texto"

    # Truncate to stay within the model's context window
    if len(full_text) > MAX_CHARS:
        full_text = full_text[:MAX_CHARS]

    return full_text, "ok"


def _extract_page_text(page: pymupdf.Page) -> str:
    """
    Extract text from a single page while respecting multi-column layouts.
    Uses block coordinates to sort reading order correctly.
    """

    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

    # Keep only text blocks (block_type == 0) with non-empty content
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

    if not text_blocks:
        return ""

    # Assign each block to the left or right column based on page midpoint
    page_width = page.rect.width
    mid_x = page_width / 2

    def sort_key(b):
        x0, y0 = b[0], b[1]
        col = 0 if x0 < mid_x else 1
        return (col, y0)

    text_blocks.sort(key=sort_key)

    return "\n".join(b[4].strip() for b in text_blocks)
