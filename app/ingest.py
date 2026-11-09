"""Extract text from an upload and split it into overlapping chunks."""

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from app.errors import AppError

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise AppError(f"Unsupported file type '{suffix or 'unknown'}'. Use one of: {allowed}.")
    if not data.strip():
        raise AppError("The file is empty.")

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
    else:
        text = data.decode("utf-8", errors="replace").strip()

    if not text:
        raise AppError("No text could be extracted from this file.")
    return text


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Keep paragraphs whole when they fit. Repeat a short tail across the boundary."""
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size // 2))

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_long(paragraph, size, overlap))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
            continue
        merged = f"{current}\n\n{piece}"
        if len(merged) <= size:
            current = merged
            continue
        chunks.append(current)
        tail_size = min(overlap, max(0, size - len(piece) - 2), len(current))
        tail = current[-tail_size:].strip() if tail_size else ""
        current = f"{tail}\n\n{piece}".strip() if tail else piece
    if current:
        chunks.append(current)
    return chunks


def _split_long(paragraph: str, size: int, overlap: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    windows: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(paragraph):
        window = paragraph[start : start + size].strip()
        if window:
            windows.append(window)
        start += step
    return windows
