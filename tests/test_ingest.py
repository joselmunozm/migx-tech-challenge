from app.ingest import chunk_text


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("  \n\n ") == []


def test_long_paragraph_chunks_never_exceed_size() -> None:
    text = "0123456789" * 100
    chunks = chunk_text(text, size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[0][-20:] == chunks[1][:20]


def test_paragraph_boundary_uses_only_available_overlap_space() -> None:
    chunks = chunk_text(f"{'a' * 70}\n\n{'b' * 70}", size=100, overlap=20)

    assert len(chunks) == 2
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert chunks[1].startswith(f"{'a' * 20}\n\n")