from app.config import Settings


def test_retrieval_defaults_match_selected_bioasq_configuration(monkeypatch) -> None:
    for name in ("CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "MIN_SIMILARITY"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 300
    assert settings.chunk_overlap == 50
    assert settings.top_k == 3
    assert settings.min_similarity == 0.3
