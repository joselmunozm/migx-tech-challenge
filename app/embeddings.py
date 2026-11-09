"""Cohere embed-v4. Documents and queries use different input types, as the model expects."""

from app.config import get_settings
from app.errors import AppError

_BATCH = 96


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        batch = texts[start : start + _BATCH]
        vectors.extend(_embed(batch, "search_document"))
    return vectors


def embed_query(text: str) -> list[float]:
    return _embed([text], "search_query")[0]


def _embed(texts: list[str], input_type: str) -> list[list[float]]:
    import cohere

    settings = get_settings()
    if not settings.cohere_api_key.strip():
        raise AppError("COHERE_API_KEY is not set.", 503)

    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    try:
        response = client.embed(
            model=settings.cohere_embed_model,
            input_type=input_type,
            texts=texts,
            embedding_types=["float"],
            output_dimension=settings.embed_dimension,
        )
    except Exception as exc:
        raise AppError(f"Cohere embeddings failed: {exc}", 502) from exc

    floats = response.embeddings.float
    if not floats or len(floats) != len(texts):
        raise AppError("Cohere returned an unexpected embedding payload.", 502)
    return [list(map(float, row)) for row in floats]
