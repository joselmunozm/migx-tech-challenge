"""Best-effort Langfuse spans for operational events."""

import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator

from app.config import get_settings

logger = logging.getLogger("migx")


@lru_cache(maxsize=1)
def _client(public_key: str, secret_key: str, host: str) -> Any:
    from langfuse import Langfuse

    return Langfuse(public_key=public_key, secret_key=secret_key, host=host)


def _get_client() -> Any:
    settings = get_settings()
    return _client(
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
        settings.langfuse_host,
    )


@contextmanager
def observation(name: str, metadata: dict[str, Any]) -> Iterator[Any | None]:
    if not get_settings().langfuse_enabled:
        yield None
        return

    try:
        context = _get_client().start_as_current_observation(
            as_type="span",
            name=name,
            metadata=metadata,
        )
        span = context.__enter__()
    except Exception:
        logger.warning("Could not start Langfuse observation %s", name, exc_info=True)
        yield None
        return

    try:
        yield span
    except BaseException as exc:
        update_observation(span, level="ERROR", status_message=type(exc).__name__)
        _finish_observation(context, name, type(exc), exc, exc.__traceback__)
        raise
    else:
        _finish_observation(context, name, None, None, None)


def update_observation(span: Any | None, **values: Any) -> None:
    if span is None:
        return
    try:
        span.update(**values)
    except Exception:
        logger.warning("Could not update Langfuse observation", exc_info=True)


def flush_langfuse() -> None:
    if not get_settings().langfuse_enabled:
        return
    try:
        _get_client().flush()
    except Exception:
        logger.warning("Could not flush Langfuse events", exc_info=True)


def _finish_observation(
    context: Any,
    name: str,
    error_type: type[BaseException] | None,
    error: BaseException | None,
    traceback: Any,
) -> None:
    try:
        context.__exit__(error_type, error, traceback)
    except Exception:
        logger.warning("Could not finish Langfuse observation %s", name, exc_info=True)
