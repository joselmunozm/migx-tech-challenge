from types import SimpleNamespace

import pytest

from app.config import Settings
from app.graph import ask
from app.observability import observation


class FakeSpan:
    def __init__(self) -> None:
        self.updates = []

    def update(self, **values) -> None:
        self.updates.append(values)


class FakeObservation:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span
        self.exit_args = None

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, *args) -> bool:
        self.exit_args = args
        return False


class FakeLangfuse:
    def __init__(self) -> None:
        self.span = FakeSpan()
        self.observation = FakeObservation(self.span)
        self.name = None
        self.metadata = None

    def start_as_current_observation(self, *, name, as_type, metadata):
        self.name = name
        self.metadata = metadata
        assert as_type == "span"
        return self.observation


def test_ask_observation_records_metrics_without_content(monkeypatch) -> None:
    client = FakeLangfuse()
    monkeypatch.setattr("app.observability.get_settings", lambda: SimpleNamespace(langfuse_enabled=True))
    monkeypatch.setattr("app.observability._get_client", lambda: client)

    result = {
        "answer": "Sensitive generated answer",
        "attempts": 1,
        "citations": [{"text": "Sensitive source text"}],
        "citation_check": {"ok": True, "cited": ["S1"], "unknown_refs": [], "abstained": False},
    }

    settings = Settings(openrouter_model="test-model")
    monkeypatch.setattr("app.graph.get_settings", lambda: settings)
    monkeypatch.setattr(
        "app.graph._ask",
        lambda question, document_id, configured: result,
    )

    assert ask("Sensitive user question", "document-123") is result

    assert client.name == "medical-document-ask"
    assert client.metadata == {"prompt_version": "answer-v2", "document_id": "document-123"}
    assert client.span.updates == [
        {
            "output": {
                "model": "test-model",
                "prompt_version": "answer-v2",
                "attempts": 1,
                "source_count": 1,
                "citation_check": result["citation_check"],
            }
        }
    ]
    assert client.observation.exit_args == (None, None, None)
    assert "Sensitive user question" not in repr(client.metadata)
    assert "Sensitive generated answer" not in repr(client.span.updates)
    assert "Sensitive source text" not in repr(client.span.updates)


def test_observation_records_error_type_and_reraises(monkeypatch) -> None:
    client = FakeLangfuse()
    monkeypatch.setattr("app.observability.get_settings", lambda: SimpleNamespace(langfuse_enabled=True))
    monkeypatch.setattr("app.observability._get_client", lambda: client)

    with pytest.raises(RuntimeError, match="sensitive failure"):
        with observation("document-indexing", {"file_type": ".pdf"}) as span:
            assert span is client.span
            raise RuntimeError("sensitive failure")

    assert client.span.updates == [{"level": "ERROR", "status_message": "RuntimeError"}]
    assert client.observation.exit_args[0] is RuntimeError
