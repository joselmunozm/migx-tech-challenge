from types import SimpleNamespace

from app.config import Settings
from app.graph import _ask, build_graph
from app.grounding import check_citations
from app.prompts import (
    ABSTAIN_RESPONSE,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    render_repair_prompt,
)


def test_prompt_allows_grounded_partial_answers_without_guessing() -> None:
    assert PROMPT_VERSION == "answer-v2"
    assert "answer that part" in SYSTEM_PROMPT
    assert "none of the excerpts contain information relevant" in SYSTEM_PROMPT
    assert "Do not use unrelated excerpts to guess." in SYSTEM_PROMPT
    repair_prompt = render_repair_prompt(
        "What was measured and when?",
        [{"ref": "S1", "filename": "note.txt", "chunk_id": "doc:0", "text": "Marker X was measured."}],
        ABSTAIN_RESPONSE,
        {"unknown_refs": [], "cited": [], "abstained": True},
    )
    assert "even if it only answers part" in repair_prompt
    assert "If no excerpt is relevant" in repair_prompt


def test_citation_check_accepts_known_reference() -> None:
    assert check_citations("The endpoint was measured at week 8 [S1].", {"S1"}) == {
        "ok": True,
        "cited": ["S1"],
        "unknown_refs": [],
        "abstained": False,
    }


def test_citation_check_rejects_unknown_or_missing_references() -> None:
    unknown = check_citations("The endpoint was measured at week 8 [S2].", {"S1"})
    missing = check_citations("The endpoint was measured at week 8.", {"S1"})

    assert unknown["ok"] is False
    assert unknown["unknown_refs"] == ["S2"]
    assert missing["ok"] is False
    assert missing["cited"] == []


def test_citation_check_accepts_abstention() -> None:
    result = check_citations(ABSTAIN_RESPONSE, set())

    assert result["ok"] is True
    assert result["abstained"] is True


class FakeModel:
    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)
        self.calls = 0

    def invoke(self, messages: list[object]) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(content=next(self.answers))


def source_hit() -> SimpleNamespace:
    return SimpleNamespace(
        chunk_id="doc-1:0",
        document_id="doc-1",
        filename="synthetic.txt",
        chunk_index=0,
        score=0.9,
        text="The primary endpoint was change in biomarker X at week 8.",
    )


def initial_state() -> dict:
    return {
        "question": "What was the endpoint?",
        "document_id": "doc-1",
        "sources": [],
        "answer": "",
        "citation_check": {},
        "attempts": 0,
    }


def test_graph_retries_once_after_unknown_citation() -> None:
    model = FakeModel(["It was biomarker X [S9].", "It was biomarker X [S1]."])
    graph = build_graph(lambda question, document_id: [source_hit()], model)

    result = graph.invoke(initial_state())

    assert model.calls == 2
    assert result["attempts"] == 2
    assert result["citation_check"]["ok"] is True


def test_graph_does_not_retry_valid_citation() -> None:
    model = FakeModel(["It was biomarker X [S1]."])
    graph = build_graph(lambda question, document_id: [source_hit()], model)

    result = graph.invoke(initial_state())

    assert model.calls == 1
    assert result["attempts"] == 1
    assert result["citation_check"]["ok"] is True


def test_graph_abstains_without_calling_model_when_no_sources() -> None:
    model = FakeModel([])
    graph = build_graph(lambda question, document_id: [], model)

    result = graph.invoke(initial_state())

    assert model.calls == 0
    assert result["answer"] == ABSTAIN_RESPONSE
    assert result["citation_check"]["ok"] is True


def test_graph_abstains_without_sources_below_similarity_threshold() -> None:
    model = FakeModel([])
    low_similarity_hit = source_hit()
    low_similarity_hit.score = 0.39
    graph = build_graph(
        lambda question, document_id: [low_similarity_hit],
        model,
        min_similarity=0.4,
    )

    result = graph.invoke(initial_state())

    assert model.calls == 0
    assert result["sources"] == []
    assert result["answer"] == ABSTAIN_RESPONSE


def test_graph_uses_selected_similarity_threshold_by_default() -> None:
    model = FakeModel([])
    low_similarity_hit = source_hit()
    low_similarity_hit.score = 0.29
    graph = build_graph(lambda question, document_id: [low_similarity_hit], model)

    result = graph.invoke(initial_state())

    assert model.calls == 0
    assert result["sources"] == []
    assert result["answer"] == ABSTAIN_RESPONSE


def test_ask_retrieves_document_context_through_mcp(monkeypatch) -> None:
    calls = []

    def mcp_search(query, document_id, top_k):
        calls.append((query, document_id, top_k))
        return [source_hit()]

    settings = Settings(
        cohere_api_key="test-key",
        openrouter_api_key="test-key",
        top_k=5,
    )
    monkeypatch.setattr("app.mcp_client.search_documents", mcp_search)
    monkeypatch.setattr("app.graph._build_chat_model", lambda _: FakeModel(["Fact [S1]."]))

    result = _ask("What was the endpoint?", "doc-1", settings)

    assert calls == [("What was the endpoint?", "doc-1", 5)]
    assert result["citations"][0]["text"] == source_hit().text