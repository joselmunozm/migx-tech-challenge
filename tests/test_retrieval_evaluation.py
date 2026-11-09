import polars as pl

from scripts.evaluate_retrieval import (
    _bioasq_metrics,
    _metrics,
    load_bioasq_subset,
    run_bioasq_evaluation,
    run_evaluation,
)


def test_metrics_measure_evidence_and_unanswerable_rejection() -> None:
    questions = [
        {"id": "answerable", "answerable": True, "expected_evidence": "fact"},
        {"id": "unanswerable", "answerable": False, "expected_evidence": None},
    ]
    ranked = {
        "answerable": [
            {"score": 0.8, "contains_expected_evidence": True},
            {"score": 0.7, "contains_expected_evidence": False},
        ],
        "unanswerable": [{"score": 0.3, "contains_expected_evidence": False}],
    }

    result = _metrics(questions, ranked, top_k=2, threshold=0.5)

    assert result == {
        "answerable_evidence_recall": 1.0,
        "answerable_evidence_precision": 0.5,
        "unanswerable_rejection_rate": 1.0,
        "answerable_mrr": 1.0,
        "mean_retrieved_chunks": 1.0,
    }


def test_run_evaluation_reuses_chunk_text_and_scores_exact_evidence() -> None:
    questions = [
        {"id": "known", "question": "Where is the fact?", "expected_evidence": "fact", "answerable": True},
        {"id": "unknown", "question": "Unanswerable?", "expected_evidence": None, "answerable": False},
    ]
    document_calls = []
    query_calls = []

    def embed_documents(chunks):
        document_calls.append(chunks)
        return [[1.0, 0.0] if "fact" in chunk else [0.0, 1.0] for chunk in chunks]

    def embed_queries(queries):
        query_calls.extend(queries)
        return [[1.0, 0.0], [0.0, 1.0]]

    report = run_evaluation("fact\n\nother", questions, embed_documents, embed_queries)

    assert len(report["configurations"]) == 27
    assert len(document_calls) == 3
    assert query_calls == ["Where is the fact?", "Unanswerable?"]
    first = report["configurations"][0]
    assert first["metrics"]["answerable_evidence_recall"] == 1.0
    assert first["queries"][0]["raw_evidence_rank_at_k"] == 1


def test_bioasq_metrics_score_relevant_articles_before_and_after_threshold() -> None:
    questions = [
        {"id": "q1", "relevant_passage_ids": {"p1", "p2"}},
    ]
    ranked = {
        "q1": [
            {"passage_id": "p1", "score": 0.8},
            {"passage_id": "p3", "score": 0.6},
            {"passage_id": "p2", "score": 0.4},
        ],
    }

    metrics = _bioasq_metrics(questions, ranked, top_k=3, threshold=0.5)

    assert metrics == {
        "raw_article_recall_at_k": 1.0,
        "article_recall_at_k": 0.5,
        "raw_article_mrr": 1.0,
        "article_mrr": 1.0,
        "mean_retrieved_articles": 2.0,
    }


def test_load_bioasq_subset_keeps_all_selected_gold_passages(tmp_path) -> None:
    questions_path = tmp_path / "questions.parquet"
    passages_path = tmp_path / "passages.parquet"
    pl.DataFrame(
        {
            "id": [0, 1, 2],
            "question": ["q0", "q1", "q2"],
            "relevant_passage_ids": ['[10, 11]', '[12]', '[13]'],
        }
    ).write_parquet(questions_path)
    pl.DataFrame(
        {
            "id": [10, 11, 12, 13, 14, 15],
            "passage": ["p10", "p11", "p12", "p13", "p14", "p15"],
        }
    ).write_parquet(passages_path)

    selected, candidates = load_bioasq_subset(
        questions_path,
        passages_path,
        question_count=2,
        candidate_count=4,
        seed=17,
    )
    selected_again, candidates_again = load_bioasq_subset(
        questions_path,
        passages_path,
        question_count=2,
        candidate_count=4,
        seed=17,
    )

    candidate_ids = {passage["id"] for passage in candidates}
    gold_ids = set().union(*(question["relevant_passage_ids"] for question in selected))
    assert [question["id"] for question in selected] == [
        question["id"] for question in selected_again
    ]
    assert candidates == candidates_again
    assert len(candidates) == 4
    assert gold_ids <= candidate_ids


def test_run_bioasq_evaluation_ranks_passages_by_best_chunk_score() -> None:
    passages = [
        {"id": "p1", "text": "gold evidence"},
        {"id": "p2", "text": "unrelated text"},
    ]
    questions = [
        {"id": "q1", "question": "question", "relevant_passage_ids": {"p1"}},
    ]

    report = run_bioasq_evaluation(
        passages,
        questions,
        lambda chunks: [[1.0, 0.0] if "gold" in chunk else [0.0, 1.0] for chunk in chunks],
        lambda _: [[1.0, 0.0]],
    )

    assert report["candidate_passage_count"] == 2
    assert len(report["configurations"]) == 27
    first = report["configurations"][0]
    assert first["metrics"]["raw_article_recall_at_k"] == 1.0
    assert first["metrics"]["article_recall_at_k"] == 1.0
