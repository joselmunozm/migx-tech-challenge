"""Run retrieval-only benchmarks on the synthetic note or BioASQ."""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cohere
import numpy as np
import polars as pl

from app.config import get_settings
from app.ingest import chunk_text

ROOT = Path(__file__).resolve().parents[1]
CHUNK_CONFIGS = [(300, 50), (500, 75), (800, 100)]
TOP_K_VALUES = (1, 2, 3)
THRESHOLDS = (0.3, 0.4, 0.5)
BIOASQ_DIR = ROOT / "evaluation" / "data" / "bioasq"
BIOASQ_QUESTION_COUNT = 20
BIOASQ_CANDIDATE_COUNT = 450
BIOASQ_SEED = 17
EMBEDDING_TOKEN_BUDGET_PER_MINUTE = 70_000


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match.")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if not left_norm or not right_norm:
        raise ValueError("Cannot calculate cosine similarity for a zero vector.")
    return math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _metrics(
    questions: list[dict[str, Any]],
    ranked_by_question: dict[str, list[dict[str, Any]]],
    top_k: int,
    threshold: float,
) -> dict[str, Any]:
    answerable = [question for question in questions if question["answerable"]]
    unanswerable = [question for question in questions if not question["answerable"]]
    selected_by_question = {
        question["id"]: [
            hit
            for hit in ranked_by_question[question["id"]][:top_k]
            if hit["score"] >= threshold
        ]
        for question in questions
    }

    found_ranks: list[int] = []
    relevant_count = 0
    answerable_retrieved_count = 0
    for question in answerable:
        selected = selected_by_question[question["id"]]
        answerable_retrieved_count += len(selected)
        ranks = [
            rank
            for rank, hit in enumerate(selected, start=1)
            if hit["contains_expected_evidence"]
        ]
        relevant_count += len(ranks)
        if ranks:
            found_ranks.append(ranks[0])

    negative_rejections = sum(
        not selected_by_question[question["id"]] for question in unanswerable
    )
    answerable_count = len(answerable)
    unanswerable_count = len(unanswerable)
    return {
        "answerable_evidence_recall": len(found_ranks) / answerable_count,
        "answerable_evidence_precision": (
            relevant_count / answerable_retrieved_count
            if answerable_retrieved_count
            else 0.0
        ),
        "unanswerable_rejection_rate": (
            negative_rejections / unanswerable_count if unanswerable_count else 0.0
        ),
        "answerable_mrr": (
            math.fsum(1 / rank for rank in found_ranks) / answerable_count
        ),
        "mean_retrieved_chunks": math.fsum(
            len(selected_by_question[question["id"]]) for question in questions
        )
        / len(questions),
    }


def run_evaluation(
    note: str,
    questions: list[dict[str, Any]],
    embed_documents,
    embed_queries,
) -> dict[str, Any]:
    query_vectors = embed_queries([question["question"] for question in questions])
    if len(query_vectors) != len(questions):
        raise ValueError("Query embedding count does not match the question set.")
    configurations = []
    for size, overlap in CHUNK_CONFIGS:
        chunks = chunk_text(note, size=size, overlap=overlap)
        document_vectors = embed_documents(chunks)
        if len(document_vectors) != len(chunks):
            raise ValueError("Document embedding count does not match the chunk count.")
        ranked_by_question: dict[str, list[dict[str, Any]]] = {}
        for question, query_vector in zip(questions, query_vectors):
            ranked = [
                {
                    "chunk_index": index,
                    "score": round(_cosine(query_vector, vector), 4),
                    "contains_expected_evidence": (
                        bool(question["expected_evidence"])
                        and question["expected_evidence"] in chunks[index]
                    ),
                }
                for index, vector in enumerate(document_vectors)
            ]
            ranked_by_question[question["id"]] = sorted(
                ranked,
                key=lambda hit: hit["score"],
                reverse=True,
            )

        for top_k in TOP_K_VALUES:
            for threshold in THRESHOLDS:
                configurations.append(
                    {
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                        "top_k": top_k,
                        "threshold": threshold,
                        "chunk_count": len(chunks),
                        "metrics": _metrics(
                            questions,
                            ranked_by_question,
                            top_k,
                            threshold,
                        ),
                        "queries": [
                            _query_result(question, ranked_by_question[question["id"]], top_k, threshold)
                            for question in questions
                        ],
                    }
                )
    return {
        "note": "data/samples/synthetic_mig217_note.txt",
        "question_count": len(questions),
        "embedding_model": get_settings().cohere_embed_model,
        "embedding_dimension": get_settings().embed_dimension,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configurations": configurations,
    }


def load_bioasq_subset(
    questions_path: Path,
    passages_path: Path,
    question_count: int = BIOASQ_QUESTION_COUNT,
    candidate_count: int = BIOASQ_CANDIDATE_COUNT,
    seed: int = BIOASQ_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Load a reproducible question subset with all gold passages and random distractors."""
    if question_count <= 0 or candidate_count <= 0:
        raise ValueError("Question and candidate counts must be positive.")

    question_rows = pl.read_parquet(
        questions_path,
        columns=["id", "question", "relevant_passage_ids"],
    ).to_dicts()
    passage_rows = pl.read_parquet(
        passages_path,
        columns=["id", "passage"],
    ).to_dicts()

    questions: list[dict[str, Any]] = []
    for row in question_rows:
        try:
            relevant_ids = json.loads(row["relevant_passage_ids"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid BioASQ relevance IDs for question {row['id']}.") from error
        if not isinstance(relevant_ids, list) or not relevant_ids:
            continue
        questions.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "relevant_passage_ids": {str(passage_id) for passage_id in relevant_ids},
            }
        )

    passage_by_id = {str(row["id"]): row["passage"] for row in passage_rows}
    if len(passage_by_id) != len(passage_rows):
        raise ValueError("BioASQ passage IDs must be unique.")
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("BioASQ question IDs must be unique.")

    selected_questions = random.Random(seed).sample(
        questions,
        min(question_count, len(questions)),
    )
    if not selected_questions:
        raise ValueError("The BioASQ question file contains no usable questions.")
    gold_ids = set().union(
        *(question["relevant_passage_ids"] for question in selected_questions)
    )
    missing_ids = gold_ids.difference(passage_by_id)
    if missing_ids:
        raise ValueError(f"BioASQ corpus is missing {len(missing_ids)} gold passages.")
    if len(gold_ids) > candidate_count:
        raise ValueError(
            f"The selected questions require {len(gold_ids)} gold passages, "
            f"more than the candidate limit of {candidate_count}."
        )

    negative_ids = sorted(set(passage_by_id).difference(gold_ids))
    negative_count = min(candidate_count - len(gold_ids), len(negative_ids))
    sampled_ids = gold_ids.union(random.Random(seed).sample(negative_ids, negative_count))
    candidates = [
        {"id": passage_id, "text": passage_by_id[passage_id]}
        for passage_id in sorted(sampled_ids)
    ]
    return selected_questions, candidates


def _bioasq_metrics(
    questions: list[dict[str, Any]],
    ranked_by_question: dict[str, list[dict[str, Any]]],
    top_k: int,
    threshold: float,
) -> dict[str, float]:
    raw_recalls: list[float] = []
    accepted_recalls: list[float] = []
    raw_reciprocal_ranks: list[float] = []
    accepted_reciprocal_ranks: list[float] = []
    retrieved_count = 0

    for question in questions:
        relevant_ids = question["relevant_passage_ids"]
        ranked = ranked_by_question[question["id"]][:top_k]
        accepted = [hit for hit in ranked if hit["score"] >= threshold]
        raw_relevant_ranks = [
            rank
            for rank, hit in enumerate(ranked, start=1)
            if hit["passage_id"] in relevant_ids
        ]
        accepted_relevant_ranks = [
            rank
            for rank, hit in enumerate(accepted, start=1)
            if hit["passage_id"] in relevant_ids
        ]
        raw_recalls.append(len(raw_relevant_ranks) / len(relevant_ids))
        accepted_recalls.append(len(accepted_relevant_ranks) / len(relevant_ids))
        raw_reciprocal_ranks.append(1 / raw_relevant_ranks[0] if raw_relevant_ranks else 0.0)
        accepted_reciprocal_ranks.append(
            1 / accepted_relevant_ranks[0] if accepted_relevant_ranks else 0.0
        )
        retrieved_count += len(accepted)

    question_count = len(questions)
    return {
        "raw_article_recall_at_k": math.fsum(raw_recalls) / question_count,
        "article_recall_at_k": math.fsum(accepted_recalls) / question_count,
        "raw_article_mrr": math.fsum(raw_reciprocal_ranks) / question_count,
        "article_mrr": math.fsum(accepted_reciprocal_ranks) / question_count,
        "mean_retrieved_articles": retrieved_count / question_count,
    }


def run_bioasq_evaluation(
    passages: list[dict[str, str]],
    questions: list[dict[str, Any]],
    embed_documents,
    embed_queries,
) -> dict[str, Any]:
    if not passages or not questions:
        raise ValueError("BioASQ evaluation requires at least one passage and question.")
    query_vectors = embed_queries([question["question"] for question in questions])
    if len(query_vectors) != len(questions):
        raise ValueError("Query embedding count does not match the question set.")

    configurations = []
    for size, overlap in CHUNK_CONFIGS:
        chunks: list[str] = []
        chunk_passage_ids: list[str] = []
        for passage in passages:
            passage_chunks = chunk_text(passage["text"], size=size, overlap=overlap)
            chunks.extend(passage_chunks)
            chunk_passage_ids.extend([passage["id"]] * len(passage_chunks))
        document_vectors = embed_documents(chunks)
        if len(document_vectors) != len(chunks):
            raise ValueError("Document embedding count does not match the chunk count.")

        document_matrix = np.asarray(document_vectors, dtype=np.float32)
        if document_matrix.ndim != 2:
            raise ValueError("Document embeddings must be non-empty vectors.")
        document_norms = np.linalg.norm(document_matrix, axis=1)
        if np.any(document_norms == 0):
            raise ValueError("Cannot calculate cosine similarity for a zero vector.")
        normalized_documents = document_matrix / document_norms[:, np.newaxis]
        passage_ids = sorted(set(chunk_passage_ids))
        passage_index = {passage_id: index for index, passage_id in enumerate(passage_ids)}
        chunk_passage_indices = np.fromiter(
            (passage_index[passage_id] for passage_id in chunk_passage_ids),
            dtype=np.intp,
            count=len(chunk_passage_ids),
        )

        ranked_by_question: dict[str, list[dict[str, Any]]] = {}
        query_matrix = np.asarray(query_vectors, dtype=np.float32)
        if query_matrix.ndim != 2 or query_matrix.shape[1] != document_matrix.shape[1]:
            raise ValueError("Query and document embedding dimensions do not match.")
        query_norms = np.linalg.norm(query_matrix, axis=1)
        if np.any(query_norms == 0):
            raise ValueError("Cannot calculate cosine similarity for a zero vector.")
        query_matrix = query_matrix / query_norms[:, np.newaxis]
        score_matrix = normalized_documents @ query_matrix.T
        for question_index, question in enumerate(questions):
            best_scores = np.full(len(passage_ids), float("-inf"), dtype=np.float32)
            np.maximum.at(
                best_scores,
                chunk_passage_indices,
                score_matrix[:, question_index],
            )
            ranked_by_question[question["id"]] = [
                {"passage_id": passage_id, "score": score}
                for passage_id, score in sorted(
                    zip(passage_ids, best_scores.tolist()),
                    key=lambda item: (-item[1], item[0]),
                )
            ]

        for top_k in TOP_K_VALUES:
            for threshold in THRESHOLDS:
                configurations.append(
                    {
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                        "top_k": top_k,
                        "threshold": threshold,
                        "chunk_count": len(chunks),
                        "metrics": _bioasq_metrics(
                            questions,
                            ranked_by_question,
                            top_k,
                            threshold,
                        ),
                    }
                )

    return {
        "dataset": "BioASQ Task B 11b training mirror",
        "question_count": len(questions),
        "candidate_passage_count": len(passages),
        "gold_passage_count": len(
            set().union(*(question["relevant_passage_ids"] for question in questions))
        ),
        "embedding_model": get_settings().cohere_embed_model,
        "embedding_dimension": get_settings().embed_dimension,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configurations": configurations,
    }


def _query_result(
    question: dict[str, Any],
    ranked: list[dict[str, Any]],
    top_k: int,
    threshold: float,
) -> dict[str, Any]:
    top_hits = ranked[:top_k]
    retrieved = [hit for hit in top_hits if hit["score"] >= threshold]
    return {
        "id": question["id"],
        "raw_evidence_rank_at_k": next(
            (
                rank
                for rank, hit in enumerate(top_hits, start=1)
                if hit["contains_expected_evidence"]
            ),
            None,
        ),
        "accepted_evidence_rank": next(
            (
                rank
                for rank, hit in enumerate(retrieved, start=1)
                if hit["contains_expected_evidence"]
            ),
            None,
        ),
        "expected_evidence_score": next(
            (hit["score"] for hit in top_hits if hit["contains_expected_evidence"]),
            None,
        ),
        "retrieved_chunk_indices": [hit["chunk_index"] for hit in retrieved],
    }


def _print_summary(report: dict[str, Any]) -> None:
    bioasq = "candidate_passage_count" in report
    if bioasq:
        print("size overlap k threshold | raw article recall | thresholded recall | MRR")
    else:
        print(
            "size overlap k threshold | evidence recall | evidence precision | "
            "negative rejection | MRR"
        )
    for result in report["configurations"]:
        metrics = result["metrics"]
        prefix = (
            f'{result["chunk_size"]:>4} {result["chunk_overlap"]:>7} '
            f'{result["top_k"]:>1} {result["threshold"]:>9.1f} | '
        )
        if bioasq:
            print(
                f'{prefix}{metrics["raw_article_recall_at_k"]:>17.1%} | '
                f'{metrics["article_recall_at_k"]:>17.1%} | '
                f'{metrics["article_mrr"]:.3f}'
            )
        else:
            print(
                f'{prefix}{metrics["answerable_evidence_recall"]:>14.1%} | '
                f'{metrics["answerable_evidence_precision"]:>18.1%} | '
                f'{metrics["unanswerable_rejection_rate"]:>18.1%} | '
                f'{metrics["answerable_mrr"]:.3f}'
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "bioasq"),
        default="synthetic",
        help="Dataset used for evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path for the JSON report (defaults to the dataset-specific report).",
    )
    parser.add_argument("--question-count", type=int, default=BIOASQ_QUESTION_COUNT)
    parser.add_argument("--candidate-passages", type=int, default=BIOASQ_CANDIDATE_COUNT)
    parser.add_argument("--seed", type=int, default=BIOASQ_SEED)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.cohere_api_key.strip():
        parser.error("COHERE_API_KEY is required to run the embedding experiment.")
    client = cohere.ClientV2(
        api_key=settings.cohere_api_key,
        timeout=45,
        max_retries=0,
    )
    embedding_window_started = time.monotonic()
    estimated_tokens_in_window = 0

    def embed(texts: list[str], input_type: str) -> list[list[float]]:
        nonlocal embedding_window_started, estimated_tokens_in_window
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 90):
            batch = texts[start : start + 90]
            estimated_tokens = max(1, math.ceil(sum(map(len, batch)) / 3))
            if estimated_tokens_in_window + estimated_tokens > EMBEDDING_TOKEN_BUDGET_PER_MINUTE:
                wait_seconds = 60 - (time.monotonic() - embedding_window_started)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                embedding_window_started = time.monotonic()
                estimated_tokens_in_window = 0
            response = client.embed(
                model=settings.cohere_embed_model,
                input_type=input_type,
                texts=batch,
                embedding_types=["float"],
                output_dimension=settings.embed_dimension,
            )
            batch_vectors = response.embeddings.float
            if batch_vectors is None or len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Cohere returned an invalid {input_type} embedding response."
                )
            vectors.extend([list(map(float, vector)) for vector in batch_vectors])
            estimated_tokens_in_window += estimated_tokens
        return vectors

    if args.dataset == "bioasq":
        questions, passages = load_bioasq_subset(
            BIOASQ_DIR / "questions.parquet",
            BIOASQ_DIR / "passages.parquet",
            question_count=args.question_count,
            candidate_count=args.candidate_passages,
            seed=args.seed,
        )
        report = run_bioasq_evaluation(
            passages,
            questions,
            lambda texts: embed(texts, "search_document"),
            lambda texts: embed(texts, "search_query"),
        )
        report["sampling"] = {
            "seed": args.seed,
            "question_count_requested": args.question_count,
            "candidate_passage_count_requested": args.candidate_passages,
            "negative_passages_sampled": len(passages) - report["gold_passage_count"],
        }
        output = args.output or ROOT / "evaluation" / "results" / "latest-bioasq.json"
    else:
        note = (ROOT / "data" / "samples" / "synthetic_mig217_note.txt").read_text(
            encoding="utf-8"
        )
        questions = json.loads(
            (ROOT / "evaluation" / "questions.json").read_text(encoding="utf-8")
        )
        report = run_evaluation(
            note,
            questions,
            lambda texts: embed(texts, "search_document"),
            lambda texts: embed(texts, "search_query"),
        )
        output = args.output or ROOT / "evaluation" / "results" / "latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_summary(report)
    print(f"\nFull report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
