# Retrieval experiments

These experiments embed questions and document text with the configured Cohere
embedding model, then score locally with cosine similarity. They do not call
OpenRouter, modify Chroma, or touch uploaded documents.

## Synthetic note

The small benchmark uses only the fictional note in `data/samples/` and the
questions in `questions.json`.

```bash
uv run python -m scripts.evaluate_retrieval
```

It compares chunk sizes/overlap `(300, 50)`, `(500, 75)`, and `(800, 100)`,
`top_k` values 1, 2, and 3, and thresholds 0.3, 0.4, and 0.5. It writes
`evaluation/results/latest.json`.

Metrics:

- **Evidence recall**: answerable questions whose exact expected passage occurs
  in at least one retrieved chunk.
- **Evidence precision**: retrieved chunks containing that expected passage,
  divided by all retrieved chunks for answerable questions.
- **Negative rejection**: unanswerable questions for which thresholding leaves
  no chunks.
- **MRR**: reciprocal rank of the first retrieved chunk containing the exact
  expected passage, averaged over answerable questions.

## BioASQ subset

The BioASQ experiment uses the `rag-datasets/rag-mini-bioasq` mirror of BioASQ
Task B 11b training data. The question records contain gold PubMed passage IDs;
the passage records are the candidate retrieval corpus. The data is attributed
to BioASQ and the U.S. National Library of Medicine (NLM), and is distributed
under CC BY 2.5. See the [dataset card](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq)
and [BioASQ](https://bioasq.org/) for source details.

Keep the downloaded files local; they are excluded from version control. Place
the mirror's `questions.parquet` and `passages.parquet` files at:

```text
evaluation/data/bioasq/questions.parquet
evaluation/data/bioasq/passages.parquet
```

Run the reproducible default subset:

```bash
uv run python -m scripts.evaluate_retrieval --dataset bioasq
```

By default, the script selects 20 questions with seed 17 and includes every
gold passage for those questions plus random distractors to make a 450-passage
candidate set. This bounds the Cohere embedding workload; it is not a ranking
over the entire 40,221-passage corpus. Override the bounds or seed with
`--question-count`, `--candidate-passages`, and `--seed`. All gold passages for
the selected questions remain in the candidate set; if the gold set exceeds
the candidate limit, the script reports an error instead of silently dropping
relevant passages. Embedding requests are sent in batches.
The runner estimates input tokens from text length and pauses between batches
to stay below Cohere's trial token-per-minute limit.

The report is written to `evaluation/results/latest-bioasq.json`. For each
configuration it reports:

- **Raw article recall@k**: mean per-question fraction of gold PubMed IDs in the
  top-k passages before thresholding.
- **Article recall@k**: the same measure after applying the similarity
  threshold.
- **Article MRR**: mean reciprocal rank of the first gold passage after
  thresholding.
- **Mean retrieved articles**: average number of top-k passages retained by the
  threshold.

Passages are deduplicated after chunk ranking by their best-scoring chunk, so
overlap does not count the same PubMed record multiple times. The report
includes the question count, candidate count, gold passage count, embedding
model, and sampling parameters for reproducibility.

Both experiments are exploratory benchmarks, not statistically representative
quality claims. Review results before selecting settings; tune against more
questions and the full corpus before relying on a threshold.
