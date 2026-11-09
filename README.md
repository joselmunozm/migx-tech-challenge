# Medical Document Assistant

A local prototype for asking questions about uploaded medical or clinical documents. It extracts PDF, TXT, and Markdown files, stores searchable chunks locally, and answers questions with references to retrieved excerpts.

Do not upload real patient records or other sensitive health information. Document excerpts are sent to Cohere for embeddings and to OpenRouter for answer generation. If Langfuse keys are configured, operational trace metadata is sent to the configured Langfuse host; questions, answers, filenames, and document text are not. Use synthetic or appropriately de-identified material only.

## Architecture

- **Streamlit** provides the user interface and communicates only with the HTTP API.
- **FastAPI** handles uploads, document listing, and questions. All routes except `/health` require `X-API-Key`.
- **Chroma** persists vectors locally under `data/chroma`; document metadata is recorded in `data/registry.json`.
- **Cohere `embed-v4.0`** embeds document chunks with `search_document` and questions with `search_query` using 1024 dimensions.
- **LangGraph** retrieves excerpts through the local MCP `search_documents` tool, filters them by a configurable cosine-similarity threshold, generates an answer, checks citation labels, and retries generation once if a label is invalid or absent. With no sufficiently relevant excerpts, it abstains without calling the chat model.
- **OpenRouter** supplies the chat model. `OPENROUTER_MODEL` selects the primary model and `OPENROUTER_FALLBACK_MODEL` is tried if its provider fails. The sample configuration keeps both choices on free endpoints.
- **Langfuse** observability is optional. When credentials are present, the API records startup/shutdown, indexing, and question spans with operational metadata only; it does not send filenames, questions, document text, or answers to Langfuse.
- **MCP** exposes document listing and excerpt search as tools and the citation policy as a resource. For each question, the API launches the stdio server as a local subprocess and LangGraph calls its search tool; document storage stays local.

The citation check confirms that answer labels refer to retrieved excerpts; it does not prove that every claim is semantically entailed by its cited text. Treat answers as research assistance, not clinical advice.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A Cohere API key and an OpenRouter API key for document ingestion and question answering
- Optional Langfuse credentials for tracing

## Setup

```bash
uv sync
cp .env.example .env
```

Set `COHERE_API_KEY` and `OPENROUTER_API_KEY` in `.env`. The default local API key is `dev-local-key`; replace it when running beyond a local demo. `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` may be left blank. Langfuse spans record status, duration, chunk/source counts, model names, and citation-check results, but not user or document content. The retrieval defaults are 300-character chunks, 50-character overlap, top-k 3, and minimum cosine similarity 0.3; they and the primary/fallback chat models and data directory can also be configured in `.env`. Re-index existing documents to apply the updated chunk size and overlap; re-indexing a file with the same filename replaces its previous version. Chat requests have a 30-second timeout and do not retry failed requests; OpenRouter can try the configured fallback model instead.

The prototype sends document excerpts to external model providers. Do not use real patient data.

## Run

Start the API in one terminal:

```bash
uv run uvicorn app.main:app --reload --reload-dir app
```

Start the interface in another terminal:

```bash
uv run streamlit run ui/streamlit_app.py
```

Open the local Streamlit URL printed in the terminal. Upload a PDF, TXT, or Markdown file and click **Index file**. Select a document or search all indexed documents, then ask a question. The answer view displays citation-check status and retrieved excerpts.

The API is available at `http://127.0.0.1:8000`; `/health` is public and the FastAPI OpenAPI page is at `/docs`. Protected requests require the configured key in the `X-API-Key` header.

The API starts the MCP server automatically when a question is asked. To connect
an additional external MCP-compatible host, start the server separately:

```bash
uv run python mcp_server.py
```

The external host can use stdio to access the same local document store and Cohere credentials.

## Tests

```bash
uv run pytest
```

The tests cover chunking, citation checks, graph retry behavior, abstention, and API route protection without calling Cohere, OpenRouter, or Langfuse.

## Retrieval evaluation

Run the small synthetic-note benchmark to compare chunking, `top_k`, and similarity-threshold settings:

```bash
uv run python -m scripts.evaluate_retrieval
```

To run the larger BioASQ subset, first place its `questions.parquet` and
`passages.parquet` files in `evaluation/data/bioasq/`, then run:

```bash
uv run python -m scripts.evaluate_retrieval --dataset bioasq
```

Both experiments use Cohere embeddings but do not call the chat model or modify
the local Chroma index. The BioASQ run uses a reproducible 20-question,
450-passage subset by default. See [evaluation/README.md](./evaluation/README.md)
for data provenance, metrics, and limitations.

## Local data

Uploaded document vectors and the document catalog are stored under `data/` and are excluded from version control. The synthetic sample under `data/samples/` is included in the repository. Deleting the local Chroma directory and catalog removes the indexed documents.
