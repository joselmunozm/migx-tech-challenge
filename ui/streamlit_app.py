"""Small UI over the HTTP API. Run the API first, then this file."""

import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "dev-local-key")

st.set_page_config(page_title="Medical Document Assistant", layout="centered")
st.title("Medical Document Assistant")
st.caption("Ask a question about an uploaded document. The answer cites the excerpts it used.")


def headers() -> dict:
    return {"X-API-Key": API_KEY}


def request(method: str, path: str, *, timeout: float = 120, **kwargs) -> httpx.Response:
    try:
        return httpx.request(method, f"{API_URL}{path}", headers=headers(), timeout=timeout, **kwargs)
    except httpx.HTTPError as exc:
        st.error(
            f"Cannot reach the API at {API_URL}. Start it with "
            "`uv run uvicorn app.main:app --reload --reload-dir app`. "
            f"({exc})"
        )
        st.stop()


def show_error(response: httpx.Response) -> None:
    detail = response.text
    try:
        detail = response.json().get("detail", detail)
    except Exception:
        pass
    st.error(f"{response.status_code}: {detail}")


health = request("GET", "/health")
if health.status_code != 200:
    show_error(health)
    st.stop()
status = health.json()

if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

with st.sidebar:
    st.header("Documents")
    if not status["embeddings_configured"] or not status["llm_configured"]:
        st.warning("Add COHERE_API_KEY and OPENROUTER_API_KEY to the .env file, then restart the API.")
    else:
        st.success(f"Model: {status['model']}")
    if status["langfuse_configured"]:
        st.caption("Langfuse tracing is on.")
    else:
        st.caption("Langfuse tracing is off.")
    if st.button("Clear conversation", disabled=not st.session_state["conversation"]):
        st.session_state["conversation"] = []

    upload = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt", "md"])
    if upload is not None and st.button("Index file", use_container_width=True):
        response = request(
            "POST",
            "/documents",
            timeout=600,
            files={"file": (upload.name, upload.getvalue(), upload.type or "application/octet-stream")},
        )
        if response.status_code == 200:
            st.success(f"Indexed {response.json()['chunk_count']} chunks.")
        else:
            show_error(response)

    catalog = request("GET", "/documents")
    documents = catalog.json()["documents"] if catalog.status_code == 200 else []
    labels = {"All documents": None}
    for row in documents:
        labels[f"{row['filename']} ({row['document_id']})"] = row["document_id"]
    selected = st.selectbox("Search in", list(labels))
    document_id = labels[selected]
    st.caption(f"{len(documents)} document(s) indexed.")


def show_answer(result: dict) -> None:
    st.markdown("**Answer**")
    st.write(result["answer"])
    check = result["citation_check"]
    if check["ok"]:
        st.success("Citations match the retrieved excerpts.")
    else:
        unknown = ", ".join(check["unknown_refs"]) or "missing citation"
        st.warning(f"Citation check failed ({unknown}).")
    if result["attempts"] >= 2:
        st.caption("The answer was rewritten once after the citation check.")
    st.caption(f"{result['model']} · prompt {result['prompt_version']} · {result['attempts']} generation(s)")

    st.markdown("**Sources**")
    if not result["citations"]:
        st.write("No excerpts were retrieved.")
    for source in result["citations"]:
        with st.expander(f"[{source['ref']}] {source['filename']} · score {source['score']}"):
            st.write(source["text"])


st.subheader("Conversation")
st.caption("Each question is answered independently using the currently selected documents.")
for turn in st.session_state["conversation"]:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        show_answer(turn["result"])

question = st.chat_input("Ask a question about the selected documents")
if question and question.strip():
    payload = {"question": question.strip()}
    if document_id:
        payload["document_id"] = document_id
    with st.spinner("Searching documents and generating an answer..."):
        response = request("POST", "/ask", json=payload)
    if response.status_code != 200:
        show_error(response)
    else:
        result = response.json()
        st.session_state["conversation"].append(
            {"question": question.strip(), "result": result}
        )
        with st.chat_message("user"):
            st.write(question.strip())
        with st.chat_message("assistant"):
            show_answer(result)
