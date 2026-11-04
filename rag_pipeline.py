"""
rag_pipeline.py — Core RAG (Retrieval-Augmented Generation) pipeline.

Responsibilities:
  - Connect to IBM watsonx.ai with IBM Granite models
  - Embed curriculum documents into a ChromaDB vector store
  - Retrieve relevant context for a given query
  - Generate grounded responses via IBM Granite LLM
"""

from __future__ import annotations

import logging
import os
from typing import Any

import chromadb
from chromadb.config import Settings
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

from config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# IBM watsonx.ai client singleton
# ─────────────────────────────────────────────────────────────────────────────

_watsonx_client: APIClient | None = None


def get_watsonx_client() -> APIClient:
    """Return (or lazily create) the shared watsonx APIClient."""
    global _watsonx_client
    if _watsonx_client is None:
        credentials = Credentials(
            url=Config.IBM_CLOUD_URL,
            api_key=Config.IBM_CLOUD_API_KEY,
        )
        _watsonx_client = APIClient(credentials=credentials)
        logger.info("IBM watsonx.ai APIClient initialised.")
    return _watsonx_client


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper (IBM Granite / Slate embeddings via watsonx.ai)
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings using IBM watsonx.ai embedding model.
    Returns a list of float vectors.
    """
    embedding_model = Embeddings(
        model_id=Config.GRANITE_EMBEDDING_MODEL_ID,
        api_client=get_watsonx_client(),
        project_id=Config.IBM_WATSONX_PROJECT_ID,
    )
    return embedding_model.embed_documents(texts)


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB vector store
# ─────────────────────────────────────────────────────────────────────────────

_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """Return (or lazily open) the ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is None:
        Config.ensure_directories()
        _chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' opened. Documents: %d",
            Config.CHROMA_COLLECTION_NAME,
            _collection.count(),
        )
    return _collection


# ─────────────────────────────────────────────────────────────────────────────
# Document ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split *text* into overlapping chunks of roughly *chunk_size* words."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def ingest_documents(documents: list[dict[str, Any]]) -> int:
    """
    Ingest a list of document dicts into the vector store.

    Each dict must have:
        - ``text``  : str  — document content
        - ``id``    : str  — unique document identifier
        - ``metadata``: dict (optional) — arbitrary metadata

    Returns the number of chunks stored.
    """
    collection = get_collection()
    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    for doc in documents:
        chunks = _chunk_text(
            doc["text"],
            Config.RAG_CHUNK_SIZE,
            Config.RAG_CHUNK_OVERLAP,
        )
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}__chunk_{idx}"
            meta = {**(doc.get("metadata") or {}), "source_id": doc["id"], "chunk_index": idx}
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_meta.append(meta)

    if not all_chunks:
        return 0

    embeddings = embed_texts(all_chunks)
    collection.upsert(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_meta,
    )
    logger.info("Ingested %d chunks from %d documents.", len(all_chunks), len(documents))
    return len(all_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_context(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant document chunks for *query*.

    Returns a list of dicts with keys: ``text``, ``id``, ``metadata``, ``distance``.
    """
    k = top_k or Config.RAG_TOP_K_RESULTS
    collection = get_collection()

    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[dict[str, Any]] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    for text, meta, dist, rid in zip(docs, metas, dists, ids):
        similarity = 1.0 - dist  # cosine distance → similarity
        if similarity >= Config.RAG_SIMILARITY_THRESHOLD:
            retrieved.append(
                {"text": text, "id": rid, "metadata": meta, "similarity": round(similarity, 4)}
            )

    logger.debug("Retrieved %d relevant chunks for query.", len(retrieved))
    return retrieved


# ─────────────────────────────────────────────────────────────────────────────
# IBM Granite LLM inference
# ─────────────────────────────────────────────────────────────────────────────

def generate_with_granite(prompt: str, max_new_tokens: int | None = None) -> str:
    """
    Send *prompt* to IBM Granite via watsonx.ai and return the generated text.
    """
    params = {
        GenParams.MAX_NEW_TOKENS: max_new_tokens or Config.LLM_MAX_NEW_TOKENS,
        GenParams.TEMPERATURE: Config.LLM_TEMPERATURE,
        GenParams.TOP_P: Config.LLM_TOP_P,
        GenParams.TOP_K: Config.LLM_TOP_K,
        GenParams.REPETITION_PENALTY: Config.LLM_REPETITION_PENALTY,
    }

    model = ModelInference(
        model_id=Config.GRANITE_LLM_MODEL_ID,
        api_client=get_watsonx_client(),
        project_id=Config.IBM_WATSONX_PROJECT_ID,
        params=params,
    )

    response = model.generate_text(prompt=prompt)
    return response.strip()


# ─────────────────────────────────────────────────────────────────────────────
# RAG-augmented generation
# ─────────────────────────────────────────────────────────────────────────────

def rag_generate(
    query: str,
    system_instruction: str,
    extra_context: str = "",
    top_k: int | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Full RAG cycle:
      1. Retrieve relevant chunks from the vector store.
      2. Build a grounded prompt.
      3. Generate a response with IBM Granite.

    Returns a dict with ``answer``, ``retrieved_chunks``, and ``prompt``.
    """
    chunks = retrieve_context(query, top_k=top_k)
    context_block = "\n\n---\n\n".join(c["text"] for c in chunks)
    if extra_context:
        context_block = extra_context + "\n\n" + context_block

    prompt = (
        f"{system_instruction}\n\n"
        f"### Retrieved Curriculum Context:\n{context_block}\n\n"
        f"### User Request:\n{query}\n\n"
        f"### Response:\n"
    )

    answer = generate_with_granite(prompt, max_new_tokens=max_new_tokens)
    return {
        "answer": answer,
        "retrieved_chunks": chunks,
        "prompt_length": len(prompt.split()),
    }
