from __future__ import annotations

import time
from typing import Any

from openai import OpenAI

from vector_store import DEFAULT_EMBEDDING_MODEL, get_cached_embedding


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_GENERATION_MODEL = "gpt-4o-mini"

DEFAULT_FALLBACK = (
    "I could not find enough relevant information in the uploaded documents "
    "to answer that question reliably."
)


# ============================================================
# RELEVANCE SCORING & GENERATION
# ============================================================

def classify_relevance(similarity: float, threshold: float) -> str:
    if similarity >= threshold + 0.15:
        return "Strongly Relevant"
    if similarity >= threshold:
        return "Potentially Relevant"
    if similarity >= threshold - 0.10:
        return "Weakly Relevant"
    return "Irrelevant"



def retrieve_and_answer(
    client: OpenAI,
    collection,
    question: str,
    top_k: int = 6,
    threshold: float = 0.35,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    generation_model: str = DEFAULT_GENERATION_MODEL,
) -> dict[str, Any]:
    """Embed the user's question, find matching chunks, check relevance, and generate a cited answer."""
    t0 = time.perf_counter()

    # Safety check: stop early if no documents have been indexed yet
    if collection.count() == 0:
        return {
            "answer": "The knowledge base is empty. Please upload and index at least 4 documents first.",
            "sources": [],
            "metrics": {"retrieval_ms": 0.0, "total_ms": 0.0},
        }

    # Step 1: Turn the user's question into a vector (reusing memory cache if asked before)
    query_vec = get_cached_embedding(client, question, model=embedding_model)

    # Make sure we don't ask ChromaDB for more chunks than exist in the database
    n_results = min(top_k, collection.count())

    # Step 2: Search ChromaDB for the closest matching document chunks
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    t_retrieval = time.perf_counter()

    # Unpack the inner lists returned by ChromaDB
    docs = results["documents"].pop(0)
    metas = results["metadatas"].pop(0)
    dists = results["distances"].pop(0)

    sources: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []

    # Step 3: Score every retrieved chunk and filter out weak/irrelevant matches
    for doc, meta, dist in zip(docs, metas, dists):
        # Convert Chroma cosine distance (0 = identical) into similarity (1 = identical)
        similarity = round(1.0 - float(dist), 4)
        label = classify_relevance(similarity, threshold)
        page_val = meta.get("page", 0)

        record = {
            "source": meta.get("source", "Unknown"),
            "page": page_val if page_val > 0 else None,
            "chunk_id": meta.get("chunk_id", "N/A"),
            "similarity": similarity,
            "label": label,
            "text": doc,
        }
        sources.append(record)

        # Only keep chunks that meet or beat the similarity cutoff
        if similarity >= threshold:
            accepted.append(record)

    # Step 4: Safe Fallback — if zero chunks passed the cutoff, refuse to guess
    if not accepted:
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        retrieval_ms = round((t_retrieval - t0) * 1000, 1)
        return {
            "answer": DEFAULT_FALLBACK,
            "sources": sources,
            "metrics": {"retrieval_ms": retrieval_ms, "total_ms": total_ms},
        }

    # Step 5: Combine the accepted chunks with their filenames and page numbers
    context_parts = []
    for item in accepted:
        page_cite = f", p. {item['page']}" if item["page"] is not None else ""
        context_parts.append(
            f"[Source: {item['source']}{page_cite} | Chunk: {item['chunk_id']}]\n{item['text']}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    # Strict instructions forcing the LLM to rely ONLY on the retrieved context
    system_prompt = (
        "You are grocery-products-assistant, a grounded AI assistant for grocery flyers, products, and store policies. "
        "Answer the user's question using ONLY the retrieved context below. "
        "Do not invent, guess, or assume prices, dates, or policy rules that are not explicitly stated. "
        "If the context only partially answers the question, state what is missing. "
        "Cite supporting evidence inline after every claim using [Source: filename, p. X] or [Source: filename]."
    )

    # Step 6: Send the context and question to OpenAI to generate the final cited answer
    completion = client.chat.completions.create(
        model=generation_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ],
    )

    answer = completion.choices.pop(0).message.content or DEFAULT_FALLBACK
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    retrieval_ms = round((t_retrieval - t0) * 1000, 1)

    return {
        "answer": answer,
        "sources": sources,
        "metrics": {"retrieval_ms": retrieval_ms, "total_ms": total_ms},
    }
