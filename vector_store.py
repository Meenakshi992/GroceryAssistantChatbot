from __future__ import annotations

import hashlib
import re
from typing import Any

import chromadb
import streamlit as st
from openai import OpenAI

from text_cleaner import clean_text, extract_uploaded_file


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PATH = "./rag_chroma_db"
COLLECTION_NAME = "grocery_products_assistant_kb"


# ============================================================
# CHROMA CLIENT & COLLECTION
# ============================================================

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(name: str = COLLECTION_NAME):
    return get_chroma_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ============================================================
# EMBEDDING & SESSION EMBEDDING CACHE
# ============================================================

def make_embedding_cache_key(text: str, model: str) -> str:
    # Combine the model name and text so different models don't collide in the cache
    payload = f"{model}:{text}".encode("utf-8")
    # Turn that string into a short, fixed-length unique fingerprint (SHA-256 hash)
    return hashlib.sha256(payload).hexdigest()


def get_embedding(
    client: OpenAI,
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """Convert text into an embedding vector."""
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    # Extract and return the actual list of numbers from OpenAI's response object
    return response.data.pop(0).embedding



def get_cached_embedding(
    client: OpenAI,
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """Check memory cache first; only call OpenAI if we haven't embedded this text yet."""
    # 1. Create an empty dictionary in Streamlit's session_state on the very first run
    if "embedding_cache" not in st.session_state:
        st.session_state["embedding_cache"] = {}

    # 2. Generate the unique SHA-256 fingerprint for this exact text + model
    cache_key = make_embedding_cache_key(text, model)

    # 3. Cache Miss: If this fingerprint is NOT in our dictionary yet, call OpenAI and save it
    if cache_key not in st.session_state["embedding_cache"]:
        st.session_state["embedding_cache"][cache_key] = get_embedding(
            client=client,
            text=text,
            model=model,
        )
    # 4. Cache Hit: Return the saved vector from memory (fast & $0 API cost)
    return st.session_state["embedding_cache"][cache_key]


# ============================================================
# SENTENCE-AWARE CHUNKING
# ============================================================

def split_into_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split cleaned text into smaller overlapping pieces without cutting sentences in half."""
    text = clean_text(text)
    if not text:
        return []

    # If the whole page already fits inside one chunk, return it as-is
    if len(text) <= chunk_size:
        return [text]

    # Break the page into individual sentences so we split at natural punctuation
    sentences = split_into_sentences(text)
    if not sentences:
        sentences = [text]

    chunks = []
    current = ""

    for sentence in sentences:
        # Test adding the next sentence to our current chunk
        candidate = sentence if not current else current + " " + sentence

        # If it still fits under chunk_size, keep adding sentences
        if len(candidate) <= chunk_size or not current:
            current = candidate
            if len(current) <= chunk_size:
                continue

        # Edge case: if a single sentence is longer than chunk_size, slice it safely
        if current and len(current) > chunk_size:
            long_text = current
            start = 0
            while start < len(long_text):
                end = min(start + chunk_size, len(long_text))
                piece = long_text[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(long_text):
                    break
                start = max(0, end - overlap)
            current = ""
            continue

        # Save the full chunk to our list
        chunks.append(current.strip())

        # Copy the last 'overlap' characters into the start of the next chunk
        # so context isn't lost across chunk boundaries
        overlap_prefix = current[-overlap:].strip() if overlap > 0 else ""
        current = (overlap_prefix + " " + sentence).strip() if overlap_prefix else sentence

    # Save any remaining text after the loop finishes
    if current.strip():
        chunks.append(current.strip())

    return chunks

# ============================================================
# INDEXING INTO PERSISTENT CHROMADB
# ============================================================

def index_uploaded_file(
    client: OpenAI,
    collection,
    uploaded_file,
    chunk_size: int,
    overlap: int,
    cleaning_options: dict[str, bool],
) -> int:
    """Extract, clean, chunk, embed (via cache), and upsert into ChromaDB."""
    records = extract_uploaded_file(uploaded_file, cleaning_options)
    if not records:
        return 0

    all_docs: list[str] = []
    all_metas: list[dict[str, Any]] = []
    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []

    stem_name = uploaded_file.name.rsplit(".", 1).pop(0)
    clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem_name)

    for rec in records:
        page_num = rec["page"]
        page_label = f"{page_num:03d}" if page_num is not None else "000"
        header_line = rec["text"].split("\n").pop(0)[:120]
        pieces = chunk_text(rec["text"], chunk_size=chunk_size, overlap=overlap)

        for idx, piece in enumerate(pieces, start=1):
            chunk_id = f"{clean_stem}_{page_label}_{idx:02d}"
            enriched_piece = f"[{header_line}] {piece}" if idx > 1 else piece

            vec = get_cached_embedding(client, enriched_piece, DEFAULT_EMBEDDING_MODEL)

            all_docs.append(enriched_piece)
            all_embeddings.append(vec)
            all_metas.append(
                {
                    "source": rec["source"],
                    "page": page_num if page_num is not None else 0,
                    "chunk_id": chunk_id,
                }
            )
            all_ids.append(chunk_id)

    if not all_ids:
        return 0

    collection.upsert(
        ids=all_ids,
        documents=all_docs,
        embeddings=all_embeddings,
        metadatas=all_metas,
    )
    return len(all_ids)
