from __future__ import annotations

import pandas as pd
import streamlit as st
from openai import OpenAI

from chat_engine import retrieve_and_answer
from vector_store import COLLECTION_NAME, get_collection, index_uploaded_file


# ============================================================
# PAGE CONFIGURATION & ORIGINAL THEME STYLING
# ============================================================

st.set_page_config(
    page_title="grocery-products-assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1120px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        .app-banner {
            border-left: 5px solid #2E7D32;
            background-color: rgba(46, 125, 50, 0.06);
            padding: 1rem 1.25rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 1.25rem;
        }
        .app-banner h1 {
            font-size: 1.85rem;
            margin: 0 0 0.3rem 0;
        }
        .app-banner p {
            margin: 0;
            font-size: 0.98rem;
            opacity: 0.85;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE SETUP
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "knowledge_ready": False,
    "collection_name": COLLECTION_NAME,
    "document_manifest": [],
    "chunk_count": 0,
    "last_sources": [],
    "last_metrics": {},
    "last_question": "",
    "embedding_cache": {},
}

for key, default_val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


# ============================================================
# OPENAI & CHROMADB INITIALIZATION
# ============================================================

def load_openai_key() -> str | None:
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


@st.cache_resource
def init_openai(key: str) -> OpenAI:
    return OpenAI(api_key=key)


api_key = load_openai_key()
if not api_key:
    st.error("API key not found. Please set OPENAI_API_KEY inside .streamlit/secrets.toml and reload.")
    st.stop()

client = init_openai(api_key)

try:
    collection = get_collection(COLLECTION_NAME)
except Exception as exc:
    st.error(f"Could not connect to ChromaDB storage: {exc}")
    st.stop()


def refresh_manifest():
    records = collection.get(include=["metadatas"])
    metas = records.get("metadatas") or []
    doc_counts: dict[str, int] = {}

    for meta in metas:
        if isinstance(meta, dict):
            doc_name = meta.get("source", "Unknown")
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1

    st.session_state["document_manifest"] = [
        {"File Name": name, "Chunks": count} for name, count in sorted(doc_counts.items())
    ]
    st.session_state["chunk_count"] = collection.count()
    st.session_state["knowledge_ready"] = len(doc_counts) >= 4


refresh_manifest()


# ============================================================
# TOP HEADER BANNER
# ============================================================

st.markdown(
    """
    <div class="app-banner">
        <h1>grocery-products-assistant</h1>
        <p>
            Grounded Grocery Flyer & Store Policy Assistant — Upload at least 4 documents to build your
            searchable knowledge base and get cited answers backed by retrieved evidence.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR: INGESTION, CLEANING & RETRIEVAL CONTROLS
# ============================================================

with st.sidebar:
    st.subheader("Step 1: Upload Grocery Files")
    st.caption("Minimum 4 documents required (PDF, TXT, MD, or DOCX).")

    uploaded_files = st.file_uploader(
        "Choose files to index",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
    )

    with st.expander("Text Preprocessing Options", expanded=True):
        cleaning_options = {
            "fix_hyphens": st.checkbox("Fix broken hyphenated words", value=True),
            "remove_page_nums": st.checkbox("Strip standalone page numbers", value=True),
            "fix_lines": st.checkbox("Merge broken lines in paragraphs", value=True),
            "clean_spaces": st.checkbox("Collapse extra spaces & blank lines", value=True),
        }

    with st.expander("Chunking & Retrieval Tuning", expanded=False):
        chunk_size = st.slider("Chunk Size (chars)", 300, 1200, 700, step=100)
        overlap = st.slider("Chunk Overlap (chars)", 0, 300, 120, step=30)
        top_k = st.slider("Top-K Chunks Retrieved", 1, 8, 4)
        threshold = st.slider("Similarity Cutoff", 0.10, 0.80, 0.35, step=0.05)

    existing_docs = {row["File Name"] for row in st.session_state["document_manifest"]}
    selected_docs = {f.name for f in uploaded_files} if uploaded_files else set()
    combined_count = len(existing_docs | selected_docs)

    if uploaded_files:
        if combined_count < 4:
            st.warning(f"Select at least 4 unique files ({combined_count}/4 ready).")
        else:
            if st.button("Index Uploaded Documents", type="primary", use_container_width=True):
                total_added = 0
                for file_item in uploaded_files:
                    try:
                        with st.spinner(f"Reading and embedding {file_item.name}..."):
                            total_added += index_uploaded_file(
                                client=client,
                                collection=collection,
                                uploaded_file=file_item,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                cleaning_options=cleaning_options,
                            )
                    except Exception as err:
                        st.error(f"Could not process {file_item.name}: {err}")

                refresh_manifest()
                if total_added > 0:
                    st.success(f"Indexed {total_added} chunks into ChromaDB.")
                    st.rerun()

    st.divider()
    st.subheader("Knowledge Base Summary")
    col_docs, col_chunks, col_cache = st.columns(3)
    col_docs.metric("Files", len(st.session_state["document_manifest"]))
    col_chunks.metric("Chunks", st.session_state["chunk_count"])
    # When you ask a new question, Cached Vectors increases by 1
    # (the new query vector)—and if you ask that same question again
    # or adjust Top-K, Cached Vectors stays the same, proving no duplicate
    # OpenAI embedding calls were made.
    col_cache.metric("Cached Vectors", len(st.session_state["embedding_cache"]))

    if st.session_state["document_manifest"]:
        st.dataframe(
            pd.DataFrame(st.session_state["document_manifest"]),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Reset Database", use_container_width=True):
            ids_to_delete = collection.get().get("ids") or []
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
            st.session_state["messages"] = []
            st.session_state["last_sources"] = []
            st.session_state["embedding_cache"] = {}
            refresh_manifest()
            st.rerun()



# ============================================================
# MAIN CHAT AREA & SOURCE VIEWER
# ============================================================

def display_source_drawer(sources: list[dict]):
    if not sources:
        return
    with st.expander("View Sources"):
        for idx, src in enumerate(sources, start=1):
            page_label = f"Page {src['page']}" if src.get("page") else "Text Section"
            with st.container(border=True):
                st.markdown(
                    f"**Source {idx}: {src['source']}** | {page_label} | Chunk `{src['chunk_id']}`"
                )
                st.caption(f"Cosine Similarity: {src['similarity']} ({src['label']})")
                st.write(src["text"])


for entry in st.session_state["messages"]:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("metrics"):
            m = entry["metrics"]
            st.caption(f"Retrieval: {m.get('retrieval_ms', 0)} ms | Total: {m.get('total_ms', 0)} ms")
        if entry.get("sources"):
            display_source_drawer(entry["sources"])


if not st.session_state["knowledge_ready"]:
    st.info("Upload and index at least 4 grocery documents in the left sidebar to unlock the chat assistant.")

if user_prompt := st.chat_input(
    "Ask about weekly flyer deals, product sizes, or store policies...",
    disabled=not st.session_state["knowledge_ready"],
):
    st.session_state["last_question"] = user_prompt
    st.session_state["messages"].append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching knowledge base and verifying citations..."):
                outcome = retrieve_and_answer(
                    client=client,
                    collection=collection,
                    question=user_prompt,
                    top_k=top_k,
                    threshold=threshold,
                )

            st.markdown(outcome["answer"])
            m = outcome["metrics"]
            st.caption(f"Retrieval: {m['retrieval_ms']} ms | Total: {m['total_ms']} ms")
            display_source_drawer(outcome["sources"])

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": outcome["answer"],
                    "sources": outcome["sources"],
                    "metrics": m,
                }
            )
        except Exception as exc:
            err_msg = f"Unable to complete request: {exc}"
            st.error(err_msg)
            st.session_state["messages"].append({"role": "assistant", "content": err_msg})
