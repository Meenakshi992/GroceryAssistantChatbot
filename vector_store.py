import hashlib
import chromadb
import streamlit as st
from text_cleaner import clean_text, clean_extract_uploaded_file

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PATH = "./chroma_rag_db_GAC"
COLLECTION_NAME = "grocery_products_assistant_kb"

KNOWN_STORES = [
    "No Frills",
    "Walmart Supercentre",
    "Real Canadian Superstore",
    "Save-On-Foods",
    "FreshCo",
    "Safeway",
    "Thrifty Foods",
]

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(name=COLLECTION_NAME):
    return get_chroma_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

def get_embedding(client, text, model=DEFAULT_EMBEDDING_MODEL):
    """Calls OpenAI to convert text into a vector."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def get_cached_embedding(client, text, model=DEFAULT_EMBEDDING_MODEL):
    """Reuses saved vectors from session_state so we don't call OpenAI twice for the same text."""
    if "embedding_cache" not in st.session_state:
        st.session_state["embedding_cache"] = {}

    cache_key = hashlib.sha256(f"{model}:{text}".encode("utf-8")).hexdigest()

    if cache_key not in st.session_state["embedding_cache"]:
        st.session_state["embedding_cache"][cache_key] = get_embedding(client, text, model)

    return st.session_state["embedding_cache"][cache_key]

def detect_store_name(filename, text):
    """Identifies which store chain a flyer belongs to, or marks it as 'Policy'."""
    combined = f"{filename} {text[:300]}".lower()
    for store in KNOWN_STORES:
        if store.lower() in combined or store.lower().replace(" ", "_") in combined:
            return store
    if "walmart" in combined:
        return "Walmart Supercentre"
    if "superstore" in combined:
        return "Real Canadian Superstore"
    return "Policy"


def chunk_text(text, chunk_size, overlap):
    """Splits text into overlapping chunks of size chunk_size."""
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        piece = text[start:start + chunk_size].strip()
        if piece != "":
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
        start += step

    return chunks


def index_uploaded_file(client, collection, uploaded_file, chunk_size, overlap, cleaning_options):
    """Extracts, chunks, embeds, and saves an uploaded file into ChromaDB with store metadata."""
    records = clean_extract_uploaded_file(uploaded_file, cleaning_options)
    if len(records) == 0:
        return {"chunks_added": 0, "extracted_text": ""}

    first_record = records[0]
    first_text = first_record["text"]
    store_tag = detect_store_name(uploaded_file.name, first_text)
    all_docs = []
    all_metas = []
    all_ids = []
    all_embeddings = []
    extracted_pages = []

    clean_name = uploaded_file.name.replace(".", "_").replace(" ", "_")

    for rec in records:
        page_num = rec["page"]
        header_line = rec["text"].split("\n")[0][:100]
        extracted_pages.append(rec["text"])

        pieces = chunk_text(rec["text"], chunk_size, overlap)

        for i in range(len(pieces)):
            chunk_id = f"{clean_name}_p{page_num}_c{i + 1}"
            piece_text = f"[{header_line}] {pieces[i]}" if i > 0 else pieces[i]
            vector = get_cached_embedding(client, piece_text)

            all_docs.append(piece_text)
            all_embeddings.append(vector)
            all_metas.append({
                "source": rec["source"],
                "page": page_num if page_num is not None else 0,
                "chunk_id": chunk_id,
                "store": store_tag
            })
            all_ids.append(chunk_id)

    if len(all_ids) > 0:
        collection.upsert(
            ids=all_ids,
            documents=all_docs,
            embeddings=all_embeddings,
            metadatas=all_metas,
        )

    return {
        "chunks_added": len(all_ids),
        "extracted_text": "\n\n".join(extracted_pages),
    }
