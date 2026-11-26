import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
import re
import pandas as pd
from chat_engine import retrieve_and_answer
from sidebar_ui import render_sidebar
from store_locator import get_nearby_stores
from vector_store import COLLECTION_NAME, get_collection

st.set_page_config(page_title="grocery-products-assistant", layout="wide")

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
    <div class="app-banner">
        <h1>grocery-products-assistant</h1>
        <p>Surrey, BC Grounded Grocery Flyer & Store Policy Assistant — Upload at least 4 documents and get cited answers backed strictly by your uploaded files.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "knowledge_base" not in st.session_state:
    st.session_state["knowledge_base"] = False
if "total_documents" not in st.session_state:
    st.session_state["total_documents"] = []
if "chunk_count" not in st.session_state:
    st.session_state["chunk_count"] = 0
if "embedding_cache" not in st.session_state:
    st.session_state["embedding_cache"] = {}

# Load OpenAI API key securely
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
    st.stop()


@st.cache_resource
def init_openai(key):
    return OpenAI(api_key=key)


client = init_openai(st.secrets["OPENAI_API_KEY"])
collection = get_collection(COLLECTION_NAME)

# checking how many files are in our chroma db using src metadata count calling it
def refresh_documents_count():
    metas = collection.get(include=["metadatas"]).get("metadatas") or []
    counts = {}
    for m in metas:
        src = m["source"]
        counts[src] = counts.get(src, 0) + 1

    files_info = []
    for name in sorted(counts.keys()):
        files_info.append({"File Name": name, "Chunks": counts[name]})

    st.session_state["total_documents"] = files_info
    st.session_state["chunk_count"] = collection.count()
    st.session_state["knowledge_base"] = len(counts) >= 4


refresh_documents_count()
cfg = render_sidebar(client, collection, refresh_documents_count)

def show_price_chart(chart_data):
    """Renders a Matplotlib bar chart of grounded store prices and highlights price gaps."""
    if not chart_data or len(chart_data) == 0:
        return

    df = pd.DataFrame(chart_data)
    if "item" not in df.columns or "store" not in df.columns or "price" not in df.columns:
        return

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    if df.empty:
        return

    pivot_df = df.pivot_table(index="item", columns="store", values="price", aggfunc="min")

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    pivot_df.plot(kind="bar", ax=ax, width=0.7, edgecolor="#444444", linewidth=0.6)

    ax.set_title("Grounded Flyer Price Comparison by Store ($ CAD)", fontsize=10, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Price ($ CAD)", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=20, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="$%.2f", padding=2, fontsize=7)

    if pivot_df.shape[1] >= 2:
        for idx, item_name in enumerate(pivot_df.index):
            row_prices = pivot_df.loc[item_name].dropna()
            if len(row_prices) >= 2:
                gap = row_prices.max() - row_prices.min()
                top_y = row_prices.max()
                ax.annotate(
                    f"Gap: ${gap:.2f}",
                    xy=(idx, top_y),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="#B71C1C",
                    bbox=dict(boxstyle="round,pad=0.18", fc="#FFEBEE", ec="#EF9A9A", lw=0.6),
                )

    ax.set_ylim(0, pivot_df.max().max() * 1.25)
    ax.legend(title="Store", fontsize=7.0, title_fontsize=7.5, frameon=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

STOP_WORDS = {"what", "where", "when", "which", "who", "how", "is", "are", "the", "a", "an", "of", "at", "in", "for", "to", "and", "or", "my", "me", "can", "do", "does", "this", "week"}

def highlight_chunk(text, question):
    """Wraps matching keywords from the user's question in yellow <mark> tags."""
    if not question:
        return text

    words = set()
    for w in re.findall(r"[a-zA-Z0-9]+", question.lower()):
        if len(w) >= 3 and w not in STOP_WORDS:
            words.add(w)

    highlighted = text
    for word in sorted(words, key=len, reverse=True):
        pattern = re.compile(rf"\b({re.escape(word)}s?)\b", re.IGNORECASE)
        highlighted = pattern.sub(r"<mark style='background-color: #FFF59D; padding: 0 3px; border-radius: 3px;'>\1</mark>", highlighted)

    return highlighted

def show_sources(sources, question=""):
    if len(sources) == 0:
        return
    with st.expander("View Sources"):
        for idx, src in enumerate(sources, start=1):
            page_txt = f"Page {src['page']}" if src["page"] else "Text Section"
            dist_val = src.get("distance", round(1.0 - src["similarity"], 3))
            with st.container(border=True):
                st.write(
                    f"**{idx}. {src['source']}** | {page_txt} | Chunk `{src['chunk_id']}` | "
                    f"Similarity: `{src['similarity']}` | Distance: `{dist_val}` ({src['label']})"
                )
                highlighted_text = highlight_chunk(src["text"], question)
                st.markdown(highlighted_text, unsafe_allow_html=True)

# Nearby Surrey Stores Panel
address, nearby_df, is_valid_pin = get_nearby_stores(cfg["pincode"], cfg["max_radius"])
stores_in_radius = nearby_df["Store"].tolist() if (is_valid_pin and not nearby_df.empty) \
    else  st.error("Sorry, this pincode not from Surrey,BC.")

with st.expander("Nearby Surrey Stores & Local Flyer Suggestions", expanded=True):
    if not is_valid_pin:
        st.error(address)
    else:
        st.caption(f"Location: {address} | Search restricted to flyers within {cfg['max_radius']} km")
    if nearby_df.empty:
        st.info("No partner stores found within this radius. Increase the Radius slider in the sidebar.")
    else:
        db_text = " ".join(collection.get(include=["documents"]).get("documents") or []).lower()
        statuses = []
        missing = []

        for idx, row in nearby_df.iterrows():
            if row["Store"].lower() in db_text:
                statuses.append("Uploaded")
            else:
                statuses.append("Missing - Upload Flyer")
                missing.append(f"{row['Store']} ({row['Branch']})")

        nearby_df["Flyer Status"] = statuses

        col1, col2 = st.columns([1.5, 1.0])
        col1.dataframe(nearby_df[["Store", "Branch", "Distance (km)", "Flyer Status"]], use_container_width=True, hide_index=True)
        col2.map(nearby_df[["lat", "lon"]], zoom=12)

        if len(missing) > 0:
            st.warning(f"Missing flyers for: **{', '.join(missing)}**. Upload flyer as a PDF/TXT and upload it in the sidebar!")

        if st.session_state["knowledge_base"] and st.button("Suggest Top Flyer Deals for My Nearby Surrey Stores",
                                                             use_container_width=True):
            store_list = ", ".join(stores_in_radius)
            prompt = f"Based ONLY on the uploaded flyers within my radius ({store_list}), list the best deals. Include flyer dates and citations."
            with st.spinner("Checking flyers within your radius..."):
                res = retrieve_and_answer(
                    client,
                    collection,
                    prompt,
                    cfg["top_k"],
                    cfg["threshold"],
                    allowed_stores=stores_in_radius,
                )
            st.markdown(res["answer"])
            show_price_chart(res.get("chart_data", []))
            show_sources(res["sources"])

# Chat History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metrics" in msg:
            st.caption(f"Retrieval: {msg['metrics']['retrieval_ms']} ms | Total: {msg['metrics']['total_ms']} ms")
        if "sources" in msg:
            # Save "question": user_prompt when appending the assistant message, then pass:
            show_sources(msg["sources"], msg.get("question", ""))

        st.markdown(msg["content"])
        if "chart_data" in msg:
            show_price_chart(msg["chart_data"])

if not st.session_state["knowledge_base"]:
    st.info("Upload and index at least 4 documents in the sidebar to start chatting.")

if user_prompt := st.chat_input("Ask about weekly flyer deals or store policies...", disabled=not st.session_state["knowledge_base"]):
    st.session_state["messages"].append({"role": "user", "content": user_prompt ,"question": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents within your radius..."):
                outcome = retrieve_and_answer(client, collection, user_prompt, cfg["top_k"], cfg["threshold"], allowed_stores=stores_in_radius)
            st.markdown(outcome["answer"])
            show_price_chart(outcome.get("chart_data", ))
            st.caption(f"Retrieval: {outcome['metrics']['retrieval_ms']} ms | Total: {outcome['metrics']['total_ms']} ms")
            show_sources(outcome["sources"], user_prompt)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": outcome["answer"],
                "chart_data": outcome.get("chart_data", ),
                "sources": outcome["sources"],
                "metrics": outcome["metrics"],
            })
        except Exception as err:
            st.error(f"Error: {err}")



