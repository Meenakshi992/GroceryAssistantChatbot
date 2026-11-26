# grocery-products-assistant

**grocery-products-assistant** is a Surrey, BC grounded Retrieval-Augmented Generation (RAG) chatbot built with **Python**, **Streamlit**, **ChromaDB**, and **OpenAI (`text-embedding-3-small` & `gpt-4o-mini`)**. Instead of guessing answers from general internet knowledge, it answers shopper questions using **only** the grocery flyers, product catalogs, and store policies you upload—displaying per-item comparison tables, Matplotlib price-gap bar charts, and inline source citations while refusing to invent answers when supporting facts are not found.

## How the App Works (11-Step RAG Pipeline)

```text
Document -> Load -> Clean -> Chunk -> Embed -> Store -> Query -> Retrieve -> Filter -> Generate -> Cite
```

1. **Document:** Users upload at least 4 relevant text-based grocery documents (`.pdf`, `.txt`, or `.md`) in the sidebar (`sidebar_ui.py`).
2. **Load:** `extract_uploaded_file()` in `text_cleaner.py` reads uploaded file bytes using `uploaded_file.getvalue()` and extracts text page-by-page using `pypdf` for PDFs and UTF-8 decoding for `.txt`/`.md` files.
3. **Clean:** `clean_text()` in `text_cleaner.py` removes formatting noise before indexing—fixing hyphenated line wraps, removing standalone page numbers, joining broken lines, and collapsing extra spaces.
4. **Chunk:** `chunk_text()` and `index_uploaded_file()` in `vector_store.py` split cleaned text into overlapping sliding-window chunks (`chunk_size=700`, `overlap=120`) and attach `source`, `page`, `chunk_id`, and `store` metadata to each chunk.
5. **Embed:** Each chunk is converted into a 1,536-dimensional vector using OpenAI's `text-embedding-3-small` model, cached in `st.session_state["embedding_cache"]` via SHA-256 keys (`get_cached_embedding()`) to eliminate duplicate API calls.
6. **Store:** All chunks, embeddings, and metadata are saved into a persistent local **ChromaDB** The ChromaDB DB do save documents so they stay available across app restarts without re-indexing.
7. **Query:** When a user submits a question in the chat interface (`app.py`), `get_cached_embedding()` converts the question into a query vector using `text-embedding-3-small`.
8. **Retrieve:** `retrieve_and_answer()` in `chat_engine.py` searches ChromaDB—filtered by the user's selected Surrey postal code radius (`allowed_stores`) and policy-keyword routing—and retrieves the **Top-K** (default `5`) most similar chunks along with cosine distance scores.
9. **Filter:** Cosine distance is converted to similarity (`1 - distance`), classified (`Strongly Relevant`, `Potentially Relevant`, `Weakly Relevant`, or `Irrelevant`), and checked against the **Similarity Cutoff** (`0.35`). If zero chunks pass, generation stops and a safe fallback message is returned.
10. **Generate:** Accepted chunks are sent along with the user's question and strict anti-hallucination rules (`temperature=0.0`) to `gpt-4o-mini` to generate a concise summary and structured price data for per-item tables (`show_item_tables()`) and `matplotlib` price-gap bar charts (`show_price_chart()`).
11. **Cite:** The assistant includes inline citations (`[Source: filename, p. 1]`) after every claim, and users can expand **View Sources** under any reply to inspect the exact retrieved text with highlighted query terms, page number, chunk ID, similarity score, and cosine distance.

---

## Design Choices & Settings

* **Chunk Size (`700`) & Overlap (`120`):** Grocery flyers and store policies consist of short product deals and compact policy clauses. A 700-character sliding-window chunk keeps a full item deal or policy rule together without diluting the embedding vector, while a 120-character overlap prevents prices or conditions from getting cut off across boundaries.
* **Top-K (`5`):** Retrieving up to 5 chunks captures matching product deals across nearby Surrey store flyers while keeping context focused and fast.
* **Similarity Cutoff (`0.35`):** Testing showed that valid grocery questions score above `0.38` (even with spelling mistakes), while unrelated questions score below `0.25`. Setting the cutoff at `0.35` cleanly rejects out-of-scope questions.
* **Surrey, BC Postal Code Store Locator (`store_locator.py`):** Resolves a user's Surrey postal code (e.g., `V3R 0A4`) to identify the single closest branch per chain (**No Frills**, **Walmart Supercentre**, **Real Canadian Superstore**, **Save-On-Foods**, **FreshCo**, **Safeway**, and **Thrifty Foods**), shows which nearby stores have uploaded flyers, and restricts ChromaDB retrieval strictly to stores within the chosen radius.

---

## Project Structure

```text
grocery-products-assistant/
├── .streamlit/
│   └── secrets.toml        # Local OpenAI API key (excluded from Git via .gitignore)
├── .gitignore              # Keeps secrets.toml and chroma_rag_db_GAC/ off GitHub
├── sidebar_ui.py           # Sidebar upload controls, cleaning options, pincode & KB summary
├── text_cleaner.py         # Text extraction (pypdf, UTF-8) and text cleaning
├── vector_store.py         # Overlapping chunking, SHA-256 embedding cache & ChromaDB
├── chat_engine.py          # Radius-filtered semantic search, cutoff & grounded generation
├── store_locator.py        # Surrey, BC postal code geolocation & nearest branch finder
├── app.py                  # Main Streamlit UI, per-item tables, Matplotlib charts & source drawer
├── requirements.txt        # Pinned Python dependencies
└── README.md               # Architecture documentation, 10-question test log & reflections
```

---

## How to Run Locally

1. **Install required packages:**
   pip install -r requirements.txt
2. **Add your OpenAI API key:**
   Create `.streamlit/secrets.toml` and add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
3. **Start the app:**
   ```bash
   python -m streamlit run app.py
   ```
4. **Index documents:** Upload at least 4 text-based grocery `.pdf`, `.txt`, or `.md` files in the sidebar, choose your cleaning options, and click **Index Uploaded Documents**.

---

## 10-Question Test Log

Tested using 4 Surrey grocery PDF documents (`no_frills_weekly_flyer.pdf`, `walmart_guildford_flyer.pdf`, `superstore_next_week_flyer.pdf`, `surrey_price_match_and_raincheck_policy.pdf`) with **Top-K = 5**, **Similarity Cutoff = 0.35**, and **Radius = 6.0 km (`V3R 0A4`)**.
1 Direct | what is the price of apples, banana, bread, chicken and milk in all nearby stores
2 Direct | What is the price of Tilda Pure Basmati Rice (4.5 kg) at Walmart Guildford?
3 Direct | When do next week's Real Canadian Superstore flyer prices start and end?
4 Paraphrased | How long do I have to use a raincheck before it expires, and where can I use it?
5 Paraphrased | What do I need to show at the register to get a competitor's lower advertised price?
6 Misspelled | How much is whole wheet bred and chedar chese at Superstor next week?
7 Multi-Document | Compare the price of Lean Ground Beef and Fresh Cavendish Bananas across No Frills, Walmart, and Superstore.
8 Multi-Document | If No Frills has Dempster's Whole Wheat Bread (675g) for $2.97 this week and Superstore has it for $3.29, can Superstore price match it?
9 Irrelevant | What are the best tourist attractions to visit in Paris, France?
10 Irrelevant | How do I bake a chocolate lava cake from scratch at 400 degrees?

check embedding cache=>
what is the price of eggs in all nearby store
which has least price of eggs among above stores

## Project Reflections & Lessons Learned

### 1. What Worked Well
* **Strict Grounding & Hallucination Prevention:** Combining a cosine similarity cutoff (`0.35`) with deterministic generation (`temperature=0.0`), strict rules against guessing unclear regular price ranges (e.g., `reg 3.69-3.79`), and inline `[Source: filename, p. 1]` citations prevented the LLM from guessing prices or answering out-of-scope questions.
* **Location-Restricted RAG (`V3R 0A4` / Surrey, BC):** Integrating postal code distance filtering directly into ChromaDB's metadata `where` clause ensured that price comparisons only pulled deals from stores within the user's chosen travel radius rather than across the entire city.
* **Visual Price Gap Charts & Per-Item Tables:** Converting retrieved flyer prices into separate item tables and grouped `matplotlib` bar charts with `Gap: \$X.XX` callouts made multi-store price differences clear and scannable.
* **Performance & Persistence:** Persisting vectors on disk (`./chroma_rag_*****`) alongside a SHA-256 session `embedding_cache` kept query retrieval fast (~150–300 ms) and eliminated duplicate OpenAI embedding calls across restarts.

### 2. Technical Challenges Solved
* **Text-Based vs. Image-Only Flyers:** Commercial grocery flyers (such as web flyer viewers on Flipp) often flatten prices into images rather than selectable text. Focusing the ingestion pipeline on fast, clean text extraction (`pypdf` and UTF-8 `.txt`/`.md`) with clear user guidance kept indexing fast, lightweight, and reliable without heavy OCR dependencies.
* **Cross-Chunk Store Identity:** Splitting multi-page flyers into 700-character chunks initially risked separating product prices from the store header at the top of the page. Prepending the store header line and tagging every chunk with `"store"` metadata preserved store attribution across every retrieved chunk.
* **Dense Semantic Search vs. Exact Keyword Matching:** Dense embeddings capture overall grocery topic similarity, meaning a general produce chunk can occasionally appear in Top-K even without an exact item keyword. Highlighting matched query terms inside **View Sources** and enforcing strict prompt grounding ensured only chunks containing confirmed prices were used in answers and charts.
