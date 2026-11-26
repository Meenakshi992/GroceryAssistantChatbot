# grocery-products-assistant

**grocery-products-assistant** is a Surrey, BC grounded Retrieval-Augmented Generation (RAG) chatbot built with **Python**, **Streamlit**, **ChromaDB**, and **OpenAI (`text-embedding-3-small` & `gpt-4o-mini`)**. Instead of guessing answers from general internet knowledge, it answers shopper questions using **only** the grocery flyers, product catalogs, and store policies you upload—displaying separate per-store sentences with bold prices, per-item comparison tables, Matplotlib price-gap bar charts, and inline source citations while refusing to invent answers when supporting facts are not found.

---

## How the App Works (11-Step RAG Pipeline)

```text
Document -> Load -> Clean -> Chunk -> Embed -> Store -> Query -> Retrieve -> Filter -> Generate -> Cite
```

1. **Document:** Users upload at least 4 relevant text-based grocery documents (`.pdf`, `.txt`, or `.md`) in the sidebar (`sidebar_ui.py`).
2. **Load:** `extract_uploaded_file()` in `text_cleaner.py` reads uploaded file bytes using `uploaded_file.getvalue()` and extracts text page-by-page using `pypdf` for PDFs and UTF-8 decoding for `.txt`/`.md` files.
3. **Clean:** `clean_text()` in `text_cleaner.py` removes formatting noise before indexing—fixing hyphenated line wraps, removing standalone page numbers, joining broken lines, and collapsing extra spaces.
4. **Chunk:** `chunk_text()` and `index_uploaded_file()` in `vector_store.py` split cleaned text into overlapping sliding-window chunks (`chunk_size=700`, `overlap=120`) and attach `source`, `page`, `chunk_id`, and `store` metadata to each chunk.
5. **Embed:** Each chunk is converted into a 1,536-dimensional vector using OpenAI's `text-embedding-3-small` model, cached in `st.session_state["embedding_cache"]` via SHA-256 keys (`get_cached_embedding()`) to eliminate duplicate API calls.
6. **Store:** All chunks, embeddings, and metadata are saved into a persistent local **ChromaDB** database (`./chroma_rag_db_GAC`) so documents stay available across app restarts without re-indexing.
7. **Query:** When a user submits a question in the chat interface (`app.py`), `get_cached_embedding()` converts the question into a query vector using `text-embedding-3-small`.
8. **Retrieve:** `retrieve_and_answer()` in `chat_engine.py` searches ChromaDB—filtered by the user's validated Surrey postal code radius (`allowed_stores`)—and retrieves the **Top-K** (default `4`, range `2–5`) most similar chunks along with cosine distance scores.
9. **Filter:** Cosine distance is converted to similarity (`1 - distance`), classified (`Strongly Relevant`, `Potentially Relevant`, `Weakly Relevant`, or `Irrelevant`), and checked against the **Similarity Cutoff** (`0.35`). If zero chunks pass, generation stops and a safe fallback message is returned.
10. **Generate:** Accepted chunks are sent along with the user's question and strict anti-hallucination rules (`temperature=0.0`) to `gpt-4o-mini` to generate separate per-store lines with bold prices (`**\$X.XX**`) and structured price data for per-item tables (`show_item_tables()`) and `matplotlib` price-gap bar charts (`show_price_chart()`).
11. **Cite:** The assistant includes inline citations (`[Source: filename, p. 1]`) after every claim, and users can expand **View Sources** under any reply to inspect the exact retrieved text with yellow highlighted query terms, page number, chunk ID, similarity score, and cosine distance.

---

## Design Choices & Settings

* **Chunk Size (`700`) & Overlap (`120`):** Grocery flyers and store policies consist of short product deals and compact policy clauses. A 700-character sliding-window chunk keeps a full item deal or policy rule together without diluting the embedding vector, while a 120-character overlap prevents prices or conditions from getting cut off across boundaries.
* **Top-K (`4`, Range `2–5`) & Natural Semantic Retrieval:** Retrieving 4 chunks by default keeps context focused and fast, ranking all matching chunks across nearby store flyers and policy documents by pure cosine similarity.
* **Similarity Cutoff (`0.35`):** Testing showed that valid grocery questions score above `0.38` (even with spelling mistakes), while unrelated questions score below `0.25`. Setting the cutoff at `0.35` cleanly rejects out-of-scope questions.
* **Surrey, BC Postal Code Validation & Radius Filter (`store_locator.py`):** Validates Canadian postal code syntax and Surrey FSA prefixes (`V3R`, `V3S`, `V3T`, `V3V`, `V3W`, `V3X`, `V3Z`, `V4A`, `V4N`, `V4P`), locates the closest branch per chain (**Walmart Supercentre — Central City Mall**, **No Frills — 148 St**, **Real Canadian Superstore**, **Save-On-Foods**, **FreshCo**, **Safeway**, and **Thrifty Foods**), and restricts ChromaDB retrieval strictly to stores within the chosen radius.

---

## Project Structure

```text
grocery-products-assistant/
├── .streamlit/
│   └── secrets.toml             # Local OpenAI API key (excluded from Git via .gitignore)
├── .gitignore                   # Keeps secrets.toml and chroma_rag_db_GAC/ off GitHub
├── sidebar_ui.py                # Sidebar upload controls, cleaning options, pincode & KB status
├── text_cleaner.py              # Text extraction (pypdf, UTF-8) and text cleaning
├── vector_store.py              # Overlapping chunking, SHA-256 embedding cache & ChromaDB
├── chat_engine.py               # Radius-filtered semantic search, cutoff & grounded generation
├── store_locator.py             # Surrey, BC postal code validation & nearest branch locator
├── app.py                       # Main Streamlit UI, per-item tables, Matplotlib charts & source drawer
├── sample.py                    # Zero-dependency script to generate 8 sample Surrey PDFs in root
├── generate_test_report.py      # Generates formatted multi-page Presentation_Test_Report.pdf
├── requirements.txt             # Pinned Python dependencies
└── README.md                    # Architecture documentation, rubric test log & reflections
```

---

## How to Run Locally

1. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Add your OpenAI API key:**
   Create `.streamlit/secrets.toml` and add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
3. *(Optional)* **Generate sample Surrey flyers & policy PDFs:**
   ```bash
   python sample.py
   ```
4. **Start the app:**
   ```bash
   python -m streamlit run app.py
   ```
5. **Index documents:** Upload at least 4 text-based grocery `.pdf`, `.txt`, or `.md` files in the sidebar, choose your cleaning options, and click **Index Uploaded Documents**.

---

## 3-Question Live Demo Highlight

1. **Multi-Store Comparison & Visuals:** *"Compare the prices of Dairyland 2% Milk, Large White Eggs, and Bananas across my nearby stores."*
2. **Cross-Document Policy & Raincheck Rules:** *"How long do I have to use a raincheck before it expires, and can Walmart at Central City Mall price match No Frills?"*
3. **Out-of-Scope Anti-Hallucination Guardrail:** *"What are the SkyTrain Expo Line operating hours from Surrey Central to Waterfront?"*

---

## Rubric-Aligned Test Log

Tested using the Surrey grocery PDF documents (`no_frills_weekly_flyer.pdf`, `walmart_surrey_central_flyer.pdf`, `superstore_weekly_flyer.pdf`, `save_on_foods_weekly_flyer.pdf`, `freshco_weekly_flyer.pdf`, `safeway_weekly_flyer.pdf`, `thrifty_foods_weekly_flyer.pdf`, and `surrey_price_match_and_raincheck_policy.pdf`) with **Top-K = 4**, **Similarity Cutoff = 0.35**, and **Active Week: November 26 – December 2, 2026**.

| # | Rubric Criterion | Test Question / Action | Result & Verification | Status |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Direct Fact Retrieval & Citation | How much is a 10 kg bag of Rogers Granulated White Sugar at No Frills? | Retrieved exact chunk above `0.35` cutoff and cited `[Source: no_frills_weekly_flyer.pdf, p. 1]`. | **PASSED** |
| **2** | Paraphrased / Semantic Query | How long do I have to use a raincheck before it expires, and where can I redeem it? | Dense embeddings matched *"before it expires"* to the 14-day window at the issuing Surrey store + 10% PC Optimum points `[Source: surrey_price_match_and_raincheck_policy.pdf, p. 1]`. | **PASSED** |
| **3** | Misspelled / Noisy Input | Wich store near me has the lowest cost on basmatty rice, spagetti noodles, and canola oil? | Handled spelling typos (`basmatty`, `spagetti`) above `0.35` cutoff and retrieved Tilda Rice, Barilla Spaghetti, and Canola Oil with citations. | **PASSED** |
| **4** | Multi-Document Synthesis | Compare the prices of Dairyland 2% Milk, Large White Eggs, Chicken Breasts, and Bananas across my nearby stores. | Combined chunks across multiple store flyers into separate per-store sentences with bold prices, per-item tables, and Matplotlib price-gap chart. | **PASSED** |
| **5** | Follow-Up Reasoning | Tell me which store has the cheapest price on eggs among those stores. | Correctly identified **No Frills** as the lowest-priced option on Large White Eggs (18 Pack Carton) with exact savings gap and citation. | **PASSED** |
| **6** | Policy & Exclusion Edge Case | Can I price match Safeway's Scene+ member price or Save-On-Foods' More Rewards price at Superstore, and can Walmart at Central City Mall price match No Frills? | Correctly answered **No** to both based on fine-print Ad Match rules in `surrey_price_match_and_raincheck_policy.pdf`. | **PASSED** |
| **7** | Metadata Radius Filter | Set Postal Code = `V3T 2W1` (Whalley / Surrey Central) & Radius = `1.5 km`, then ask for milk & banana prices. | ChromaDB metadata `where` filter restricted retrieval strictly to **Walmart (Central City)** and **Superstore** within `1.5 km`. | **PASSED** |
| **8** | Input Validation | Enter invalid postal code `12345` or non-Surrey code `V6B 1A1` in the Surrey Postal Code field. | Input validation blocked malformed and out-of-area postal codes cleanly with a red error banner listing valid Surrey FSA prefixes. | **PASSED** |
| **9** | Out-of-Scope Guardrail | What are the SkyTrain Expo Line operating hours from Surrey Central to Waterfront? | Cosine similarity fell below `0.35` cutoff; stopped before calling `gpt-4o-mini` and returned safe fallback message with zero hallucination. | **PASSED** |
| **10** | Caching & Persistence | Ask the same comparison question **4 times** and restart Streamlit once. | SHA-256 `embedding_cache` reused query vectors (`0 ms` repeat embedding calls), and all 8 documents reloaded from `./chroma_rag_db_GAC` after restart. | **PASSED** |

---

## Project Reflections & Lessons Learned

### 1. What Worked Well
* **Strict Grounding & Hallucination Prevention:** Combining a cosine similarity cutoff (`0.35`) with deterministic generation (`temperature=0.0`), strict rules against guessing unclear regular price ranges (e.g., `reg 3.69-3.79`), and inline `[Source: filename, p. 1]` citations prevented the LLM from guessing prices or answering out-of-scope questions.
* **Location-Restricted RAG (`V3R 0A4` / `V3T 2W1` in Surrey, BC):** Validating Surrey postal codes and passing nearby stores directly into ChromaDB's metadata `where` clause ensured that price comparisons only pulled deals from stores within the user's chosen travel radius.
* **Visual Price Gap Charts & Scannable Layout:** Formatting each store on a separate line with bold prices (`**\$X.XX**`), paired with per-item tables and grouped `matplotlib` bar charts (`Gap: \$X.XX`), made multi-store price differences immediately scannable.
* **Performance & Persistence:** Persisting vectors on disk (`./chroma_rag_db_GAC`) alongside a SHA-256 session `embedding_cache` kept query retrieval fast (~150–300 ms) and eliminated duplicate OpenAI embedding calls across repeated queries and restarts.

### 2. Technical Challenges Solved
* **Text-Based vs. Image-Only Flyers:** Commercial grocery flyers (such as web flyer viewers on Flipp) often flatten prices into images rather than selectable text. Focusing the ingestion pipeline on fast, clean text extraction (`pypdf` and UTF-8 `.txt`/`.md`) with clear user guidance kept indexing fast, lightweight, and reliable without heavy OCR dependencies.
