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
8. **Retrieve:** `retrieve_and_answer()` in `chat_engine.py` searches ChromaDB—filtered by the user's selected Surrey postal code radius (`allowed_stores`) and policy-keyword routing—and retrieves the **Top-K** (default `4`) most similar chunks along with cosine distance scores.
9. **Filter:** Cosine distance is converted to similarity (`1 - distance`), classified (`Strongly Relevant`, `Potentially Relevant`, `Weakly Relevant`, or `Irrelevant`), and checked against the **Similarity Cutoff** (`0.35`). If zero chunks pass, generation stops and a safe fallback message is returned.
10. **Generate:** Accepted chunks are sent along with the user's question and strict anti-hallucination rules (`temperature=0.0`) to `gpt-4o-mini` to generate a concise summary and structured price data for per-item tables (`show_item_tables()`) and `matplotlib` price-gap bar charts (`show_price_chart()`).
11. **Cite:** The assistant includes inline citations (`[Source: filename, p. 1]`) after every claim, and users can expand **View Sources** under any reply to inspect the exact retrieved text with highlighted query terms, page number, chunk ID, similarity score, and cosine distance.

---

## Design Choices & Settings

* **Chunk Size (`700`) & Overlap (`120`):** Grocery flyers and store policies consist of short product deals and compact policy clauses. A 700-character sliding-window chunk keeps a full item deal or policy rule together without diluting the embedding vector, while a 120-character overlap prevents prices or conditions from getting cut off across boundaries.
* **Top-K (`4`):** Retrieving up to 5 chunks captures matching product deals across nearby Surrey store flyers while keeping context focused and fast.
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
generate_test_report.py      # ├──Generates formatted multi-page Presentation_Test_Report.pdf
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
Tested using the Surrey grocery PDF documents (`no_frills_weekly_flyer.pdf`, `walmart_surrey_central_flyer.pdf`, `superstore_weekly_flyer.pdf`, `save_on_foods_weekly_flyer.pdf`, `freshco_weekly_flyer.pdf`, `safeway_weekly_flyer.pdf`, `thrifty_foods_weekly_flyer.pdf`, and `surrey_price_match_and_raincheck_policy.pdf`) with **Top-K = 4**, **Similarity Cutoff = 0.35**, and **Active Week: November 26 – December 2, 2026**.
** Each of the 1-page sample PDFs (no_frills_weekly_flyer.pdf, walmart_surrey_central_flyer.pdf, superstore_weekly_flyer.pdf) is ~900 characters long, which splits into only 2 chunks per PDF. **
| # | Test Category | Test Question / Input | Retrieved Right Chunks? | Accurate Answer? | Correct Citation? | Rejected Invalid / Irrelevant? | Result / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Direct Fact Lookup | How much is a 10 kg bag of Rogers Granulated White Sugar at No Frills? | Yes | Yes (**\$13.99** each) | Yes `[Source: no_frills_weekly_flyer.pdf, p. 1]` | N/A | Strongly Relevant match with bold price formatting. |
| **2** | Direct Fact Lookup | What is the price of Tilda Pure Basmati Rice (4.5 kg) at Walmart Central City Mall? | Yes | Yes (**\$12.97** Special Import Price) | Yes `[Source: walmart_surrey_central_flyer.pdf, p. 1]` | N/A | Verified updated Central City Mall (`10153 King George Blvd`) branch. |
| **3** | Paraphrased Policy | How long do I have to use a raincheck before it expires, and where can I redeem it? | Yes | Yes (Within 14 days at the issuing Surrey store + 10% back in PC Optimum points at No Frills/Superstore) | Yes `[Source: surrey_price_match_and_raincheck_policy.pdf, p. 1]` | N/A | Matched "before it expires" to "valid for 14 days". |
| **4** | Spelling Typos | Wich store near me has the lowest cost on basmatty rice, spagetti noodles, and canola oil? | Yes | Yes (Tilda Basmati Rice, Barilla Spaghetti, and Great Value Canola Oil retrieved accurately) | Yes (Cites matching store flyer PDFs, p. 1) | N/A | Embeddings handled typos above 0.35 cutoff. |
| **5** | Multi-Store Comparison | Compare the prices of Dairyland 2% Milk, Large White Eggs, Chicken Breasts, and Bananas across my nearby stores. | Yes | Yes (Displays separate per-store lines with bold prices, per-item tables & savings gaps) | Yes (Cites all matching flyer PDFs, p. 1) | N/A | Rendered per-item tables + grouped Matplotlib `Gap: \$X.XX` chart. |
| **6** | Raincheck Exception Trap | If No Frills runs out of No Name Apples before I get there, can I get a raincheck and earn PC Optimum points? | Yes | Yes (**No** — No Name Apples are marked *"while quantities last"*, which is excluded from rainchecks) | Yes `[Source: no_frills_weekly_flyer.pdf, p. 1]` & `[Source: surrey_price_match_and_raincheck_policy.pdf, p. 1]` | N/A | Balanced `2 flyer + 2 policy` retrieval caught the item-level exclusion. |
| **7** | Non-Participating Store Trap | Can I bring the No Frills flyer to Walmart at Central City Mall to get a price match at the register? | Yes | Yes (**No** — Walmart Central City Mall does not offer competitor price matching at the register) | Yes `[Source: surrey_price_match_and_raincheck_policy.pdf, p. 1]` | N/A | Cross-checked store participation list accurately. |
| **8** | Loyalty Price Exclusion Trap | Can I price match Safeway's Scene+ member price or Save-On-Foods' More Rewards price at Superstore? | Yes | Yes (**No** — member-only loyalty prices such as Scene+ and More Rewards are excluded from Ad Match) | Yes `[Source: surrey_price_match_and_raincheck_policy.pdf, p. 1]` | N/A | Caught fine-print Ad Match exclusion rule. |
| **9** | Freshness Guarantee Trap | I bought a pre-packaged salad kit four days ago and its Best Before date passed yesterday — is it covered? | Yes | Yes (**No** — pre-packaged salad kits past their printed Best Before date are excluded from the 7-day guarantee) | Yes `[Source: surrey_price_match_and_raincheck_policy.pdf, p. 1]` | N/A | Correctly applied policy exception reasoning. |
| **10** | Surrey Radius Filter | Set Postal Code = `V3T 2W1` (Surrey Central) & Radius = `1.5 km`, then ask for milk & banana prices. | Yes | Yes (Includes only Walmart Central City & Superstore; excludes stores outside 1.5 km) | Yes `[Source: walmart_surrey_central_flyer.pdf, p. 1]` & `[Source: superstore_weekly_flyer.pdf, p. 1]` | N/A | ChromaDB `where` filter restricted results strictly by travel distance. |
| **11** | Postal Code Validation | Enter `12345` or `V6B 1A1` (Vancouver) in the Surrey Postal Code input. | N/A | Yes (Displays red validation error listing valid Surrey FSA prefixes) | N/A | **Yes** | Blocked malformed and non-Surrey postal codes cleanly. |
| **12** | Out-of-Scope Guardrail | What are the SkyTrain Expo Line operating hours from Surrey Central to Waterfront? | No (0 passed) | Yes (Returned safe fallback message) | N/A (Refused) | **Yes** | Cosine similarity fell below 0.35; refused to hallucinate. |
| **13** | check embedding cache logic => what is the price of eggs in all nearby store and which has least price of eggs among above stores

## Project Reflections & Lessons Learned

### 1. What Worked Well
* **Strict Grounding & Hallucination Prevention:** Combining a cosine similarity cutoff (`0.35`) with deterministic generation (`temperature=0.0`), strict rules against guessing unclear regular price ranges (e.g., `reg 3.69-3.79`), and inline `[Source: filename, p. 1]` citations prevented the LLM from guessing prices or answering out-of-scope questions.
* **Location-Restricted RAG (`V3R 0A4` / Surrey, BC):** Integrating postal code distance filtering directly into ChromaDB's metadata `where` clause ensured that price comparisons only pulled deals from stores within the user's chosen travel radius rather than across the entire city.
* **Visual Price Gap Charts & Per-Item Tables:** Converting retrieved flyer prices into separate item tables and grouped `matplotlib` bar charts with `Gap: \$X.XX` callouts made multi-store price differences clear and scannable.
* **Performance & Persistence:** Persisting vectors on disk (`./chroma_rag_*****`) alongside a SHA-256 session `embedding_cache` kept query retrieval fast (~150–300 ms) and eliminated duplicate OpenAI embedding calls across restarts.
