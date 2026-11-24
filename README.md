# grocery-products-assistant

**grocery-products-assistant** is a Retrieval-Augmented Generation (RAG) chatbot built with **Python**, **Streamlit**, **ChromaDB**, and **OpenAI**. Instead of guessing answers from general internet knowledge, it answers questions using **only** the grocery flyers, product catalogs, and store policies you upload—providing inline source citations and refusing to invent answers when supporting facts are not found.

---

## How the App Works (11-Step RAG Pipeline)

```text
Document -> Load -> Clean -> Chunk -> Embed -> Store -> Query -> Retrieve -> Filter -> Generate -> Cite
```

1. **Document:** You upload at least 4 relevant grocery documents (`.pdf` or `.txt`) in the sidebar (`app.py`).
2. **Load:** `load_and_clean_file()` in `text_cleaner.py` reads each uploaded file page by page using `PyMuPDF` (`fitz`) for PDFs or standard UTF-8 reading for `.txt` files.
3. **Clean:** `clean_text()` in `text_cleaner.py` removes formatting noise before indexing—fixing hyphenated line breaks, removing standalone page numbers, joining broken sentences across lines, and collapsing extra spaces.
4. **Chunk:** `save_document_to_db()` in `vector_store.py` uses `RecursiveCharacterTextSplitter` to break cleaned text into overlapping chunks (`chunk_size=700`, `chunk_overlap=120`) and attaches metadata (`source`, `page`, and `chunk_id`) to each chunk.
5. **Embed:** Each chunk is converted into a numerical vector using OpenAI's `text-embedding-3-small` model.
6. **Store:** All chunks, embeddings, and metadata are saved into a persistent local **ChromaDB** database (`./chroma_rag_db`) so documents are embedded once and reused across sessions.
7. **Query:** When you type a question in the chat interface, `text-embedding-3-small` converts your question into a query vector.
8. **Retrieve:** `ask_knowledge_base()` in `chat_engine.py` searches ChromaDB and retrieves the **Top-K** (default `4`) most similar chunks along with their distance scores.
9. **Filter:** Each distance score is converted into a similarity score (`1 - distance`) and compared against the **Similarity Threshold** (`0.35`). If no chunks pass the threshold, the app stops and returns a safe fallback message instead of guessing.
10. **Generate:** Chunks that pass the threshold are sent along with your question and strict grounding instructions to `gpt-4o-mini` to write a clear answer.
11. **Cite:** The assistant includes inline citations (`[Source: filename, p. 1]`) after its facts, and you can expand **View Sources** under any reply to inspect the exact retrieved text, page number, chunk ID, and similarity score.

---

## Design Choices & Settings

* **Chunk Size (`700`) & Overlap (`120`):** Grocery flyers and store policies consist of short bullet points and compact paragraphs. A 700-character chunk keeps a full product deal or policy rule together without mixing unrelated categories, while a 120-character overlap prevents prices or conditions from getting cut off at chunk edges.
* **Top-K (`4`):** Retrieving 4 chunks allows the assistant to combine information across different documents (such as comparing two store flyers or connecting a flyer item with a raincheck rule).
* **Similarity Threshold (`0.35`):** Testing showed that valid grocery questions score above `0.38` (even with spelling mistakes), while unrelated questions score below `0.25`. Setting the threshold at `0.35` cleanly rejects out-of-scope questions.

---

## Project Structure

```text
grocery-products-assistant/
├── .streamlit/
│   └── secrets.toml        # Stores OPENAI_API_KEY locally (ignored by Git)
├── .gitignore              # Keeps secrets.toml and chroma_rag_db/ off GitHub
├── text_cleaner.py         # File loading and text cleaning functions
├── vector_store.py         # Text chunking, embeddings, and ChromaDB storage
├── chat_engine.py          # Semantic search, threshold filter, and OpenAI generation
├── app.py                  # Streamlit chat interface and sidebar controls
├── requirements.txt        # Required Python libraries
└── README.md               # Project architecture and 10-question test log
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
3. **Fresh Start the app:**
   ```bash
   python -m streamlit run app.py
   ```
4. **Index documents:** Upload at least 4 grocery `.pdf` or `.txt` files in the sidebar, choose your cleaning options, and click **Index Documents**.

---

## 10-Question Test Log

Tested using 4 grocery documents (`no_frills_weekly_flyer.txt`, `walmart_guildford_flyer.txt`, `raincheck_and_loyalty_policy.txt`, `fresh_produce_guarantee_policy.txt`) with **Top-K = 4** and **Similarity Threshold = 0.35**.

| # | Question Type | Test Question | Retrieved Right Chunks? | Accurate Answer? | Correct Citation? | Rejected Irrelevant Question? | Result / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | Direct | What is the price per kg of Eye of Round Roast at No Frills? | Yes | Yes ($12.96 / kg) | Yes `[Source: no_frills_weekly_flyer.txt, p. 1]` | N/A | Strongly Relevant match. |
| **2** | Direct | How much is Dairyland 2% Milk (4L) at Walmart Guildford? | Yes | Yes ($5.47 Rollback, limit 2) | Yes `[Source: walmart_guildford_flyer.txt, p. 1]` | N/A | Found exact price and limit. |
| **3** | Direct | How many days do I have to return fresh produce under the Freshness Guarantee? | Yes | Yes (7 days with receipt) | Yes `[Source: fresh_produce_guarantee_policy.txt, p. 1]` | N/A | Found exact return window. |
| **4** | Paraphrased | If a sale item is sold out on the shelf, what do I get? | Yes | Yes (A raincheck for the flyer price + 10% back in PC Optimum points within 14 days) | Yes `[Source: raincheck_and_loyalty_policy.txt, p. 1]` | N/A | Matched "sold out" to "out of stock". |
| **5** | Paraphrased | Are bagged salads covered if they go bad within a week? | Yes | Yes (No, items with a printed Best Before date like bagged salad kits are excluded) | Yes `[Source: fresh_produce_guarantee_policy.txt, p. 1]` | N/A | Caught policy exclusion accurately. |
| **6** | Misspelled | Wat is the price of chiken brest and banannas at Walmrt? | Yes | Yes (Chicken breast $11.98/kg, Bananas $1.50/kg) | Yes `[Source: walmart_guildford_flyer.txt, p. 1]` | N/A | Embeddings handled typos above 0.35 threshold. |
| **7** | Multi-Document | Can I get a raincheck if No Frills runs out of Eye of Round Roast? | Yes | Yes (No, because Eye of Round Roast is marked "while quantities last," which is excluded from rainchecks) | Yes `[Source: no_frills_weekly_flyer.txt, p. 1]` & `[Source: raincheck_and_loyalty_policy.txt, p. 1]` | N/A | Combined item label from flyer with rule from policy file. |
| **8** | Multi-Document | Compare the meat deals at No Frills and Walmart Guildford. | Yes | Yes (Lists beef/pork/ribs at No Frills and chicken breasts at Walmart) | Yes (Cites both flyer files, p. 1) | N/A | Combined chunks across two store flyers. |
| **9** | Irrelevant | How do I replace the alternator on a 2018 Honda Civic? | No (0 passed) | Yes (Returned safe fallback message) | N/A (Refused) | **Yes** | Score fell below 0.35; refused to invent an answer. |
| **10** | Irrelevant | What is the current stock price of Apple Inc.? | No (0 passed) | Yes (Returned safe fallback message) | N/A (Refused) | **Yes** | Score fell below 0.35; refused to invent an answer. |
