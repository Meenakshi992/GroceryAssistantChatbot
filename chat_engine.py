from datetime import datetime
import json
import time
from vector_store import DEFAULT_EMBEDDING_MODEL, get_cached_embedding

DEFAULT_GENERATION_MODEL = "gpt-4o-mini"
DEFAULT_FALLBACK1 = "I could not find enough relevant information in the uploaded documents for your nearby stores to answer that question reliably."
DEFAULT_FALLBACK2 = "I cannot answer that question because no uploaded flyer data was found for your nearby stores within the selected radius. Please upload their weekly flyer PDF or screenshot in the sidebar, or increase your Search Radius."

def get_relevance_label(similarity, threshold):
    if similarity >= threshold + 0.15:
        return "Strongly Relevant"
    elif similarity >= threshold:
        return "Potentially Relevant"
    elif similarity >= threshold - 0.10:
        return "Weakly Relevant"
    else:
        return "Irrelevant"


def retrieve_and_answer(client, collection, question, top_k=5, threshold=0.35, allowed_stores=None, include_upcoming=False):
    """Searches ChromaDB within radius, generates a cited summary, and extracts grounded price data for charts."""
    start_time = time.time()

    if collection.count() == 0:
        return {
            "answer": "The knowledge base is empty. Please upload at least 4 documents first.",
            "sources":[],
            "chart_data":[],
            "metrics": {"retrieval_ms": 0.0, "total_ms": 0.0},
        }

    if allowed_stores is not None and len(allowed_stores) == 0:
        return {
            "answer": "No partner grocery stores were found within your selected radius. Increase the Radius slider in the sidebar.",
            "sources":[],
            "chart_data":[],
            "metrics": {"retrieval_ms": 0.0, "total_ms": 0.0},
        }

    query_vec = get_cached_embedding(client, question, DEFAULT_EMBEDDING_MODEL)
    num_results = min(top_k, collection.count())

    where_filter = None

    if allowed_stores is not None:
        valid_tags = list(allowed_stores) + ["Policy"]
        where_filter = {"store": {"$in": valid_tags}}

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=num_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    retrieval_ms = round((time.time() - start_time) * 1000, 1)
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    sources = []
    accepted = []

    for i in range(len(docs)):
        distance = round(float(dists[i]), 3)
        similarity = round(1.0 - float(dists[i]), 3)
        label = get_relevance_label(similarity, threshold)
        page_num = metas[i]["page"]

        record = {
            "source": metas[i]["source"],
            "page": page_num if page_num > 0 else None,
            "chunk_id": metas[i]["chunk_id"],
            "distance": distance,
            "similarity": similarity,
            "label": label,
            "text": docs[i],
        }
        sources.append(record)

        if similarity >= threshold:
            accepted.append(record)

    if len(accepted) == 0:
        total_ms = round((time.time() - start_time) * 1000, 1)
        if allowed_stores:
            store_names = ", ".join(allowed_stores)
            fallback_msg = (
                f"I cannot answer that question because no uploaded flyer data was found for your nearby stores "
                f"within the selected radius ({store_names}). Please upload their weekly flyer PDF or screenshot in the sidebar, "
                "or increase your Search Radius."
            )
        else:
            fallback_msg = DEFAULT_FALLBACK2

        return {
            "answer": fallback_msg,
            "sources": sources,
            "metrics": {"retrieval_ms": retrieval_ms, "total_ms": total_ms},
        }

    context_text = ""
    for chunk in accepted:
        page_str = f", p. {chunk['page']}" if chunk["page"] is not None else ""
        context_text += f"[Source: {chunk['source']}{page_str}]\n{chunk['text']}\n\n"

    system_prompt = (
        "You are grocery-products-assistant. Answer the user's question using ONLY the provided context. "
        "STRICT ANTI-HALLUCINATION RULES:\n"
        "1. Never guess, estimate, or assume a single sale price if the text only shows a regular price range (e.g. 'reg 3.69-3.79') or if the sale price is unclear/missing.\n"
        "2. If an item's exact sale price is unclear or only a regular price range is listed, explicitly state that the exact sale price is not clearly listed in the flyer and quote the exact text.\n"
        "Format your 'answer' so that each store's information is on its own separate line (separated by a blank line) as a clear, standalone sentence, with every price wrapped in bold markdown (for example: **$5.39**). Include the cheapest store, exact price gap, and flyer valid dates.\n"
        "4. Always include inline citations like [Source: filename, p. 1] after your facts.\n"
        "Return a valid JSON object with two keys:\n"
        '1. "answer": your concise cited text summary.\n'
        '2. "prices": a list of objects {"item": "Short Item Name", "store": "Store Name", "price": 5.39} '
        "ONLY for items with a single confirmed price (empty list if none apply)."
    )

    response = client.chat.completions.create(
        model=DEFAULT_GENERATION_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\nQuestion: {question}"},
        ],
    )
    first_choice = response.choices.pop(0)
    raw_json = first_choice.message.content or "{}"

    parsed = json.loads(raw_json)
    answer = parsed.get("answer", DEFAULT_FALLBACK1)
    answer = answer.replace("$", r"\$")
    chart_data = parsed.get("prices",)

    total_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "answer": answer,
        "sources": sources,
        "chart_data": chart_data,
        "metrics": {"retrieval_ms": retrieval_ms, "total_ms": total_ms},
    }

