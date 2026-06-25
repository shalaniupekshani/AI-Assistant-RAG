import numpy as np
import faiss
import ollama

def ask_question(question, index, metadata, embedding_model, reranker, TOP_K=10, FINAL_K=3):

    # =====================
    # QUERY EMBEDDING
    # =====================
    q_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(q_embedding, TOP_K)

    retrieved = []

    for idx in indices[0]:
        if idx < 0:
            continue
        retrieved.append(metadata[idx])

    # =====================
    # RERANKING
    # =====================
    pairs = [(question, item["text"]) for item in retrieved]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(retrieved, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_chunks = ranked[:FINAL_K]

    # =====================
    # BUILD CONTEXT
    # =====================
    context_parts = []
    sources_used = set()

    for item, score in top_chunks:
        sources_used.add(item["source"])

        context_parts.append(
            f"[Source: {item['source']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    # =====================
    # PROMPT
    # =====================
    prompt = f"""
You are a strict document assistant.

RULES:
- Use ONLY the context below
- If answer not found, say "Not found in documents"
- Be precise and short

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # =====================
    # LLM CALL
    # =====================
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response["message"]["content"],
        "sources": list(sources_used),
        "chunks": context_parts
    }