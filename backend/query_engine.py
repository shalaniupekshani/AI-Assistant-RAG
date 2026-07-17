import numpy as np
import faiss
from groq import Groq
import os

def ask_question(question, index, metadata, embedding_model, reranker, TOP_K=10, FINAL_K=3):

    #Convert user question into vector
    q_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).astype("float32")

    #Search FAISS
    distances, indices = index.search(q_embedding, TOP_K)

    retrieved = []

    # for meta data
    for idx in indices[0]:
        if idx < 0:
            continue
        retrieved.append(metadata[idx])

    #create pairs for the CrossEncoder (question, document)
    pairs = [(question, item["text"]) for item in retrieved]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(retrieved, scores), # Combine retrieved and score into (chunk1,0.98)(chunk2,0.61)
        key=lambda x: x[1], # sort according to score
        reverse=True
    )

    top_chunks = ranked[:FINAL_K]

   #Context build
    context_parts = []
    sources_used = set()

    for item, score in top_chunks:
        sources_used.add(item["source"]) #store file name

        context_parts.append(
            f"[Source: {item['source']}]\n{item['text']}" #Append formatted text
        )

    context = "\n\n".join(context_parts)

    # Prompt
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

    
    # LLM call
    client = Groq(
        api_key=os.getenv("GROQ_API_KEY")
    )

    if not client.api_key:
        raise ValueError("GROQ_API_KEY is missing")


    response = client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )


    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": list(sources_used),
        "chunks": context_parts
    }