import streamlit as st
import faiss
import json

from backend.load_models import load_models
from backend.query_engine import ask_question

# =====================
# LOAD MODELS
# =====================
embedding_model, reranker = load_models()

# =====================
# LOAD INDEX + METADATA
# =====================
index = faiss.read_index("index/index.faiss")

with open("index/metadata.json", "r") as f:
    metadata = json.load(f)

# =====================
# UI
# =====================
st.set_page_config(page_title="RAG Assistant", layout="wide")

st.title("📚 AI Document Assistant (RAG System)")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("Ask your question:")

if user_input:
    result = ask_question(
        user_input,
        index,
        metadata,
        embedding_model,
        reranker
    )

    st.session_state.chat.append(
        (user_input, result["answer"], result["sources"])
    )

# =====================
# CHAT DISPLAY
# =====================
for q, a, s in st.session_state.chat:
    st.markdown("### 🧑 You")
    st.write(q)

    st.markdown("### 🤖 AI")
    st.write(a)

    st.markdown("**📌 Sources:** " + ", ".join(s))
    st.markdown("---")