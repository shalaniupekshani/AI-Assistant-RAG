# DocuMate — AI Document Assistant

<p align="center">
  <b>Retrieval-Augmented Generation (RAG) based AI assistant for intelligent document querying</b>
</p>

---

## 📌 Overview

**DocuMate** is an AI-powered document assistant that enables users to interact with their PDF documents using natural language queries.

The system implements a complete **Retrieval-Augmented Generation (RAG)** pipeline by combining semantic search, vector databases, reranking models, and Large Language Models (LLMs) to provide accurate, context-aware responses based only on uploaded documents.

The application allows users to upload documents, build a searchable knowledge base, and ask questions through a modern chat-based interface.

---

## ✨ Key Features

- 📄 **PDF Document Processing**
  - Extracts text from uploaded PDF files
  - Splits large documents into optimized chunks

- 🔎 **Semantic Document Retrieval**
  - Generates document embeddings using Sentence Transformers
  - Performs fast similarity search using FAISS vector database

- 🎯 **Retrieval Optimization**
  - Uses CrossEncoder reranking to improve retrieved context relevance

- 🤖 **LLM-Based Answer Generation**
  - Uses Llama 3.1 through Groq API
  - Generates answers based on retrieved document context

- 💬 **Interactive Chat Interface**
  - Modern Streamlit-based conversational UI
  - Supports multiple document uploads

- ☁️ **Cloud Deployment**
  - Deployed using Streamlit Cloud

---

# 🛠️ Technology Stack

## Programming Language

- Python

## Frontend

- Streamlit

## AI / NLP

- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- CrossEncoder Models
- Large Language Models (LLMs)

## Vector Database

- FAISS (Facebook AI Similarity Search)

## Document Processing

- PyPDF
- LangChain Text Splitters

## LLM Provider

- Llama 3.1 via Groq API

## Deployment

- Streamlit Cloud

---

# 📂 Project Structure

```
AI-Assistant-RAG
│
├── backend/
│   ├── embed_index.py
│   │       └── Document embedding and FAISS index generation
│   │
│   ├── load_models.py
│   │       └── Loading embedding and reranking models
│   │
│   └── query_engine.py
│           └── RAG retrieval and LLM response generation
│
├── ui/
│   └── app.py
│           └── Streamlit user interface
│
├── data/
│   └── Uploaded PDF documents
│
├── index/
│   ├── index.faiss
│   └── metadata.json
│
├── requirements.txt
└── README.md
```

---

# ⚙️ How It Works

### 1. Document Ingestion

Users upload PDF documents through the application.

### 2. Text Processing

The extracted text is divided into smaller overlapping chunks to improve retrieval performance.

### 3. Embedding Generation

Each chunk is converted into a numerical vector representation using a Sentence Transformer model.

### 4. Vector Search

FAISS performs similarity search to retrieve the most relevant document sections.

### 5. Reranking

A CrossEncoder model evaluates retrieved chunks and selects the most relevant context.

### 6. Response Generation

The selected context is provided to Llama 3.1, which generates a final answer grounded in the uploaded documents.

---

# 🚀 Installation & Usage

## Clone Repository

```bash
git clone https://github.com/shalaniupekshani/AI-Assistant-RAG.git
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
streamlit run ui/app.py
```

---

# 📊 RAG Pipeline

| Component | Technology |
|---|---|
| Document Extraction | PyPDF |
| Text Chunking | LangChain Recursive Splitter |
| Embedding Model | Sentence Transformer |
| Vector Database | FAISS |
| Retrieval Enhancement | CrossEncoder |
| Language Model | Llama 3.1 |
| Interface | Streamlit |

---

