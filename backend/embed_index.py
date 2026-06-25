from pathlib import Path
from pypdf import PdfReader
import numpy as np
import faiss
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.load_models import load_models

# ======================
PDF_FOLDER = Path("data")
INDEX_PATH = "index/index.faiss"
META_PATH = "index/metadata.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ======================
def build_index():

    embedding_model, _ = load_models()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    metadata = []

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_file in pdf_files:
        reader = PdfReader(str(pdf_file))

        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

        full_text = "\n".join(full_text).strip()

        if not full_text:
            continue

        file_chunks = splitter.split_text(full_text)

        for i, chunk in enumerate(file_chunks):
            chunk = chunk.strip()
            if len(chunk) < 20:
                continue

            chunks.append(chunk)
            metadata.append({
                "source": pdf_file.name,
                "chunk_id": i,
                "text": chunk
            })

    print(f"Total chunks: {len(chunks)}")

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    print("Index built successfully!")


if __name__ == "__main__":
    build_index()