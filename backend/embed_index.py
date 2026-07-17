from pathlib import Path
from pypdf import PdfReader
import numpy as np
import faiss #store vectors and do fast neighbour search
import json #metadata saved in jason file

from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.load_models import load_models #Loading functions such as embedded model or LLM


PDF_FOLDER = Path("data") #path for the pdf folder
INDEX_PATH = "index/index.faiss"
META_PATH = "index/metadata.json" # It stores information about every chunk.
Path("index").mkdir(exist_ok=True)  # prevent errors if not exist

CHUNK_SIZE = 500 #500 chracters
CHUNK_OVERLAP = 100 # adjacent chunks share 100 characters


def build_index():

    embedding_model, _ = load_models() # Only the embedded model loads

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    metadata = []

    pdf_files = list(PDF_FOLDER.glob("*.pdf")) # find every pdf
    

    for pdf_file in pdf_files:
        reader = PdfReader(str(pdf_file))

        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

        full_text = "\n".join(full_text).strip() # remove whitespaces

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

    #Converting text chunks into vectors
    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True #Normalizes vectors to have unit length, making cosine similarity equivalent to inner product.
    ).astype("float32")  #FAISS expected float32

    dimension = embeddings.shape[1] 

    index = faiss.IndexFlatIP(dimension) #Creates an index that uses inner product for similarity search.
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    return len(pdf_files), len(chunks)


if __name__ == "__main__":
    pdfs, chunks = build_index()
    print(f"Index built successfully!")
    print(f"PDFs indexed: {pdfs}")
    print(f"Chunks created: {chunks}")