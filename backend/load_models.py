#SentenceTransformer - create embeddings which is vector representation of text
#CrossEncoder - Re-ranks search results to find the most relevant ones
from sentence_transformers import SentenceTransformer, CrossEncoder

def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedding_model, reranker