from sentence_transformers import SentenceTransformer, CrossEncoder

def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedding_model, reranker