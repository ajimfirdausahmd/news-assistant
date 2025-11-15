from typing import List, Dict
from src.models.embeddings_client import embed_documents as _embed_docs

def embed_documents(docs: List[Dict]) -> List[list[float]]:
    texts = [d["text"] for d in docs]
    return _embed_docs(texts)