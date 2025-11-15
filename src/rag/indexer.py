from typing import List, Dict
from pathlib import Path
import faiss
import numpy as np
import json

VECTOR_DIR = Path("data/vector_store")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

def build_faiss_index(embeddings: List[List[float]]):
    vecs = np.array(embeddings).astype("float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    return index

def save_index(index, docs: List[Dict]):
    faiss.write_index(index, str(VECTOR_DIR / "news.index"))
    with open(VECTOR_DIR / "news_metadata.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

def load_index_and_metadata():
    index = faiss.read_index(str(VECTOR_DIR / "news.index"))
    with open(VECTOR_DIR / "news_metadata.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    return index, docs