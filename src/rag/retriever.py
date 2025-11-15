import faiss
import numpy as np
import os
import json
from typing import List, Dict, Any

from src.models.embeddings_client import embed_query

# Resolve paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "data", "vector_store", "news.index")
META_PATH = os.path.join(BASE_DIR, "data", "vector_store", "news_metadata.json")


def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at: {INDEX_PATH}")
    return faiss.read_index(INDEX_PATH)


def load_metadata() -> List[Dict[str, Any]]:
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found at: {META_PATH}")
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top_k(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Returns top-k documents from FAISS based on query embedding.
    Each item returned is a dict:
    {
        "text": ...,
        "metadata": {...},
        "score": float
    }
    """
    # Load index + metadata
    index = load_faiss_index()
    metadata = load_metadata()

    # Embed query
    q_emb = np.array(embed_query(query)).astype("float32").reshape(1, -1)

    # Search
    scores, idxs = index.search(q_emb, k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        item = metadata[idx]
        results.append(
            {
                "text": item["text"],
                "metadata": item["metadata"],
                "score": float(score),
            }
        )

    return results