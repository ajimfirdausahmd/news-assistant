from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-large")

def embed_documents(texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts,normalize_embeddings=True)
    return embeddings.tolist()

def embed_query(text:str) -> list[float]:
    embedding = model.encode([text],normalize_embeddings=True)[0]
    return embedding.tolist()