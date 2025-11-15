from .loader import load_news_df, build_documents
from .embedder import embed_documents
from .indexer import build_faiss_index, save_index

def main():
    print("Loading dataset...")
    df = load_news_df()

    print("Building documents...")
    docs = build_documents(df)

    print("Embedding documents...")
    embeddings = embed_documents(docs)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving index + metadata...")
    save_index(index, docs)

    print("DONE: Vector index successfully built!")

if __name__ == "__main__":
    main()