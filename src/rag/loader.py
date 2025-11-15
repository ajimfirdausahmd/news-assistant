import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/news.csv")

def load_news_df() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print("Loaded CSV Columns:", df.columns.tolist())
    
    df = df[["title","article_content","summary","author","sentiment"]].fillna("")
    return df

def build_documents(df: pd.DataFrame):
    docs = []
    for idx, row in df.iterrows():
        text = (
            f"Title: {row['title']}\n"
            f"Summary: {row['summary']}\n"
            f"Content: {row['article_content']}\n"
            f"Author: {row['author']}"
        )
        docs.append(
            {
                "id": idx,
                "text": text,
                "metadata": {
                    "title": row["title"],
                    "author": row["author"],
                    "sentiment": row["sentiment"],
                }
            }
        )
    return docs