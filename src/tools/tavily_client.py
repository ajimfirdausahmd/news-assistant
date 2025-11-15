import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY in .env file")

client = TavilyClient(api_key=TAVILY_API_KEY)


def tavily_search(query: str, max_results: int = 5):
    response = client.search(
        query=query,
        max_results=max_results,
        include_raw_content=True
    )

    results = []
    for item in response.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("raw_content", item.get("content", "")),
                "score": item.get("score", 0),
                "published": item.get("published_date", ""),
                "source": "web-search",
            }
        )

    return results