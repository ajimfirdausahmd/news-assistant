import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in .env file")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key
    )