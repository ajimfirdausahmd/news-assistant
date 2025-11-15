from typing import TypedDict, List, Optional

class GraphState(TypedDict):
    question: str
    rag_docs: Optional[List[dict]]
    answer: Optional[str]
    route: Optional[str]