import re
from datetime import datetime
from typing import List
from langchain_core.documents import Document

from src.retriever import retrieve
from src.search import search_web
from src.rag_answer import answer_with_context
from src.rag_answer import validate_answer

# ---- Heuristic routing ----
RECENCY_PAT = re.compile(r"\b(latest|today|recent|202[3-9])\b", re.I)

def heuristic_route(query: str, has_local: bool):
    if RECENCY_PAT.search(query):
        return "hybrid" if has_local else "web"
    return "local" if has_local else "web"


# ---- CrewAI Router ----
from crewai import Agent, Task, Crew, Process

def classify_route_llm(query: str):
    try:
        router = Agent(
            role="Router",
            goal="Return exactly: local | web | hybrid",
            backstory="Smart query router",
            verbose=False
        )

        task = Task(
            description=f"Query: {query}",
            expected_output="local | web | hybrid",
            agent=router
        )

        result = Crew(
            agents=[router],
            tasks=[task],
            process=Process.sequential
        ).kickoff()

        out = result.raw.strip().lower()
        if out not in ["local", "web", "hybrid"]:
            return "hybrid"

        return out

    except:
        return None


# ---- BGE Reranker ----
try:
    from FlagEmbedding import FlagReranker
    RERANKER = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
except:
    RERANKER = None


def rerank(query: str, docs: List[Document], top_n=5):
    if not docs or not RERANKER:
        return docs[:top_n]

    pairs = [[query, d.page_content[:4000]] for d in docs]
    scores = RERANKER.compute_score(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


# ---- MAIN PIPELINE ----
def run_query(query: str):
    local_docs = retrieve(query)

    route = classify_route_llm(query)
    if not route:
        route = heuristic_route(query, bool(local_docs))

    web_docs = search_web(query) if route in ["web", "hybrid"] else []

    web_docs = [
        Document(
            page_content=w["text"],   # ✅ FIX
            metadata={
                "source": "web",
                "url": w.get("url"),
                "title": w.get("title"),
            }
        )
        for w in web_docs if w.get("text")
    ]

    all_docs = local_docs + web_docs

    # 🔥 RERANK HERE
    ranked_docs = rerank(query, all_docs)

    # Generate answer
    answer_stream = answer_with_context(query, ranked_docs)

    # Collect answer (for validation)
    full_answer = "".join(list(answer_stream))

    # 🔥 VALIDATION STEP
    validation = validate_answer(query, full_answer, ranked_docs, web_docs)

    return full_answer + "\n\n---\nValidation:\n" + validation