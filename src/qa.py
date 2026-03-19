from typing import List
import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from crewai import Agent, Task, Crew, Process, LLM

from src.retriever import retrieve
from src.search import search_web
from src.rag_answer import answer_with_context, validate_answer

# ---- Load env ---- #
load_dotenv()

# ---- Configure Groq LLM for CrewAI ---- #
llm = LLM(
    model="groq/llama-3.1-8b-instant",   # ✅ IMPORTANT
    api_key=os.getenv("GROQ_API_KEY")
)

# ---- AGENTS ---- #

router_agent = Agent(
    role="Router Agent",
    goal="Decide whether to use local documents, web search, or both",
    backstory="Expert at understanding user intent and selecting best data source",
    llm=llm,
    verbose=False
)

retriever_agent = Agent(
    role="Retriever Agent",
    goal="Retrieve relevant information from local vector database",
    backstory="Expert in semantic search over PDFs",
    llm=llm,
    verbose=False
)

web_agent = Agent(
    role="Web Search Agent",
    goal="Retrieve latest information from the web",
    backstory="Expert in Tavily search",
    llm=llm,
    verbose=False
)

answer_agent = Agent(
    role="Answer Generator",
    goal="Generate accurate answers using provided context",
    backstory="Expert in reasoning and synthesis",
    llm=llm,
    verbose=False
)

validator_agent = Agent(
    role="Validator",
    goal="Validate answer correctness and detect hallucinations",
    backstory="Expert in fact-checking",
    llm=llm,
    verbose=False
)


# ---- RERANKER ---- #
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


# ---- MAIN PIPELINE ---- #

def run_query(query: str):

    # ---- ROUTER TASK ---- #
    routing_task = Task(
        description=f"""
    You are a smart routing agent.

    Decide best data source for the query.

    Rules:
    - If query refers to uploaded document, PDF → use "local"
    - If query asks latest/current info → use "web"
    - If both needed → use "hybrid"

    Query: {query}

    Return ONLY one word: local OR web OR hybrid
    """,
        expected_output="local | web | hybrid",
        agent=router_agent
    )
    routing_result = Crew(
        agents=[router_agent],
        tasks=[routing_task],
        process=Process.sequential
    ).kickoff()

    route = routing_result.raw.strip().lower()

    if route not in ["local", "web", "hybrid"]:
        route = "hybrid"

    # ---- RETRIEVAL ---- #
    local_docs = []
    web_docs = []

    if route in ["local", "hybrid"]:
        local_docs = retrieve(query)

    if route in ["web", "hybrid"]:
        web_results = search_web(query)

        web_docs = [
            Document(
                page_content=w["text"],
                metadata={
                    "source": "web",
                    "url": w.get("url"),
                    "title": w.get("title"),
                }
            )
            for w in web_results if w.get("text")
        ]

    all_docs = local_docs + web_docs

    # ---- RERANK ---- #
    ranked_docs = rerank(query, all_docs)

    # ---- ANSWER GENERATION ---- #
    answer = answer_with_context(query, ranked_docs)

    # ---- VALIDATION ---- #
    validation = validate_answer(query, answer, ranked_docs, web_docs)

    return f"""
{answer}

---

📌 Route Used: {route}

📊 Sources:
- Local Docs: {len(local_docs)}
- Web Docs: {len(web_docs)}

🔍 Validation:
{validation}
"""