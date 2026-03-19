from tavily import TavilyClient
import os
import streamlit as st
tavily_key = st.secrets["TAVILY_API_KEY"]
def search_web(query: str, top_k: int = 5):
    client = TavilyClient(api_key=tavily_key)
    # client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    res = client.search(
        query=query,
        max_results=top_k,
        include_raw_content=True,
    )

    return [
        {
            "text": r.get("content") or r.get("raw_content"),
            "url": r.get("url"),
            "title": r.get("title"),
        }
        for r in res.get("results", [])
    ]