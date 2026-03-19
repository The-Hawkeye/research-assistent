from groq import Groq
import os
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
import streamlit as st
grok_key = st.secrets["GROQ_API_KEY"]

load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
client = Groq(api_key=grok_key)

# ✅ MAIN ANSWER FUNCTION (MISSING)
def answer_with_context(query: str, docs: List[Document]):
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer the question using ONLY the context below.

Question:
{query}

Context:
{context}

If answer is not in context, say "I don't know".
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ✅ VALIDATION (you already had this)
def validate_answer(query, answer, local_docs, web_docs):
    prompt = f"""
Check if answer is supported by context.

Question: {query}

Answer:
{answer}

Return unsupported claims + confidence (0-1).
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    except Exception as e:
        return f"Validation skipped: {e}"