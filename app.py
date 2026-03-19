import streamlit as st
import os, shutil
from src.ingest import ingest_pdfs
from src.qa import run_query

UPLOAD_DIR = "uploaded_pdfs"

st.set_page_config(page_title="Agentic Research Assistant", layout="wide")

# ---- SESSION STATE INIT ---- #
if "chat" not in st.session_state:
    st.session_state.chat = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ---- RESET FUNCTION ---- #
def reset_all():
    st.session_state.chat = []
    st.session_state.query_input = ""
    st.session_state.uploader_key += 1  # reset uploader UI

    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


# ---- QUERY HANDLER ---- #
def handle_query():
    query = st.session_state.query_input

    if query:
        answer = run_query(query)
        st.session_state.chat.append((query, answer))

        # clear input safely inside callback
        st.session_state.query_input = ""


# ---- UI ---- #
st.title("🚀 Agentic Research Assistant")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🆕 New Chat"):
        reset_all()
        st.rerun()


# ---- FILE UPLOAD ---- #
uploaded = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key=st.session_state.uploader_key
)

if uploaded:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for f in uploaded:
        with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
            out.write(f.getbuffer())

    if st.button("📥 Ingest PDFs"):
        ingest_pdfs()
        st.success("Indexed successfully!")


# ---- QUERY INPUT ---- #
st.text_input(
    "Ask your question",
    key="query_input",
    on_change=handle_query
)


# ---- CHAT DISPLAY ---- #
for q, a in reversed(st.session_state.chat):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Assistant:** {a}")