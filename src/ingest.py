from pathlib import Path
from tqdm import tqdm
import chromadb

from src.pdf_utils import extract_pdf_text_by_page, chunk_text
from src.embedder import get_embedder
from src.config import PDF_PATH, COLLECTION_NAME

# ---- helper ----
def get_all_pdfs():
    return list(Path(PDF_PATH).glob("*.pdf"))

# ---- MAIN FUNCTION (USED BY STREAMLIT) ----
def ingest_pdfs(backend: str = "bge", reset: bool = False):
    client = chromadb.PersistentClient(path="data/dynamic_chroma")

    embed, metric = get_embedder(backend)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass

    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": metric}
    )

    pdf_files = get_all_pdfs()

    if not pdf_files:
        print("⚠️ No PDFs found")
        return

    for pdf in pdf_files:
        pages = extract_pdf_text_by_page(pdf)

        docs, metas, ids = [], [], []

        for p in pages:
            for idx, ch in enumerate(chunk_text(p["text"])):
                docs.append(ch)
                metas.append({
                    "source": pdf.name,
                    "page": p["page"],
                    "chunk_index": idx,
                })
                ids.append(f"{pdf.name}_{p['page']}_{idx}")

        for i in tqdm(range(0, len(docs), 64)):
            batch_docs = docs[i:i+64]
            batch_embs = embed(batch_docs)

            coll.add(
                documents=batch_docs,
                embeddings=batch_embs,
                metadatas=metas[i:i+64],
                ids=ids[i:i+64],
            )

    print("✅ All PDFs indexed")