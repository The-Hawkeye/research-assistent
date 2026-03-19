from functools import lru_cache
import chromadb
from langchain_core.documents import Document

from src.config import DYNAMIC_CHROMA_DIR, COLLECTION_NAME
from src.config import TOP_K

# ✅ Local embedding model (BGE)
from sentence_transformers import SentenceTransformer

# Load once (important for performance)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# ✅ Query embedding using BGE
def embed_query(text: str):
    return model.encode([text])[0].tolist()


@lru_cache(maxsize=100)
def retrieve_cached(query: str, k: int):
    db = chromadb.PersistentClient(path=DYNAMIC_CHROMA_DIR)
    coll = db.get_collection(COLLECTION_NAME)

    # 🔥 FIX: use local embedding
    q_emb = embed_query(query)

    res = coll.query(query_embeddings=[q_emb], n_results=k)

    docs = res["documents"][0]
    metas = res["metadatas"][0]

    return tuple(
        Document(page_content=d, metadata=m)
        for d, m in zip(docs, metas)
    )


def retrieve(query: str, k: int = TOP_K):
    return list(retrieve_cached(query, k))