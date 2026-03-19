from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def get_embedder(backend="bge"):
    def embed(texts):
        return model.encode(texts).tolist()
    return embed, "cosine"