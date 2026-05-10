import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vectorstore/faiss_index")

with open("vectorstore/texts.pkl", "rb") as f:
    texts = pickle.load(f)

def retrieve(query, k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), k
    )

    return [texts[i] for i in indices[0] if i < len(texts)]