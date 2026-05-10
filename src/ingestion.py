from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import pickle
import os

DATA_PATH = "data"
FAISS_PATH = "faiss_store"


# CLEAN OCR / PDF TEXT
def clean_text(text):

    corrections = {
        "Pyhton": "Python",
        "P ython": "Python",
        "T o": "To",
        "machien": "machine",
        "lernning": "learning",
    }

    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)

    return text


# LOAD ALL PDF DOCUMENTS
def load_documents():

    all_docs = []

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(DATA_PATH, file)

            loader = PyPDFLoader(pdf_path)

            try:
                docs = loader.load()

                for doc in docs:
                    doc.page_content = clean_text(doc.page_content)

                all_docs.extend(docs)

                print(f"Loaded: {file}")

            except Exception as e:
                print(f"Error loading {file}: {e}")



            # CLEAN TEXT
            for doc in docs:
                doc.page_content = clean_text(doc.page_content)

            all_docs.extend(docs)

            print(f"Loaded: {file}")

    return all_docs


# SPLIT DOCUMENTS INTO CHUNKS
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks


# CREATE EMBEDDINGS
def create_embeddings(texts):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


# CREATE FAISS VECTOR STORE
def create_vector_store():

    print("Loading PDFs...")

    documents = load_documents()

    print(f"Total documents loaded: {len(documents)}")

    print("Splitting text into chunks...")

    chunks = split_documents(documents)

    texts = [doc.page_content for doc in chunks]

    print(f"Total chunks created: {len(texts)}")

    print("Creating embeddings...")

    embeddings = create_embeddings(texts)

    dimension = embeddings.shape[1]

    print("Creating FAISS index...")

    index = faiss.IndexFlatIP(dimension)

    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(FAISS_PATH, exist_ok=True)

    faiss.write_index(index, f"{FAISS_PATH}/index.faiss")

    with open(f"{FAISS_PATH}/texts.pkl", "wb") as f:

        pickle.dump(texts, f)

    print("FAISS Vector Store Created Successfully!")


if __name__ == "__main__":

    create_vector_store()