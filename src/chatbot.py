import faiss
import pickle
import numpy as np
import re

from sentence_transformers import SentenceTransformer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load vector store
def load_vector_store():

    index = faiss.read_index("faiss_store/index.faiss")

    with open("faiss_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    return index, texts


# Retrieve relevant context
def retrieve_context(question, index, texts, k=5):

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    scores, indices = index.search(
        question_embedding,
        k
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx < len(texts):

            chunk = texts[idx]

            # Keyword filtering
            if any(
                word.lower() in chunk.lower()
                for word in question.split()
            ):

                retrieved_chunks.append(chunk)

    if not retrieved_chunks:
        return ""

    return " ".join(retrieved_chunks[:2])


# Clean final answer
def clean_answer(text):

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    # Remove page-number joins
    text = re.sub(
        r'([a-zA-Z])(\d+)',
        r'\1 ',
        text
    )

    # Remove repeated words
    text = re.sub(
        r'\b(\w+)( \1\b)+',
        r'\1',
        text,
        flags=re.IGNORECASE
    )

    return text


# Main chatbot function
def chat(question):

    index, texts = load_vector_store()

    context = retrieve_context(
        question,
        index,
        texts
    )

    if not context:
        return "Answer not found in study materials."

    context = clean_answer(context)

    sentences = context.split(". ")

    final_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence.split()) < 6:
            continue

        if any(
            word.lower() in sentence.lower()
            for word in question.split()
        ):

            final_sentences.append(sentence)

    # Fallback
    if not final_sentences:
        final_sentences = sentences[:2]

    # 2–3 line answer
    answer = ". ".join(final_sentences[:3])

    return answer.strip() + "."