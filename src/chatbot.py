import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from src.llm import generate_answer


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load FAISS vector store
def load_vector_store():

    index = faiss.read_index("faiss_store/index.faiss")

    with open("faiss_store/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    return index, texts


# Retrieve relevant context
def retrieve_context(question, index, texts, k=1):

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

    retrieved_texts = [texts[i] for i in indices[0]]

    return "\n".join(retrieved_texts)


# Main chatbot function
def chat(question):

    index, texts = load_vector_store()

    context = retrieve_context(
        question,
        index,
        texts
    )

    prompt = f"""
You are an AI Education Chatbot.

Answer ONLY from the given context.
If the answer is not present in the context, say:
"Answer not found in study materials."

Keep the answer short and student-friendly.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = generate_answer(prompt)

    return answer


# Run chatbot in terminal
if __name__ == "__main__":

    print("\nChatbot is running... type 'exit' to stop\n")

    while True:

        q = input("Ask a question: ")

        if q.lower() == "exit":
            print("Goodbye!")
            break

        answer = chat(q)

        print("\nAnswer:\n", answer)

        print("\n" + "-" * 50 + "\n")