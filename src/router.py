from src.llm import generate_response
from src.retriever import retrieve_docs

def route_query(query):

    query = query.lower()

    if "pdf" in query or "document" in query or "chapter" in query:
        docs = retrieve_docs(query)
        return generate_response(query, docs)

    elif "teach" in query or "explain" in query:
        prompt = f"You are a tutor. Explain clearly:\n{query}"
        return generate_response(prompt)

    else:
        return generate_response(query)