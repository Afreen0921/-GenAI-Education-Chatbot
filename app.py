import streamlit as st
from src.chatbot import chat


# Page settings
st.set_page_config(
    page_title="GenAI Education Chatbot",
    page_icon="📚",
    layout="centered"
)


# Title
st.title("📚 GenAI Education Chatbot")


# Description
st.markdown("""
Ask questions from your PDF study materials.

Examples:
- What is algebra?
- Explain vectors
- What are algebraic expressions?
""")


# User input
user_question = st.text_input(
    "Enter your question:"
)


# Ask button
if st.button("Ask Question"):

    if user_question.strip() != "":

        with st.spinner("Searching your PDF documents..."):

            answer = chat(user_question)

        st.subheader("Answer")

        st.write(answer)

    else:

        st.warning("Please enter a question.")