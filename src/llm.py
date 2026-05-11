import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    token=os.getenv("HF_TOKEN")
)

def generate_answer(prompt):

    response = client.text_generation(
        prompt,
        model="google/flan-t5-base",
        max_new_tokens=80
    )

    return response