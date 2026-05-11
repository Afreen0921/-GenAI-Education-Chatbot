from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

def generate_answer(prompt):

    result = generator(
        prompt,
        max_new_tokens=60,
        truncation=True
    )

    return result[0]["generated_text"]