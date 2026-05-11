from transformers import pipeline

generator = pipeline(
    task="text-generation",
    model="google/flan-t5-base"
)

def generate_answer(prompt):

    result = generator(
        prompt,
        max_new_tokens=50
    )

    return result[0]["generated_text"]