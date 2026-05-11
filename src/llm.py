from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def generate_answer(prompt):

    result = generator(
        prompt,
        max_new_tokens=60,
        do_sample=False
    )

    return result[0]["generated_text"].strip()