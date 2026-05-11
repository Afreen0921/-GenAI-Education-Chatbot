from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

def generate_answer(prompt):

    result = generator(
        prompt,
        max_new_tokens=40,
        truncation=True,
        do_sample=False
    )

    generated_text = result[0]["generated_text"]

    answer = generated_text[len(prompt):].strip()

    answer = answer.replace("Answer:", "").strip()

    if answer == "":
        answer = "Answer not found in study materials."

    return answer