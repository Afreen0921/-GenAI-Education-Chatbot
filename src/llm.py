from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

def generate_answer(prompt):

    result = generator(
        prompt,
        max_new_tokens=40,
        do_sample=False,
        truncation=True
    )

    generated_text = result[0]["generated_text"]

    # Remove prompt from output
    answer = generated_text.replace(prompt, "").strip()

    return answer