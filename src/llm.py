import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

def generate_answer(prompt):

    response = client.chat_completion(
        model="HuggingFaceH4/zephyr-7b-beta",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=80
    )

    return response.choices[0].message.content