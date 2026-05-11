from dotenv import load_dotenv
import os

load_dotenv()

print("HF TOKEN:", os.getenv("HF_TOKEN"))