# -GenAI-Education-Chatbot
📚 GenAI Education Chatbot
An intelligent AI-powered Education Chatbot built using Retrieval-Augmented Generation (RAG), FAISS, Sentence Transformers, Ollama, and Streamlit.
The chatbot answers questions directly from uploaded PDF study materials such as Mathematics, Science, Physics, English.

| Feature                    | Description                                             |
| -------------------------  | ------------------------------------------------------- |
| 📄 PDF Question Answering | Ask questions directly from uploaded PDFs               |
| 🧠 RAG Pipeline           | Uses Retrieval-Augmented Generation                     |
| 🔎 Semantic Search        | Retrieves relevant content using FAISS                  |
| 🤖 Local LLM              | Uses Ollama models like Phi and TinyLlama               |
| 📚 Multi-Subject Support  | Supports Maths, Science, Physics, English, Python, etc. |
| 🧹 OCR Cleaning           | Cleans noisy PDF/OCR text automatically                 |
| ⚡ Fast Retrieval         | Uses embeddings for efficient searching                 |
| 🛡️ Error Handling         | Skips corrupted PDFs safely                             |
| 💬 Interactive UI         | Streamlit-based chatbot interface                       |


⚙️ System Workflow
User Question
      │
      ▼
FAISS Retriever (chatbot.py)
      │
      ▼
Relevant PDF Chunks Retrieved
      │
      ▼
Embeddings (Sentence Transformers)
      │
      ▼
Ollama LLM (llm.py)
      │
      ▼
Generated Answer

🚀 Installation Guide
Step	Command
Create virtual environment	python -m venv venv
Activate environment	venv\Scripts\activate
Install dependencies	pip install -r requirements.txt


🦙 Install Ollama

Download Ollama from:

Ollama Official Website

📥 Download AI Model

Model	Command

Phi	ollama run phi

TinyLlama	ollama run tinyllama

📄 Add Study Materials

Place all PDF files inside:

data/
Example PDFs
Subject	       File Name
Mathematics	   maths.pdf
Physics        physics.pdf


🧠 Create Vector Database

Run:
python src/ingestion.py

What Happens During Ingestion
| Process            | Description                           |
| ------------------ | ------------------------------------- |
| PDF Loading        | Reads all PDFs from `data/`           |
| Text Cleaning      | Fixes OCR and spelling issues         |
| Chunking           | Splits large text into smaller chunks |
| Embedding Creation | Converts text into vectors            |
| FAISS Storage      | Saves vectors for semantic search     |

▶️ Run Application
streamlit run app.py


💬 Example Questions
| Subject     | Example Questions        |
| ----------- | ------------------------ |
| Mathematics | What is algebra?         |
| Mathematics | Explain vectors          |
| Physics     | What is photosynthesis?  |
| Physics     | Define gravity           |


🛠️ Technology Stack
| Component       | Technology Used       |
| --------------- | --------------------- |
| Frontend        | Streamlit             |
| LLM             | Ollama                |
| Embeddings      | Sentence Transformers |
| Vector Database | FAISS                 |
| PDF Parsing     | LangChain + PyPDF     |
| Backend         | Python                |

🔮 Future Improvements
| Feature                | Status  |
| ---------------------- | ------- |
| Voice Assistant        | Planned |
| Chat History           | Planned |
| PDF Upload UI          | Planned |
| Subject Filtering      | Planned |
| Better OCR Support     | Planned |
| Multi-language Support | Planned |

📄 License
This project is created for educational and learning purposes only.
