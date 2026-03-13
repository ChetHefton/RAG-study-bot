# RAG Study Bot

A Retrieval-Augmented Generation (RAG) study assistant built over Anatomy & Physiology course materials using a local Large Language Model (LLM). 

Designed with future agentic AIs in the healthcare industry having to cite their knowledge and reduce abstraction in their decisions. LLMs are probabilistic and may hallucinate. RAG constrains generation to verified source material and improves factual reliability.

This project demonstrates how to:

- Run a local 7B instruction-tuned LLM (Mistral)
- Use GPU acceleration with PyTorch
- Structure an AI project using virtual environments and dependency management
- Build toward a retrieval-grounded educational assistant

The long-term goal is to create a healthcare-focused study assistant that answers questions grounded in textbook content rather than relying solely on pretrained model knowledge.

---

## Tech Stack

- Python 3.12
- PyTorch (CUDA enabled)
- Hugging Face Transformers
- Mistral-7B-Instruct-v0.3
- Virtual environment (venv)

---

## Current Status

- Virtual environment configured
- GPU PyTorch installed
- Mistral-7B-Instruct-v0.3 running locally
- Next: Embeddings + FAISS vector search (RAG pipeline)

---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/your-username/ap-rag-study-bot.git  
cd ap-rag-study-bot

### 2. Create Virtual Environment

python -m venv venv

### 3. Activate Virtual Environment

Windows (PowerShell):

.\venv\Scripts\Activate

Mac/Linux:

source venv/bin/activate

### 4. Install Dependencies

pip install -r requirements.txt

---

## Running the Project

### Option 1: Terminal Model Test

To verify the LLM is working locally without the UI:

python testLLM.py

On first run, the model weights (~14GB) will download and cache locally via Hugging Face.

You should see:
- CUDA availability confirmation
- A generated response from the Mistral model

The model is cached outside the repository and will not re-download on future runs.

---

### Option 2: Launch the Web UI

To start the Flask chat interface:

python app.py

Then open your browser and navigate to:

http://127.0.0.1:5000

You should see the local chat interface connected to the Mistral model.

The model loads once at server startup. After it finishes loading, you can begin asking questions through the UI.

---

## Notes on First Run

- The first model download is approximately 14GB.
- This download happens only once and is cached locally.
- A CUDA-enabled GPU is recommended for reasonable performance.
- The model runs fully locally and does not use any external API.

## Notes

- The virtual environment (venv/) is ignored via .gitignore
- Model weights are cached locally and are not stored in the repository
- This project is for educational AI experimentation and does not provide medical advice

---

## Future Roadmap

- Textbook PDF ingestion
- Heading-aware chunking
- Embedding generation (MiniLM)
- FAISS vector search
- Prompt grounding with retrieved context
- Flask web interface
- Citation-backed responses
- Quiz / flashcard mode

---

## Disclaimer

This is an educational AI study assistant and is not intended for clinical or diagnostic use.