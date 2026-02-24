# GenAI Multi-PDF Q&A (Streamlit + Gemini + FAISS)

Lightweight Streamlit app to upload multiple PDFs, extract text, create vector embeddings with HuggingFace / Gemini, store them with FAISS, and answer user questions using Google Gemini LLM.



## Features
- Upload multiple PDF files and extract combined text.
- Split text into chunks for better embeddings.
- Generate embeddings and store locally with FAISS.
- Retrieve top-k similar chunks and answer questions via Gemini (ChatGoogleGenerativeAI) chain.
- Simple Streamlit UI for processing and querying.

## Setup

1. Create a Python virtual environment and activate it:
```sh
python -m venv .venv
source .venv/bin/activate 

2. Install dependencies:

pip install -r requirements.txt

3. Set environment variables
GOOGLE_API_KEY=your_google_api_key_here

4. run
streamlit run app.py


GenAI_multi_pdf/sample1.png
GenAI_multi_pdf/sample2.png
GenAI_multi_pdf/sample3.png
