import os
import requests
from typing import List
from fastapi import UploadFile, HTTPException
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import time

# --- Hugging Face Inference API Configuration ---
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class HuggingFaceInferenceAPIEmbeddings(Embeddings):
    """Custom embedding class to use the Hugging Face Inference API."""
    def __init__(self, api_url: str, token: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Helper function to get embeddings for a list of texts with retries."""
        retries = 3
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": texts, "options": {"wait_for_model": True}}
                )
                response.raise_for_status()
                embeddings = response.json()
                if isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
                    return embeddings
                else:
                    # This can happen if the API returns an error message instead of embeddings
                    print(f"Unexpected API response format: {embeddings}")
                    raise ValueError("Unexpected response format from Hugging Face API")
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2) # Wait for 2 seconds before retrying
                else:
                    raise RuntimeError(f"Failed to get embeddings from Hugging Face API after {retries} attempts: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        result = self._embed([text])
        return result[0]

def get_text_from_files(files: List[UploadFile]) -> str:
    """Extracts raw text from a list of uploaded files (PDFs and TXT)."""
    text = ""
    for file in files:
        if file.content_type == "application/pdf":
            try:
                pdf_reader = PdfReader(file.file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            except Exception:
                raise HTTPException(status_code=400, detail=f"Error reading PDF file: {file.filename}")
        elif file.content_type == "text/plain":
            text += file.file.read().decode("utf-8")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}. Only PDF and TXT are allowed.")
    return text

def get_text_chunks(text: str) -> List[str]:
    """Splits a long text into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # Smaller chunk size for faster processing per chunk
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks: List[str]):
    """
    Creates a FAISS vector store by fetching embeddings in batches.
    """
    if not text_chunks:
        raise ValueError("Cannot create vector store from empty text chunks.")
    
    if not HF_TOKEN:
        raise ValueError("Hugging Face API token is not set in environment variables.")

    embeddings = HuggingFaceInferenceAPIEmbeddings(api_url=HF_API_URL, token=HF_TOKEN)
    
    # --- BATCH PROCESSING LOGIC ---
    batch_size = 20  # Process 20 chunks at a time
    vector_store = None
    
    print(f"Starting embedding process for {len(text_chunks)} chunks in batches of {batch_size}...")
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        if vector_store is None:
            # Create the initial vector store with the first batch
            vector_store = FAISS.from_texts(texts=batch, embedding=embeddings)
        else:
            # Add subsequent batches to the existing store
            vector_store.add_texts(texts=batch)
            
        time.sleep(1) # Add a small delay between API calls to avoid rate limiting
        
    print("Embedding process complete.")
    return vector_store
