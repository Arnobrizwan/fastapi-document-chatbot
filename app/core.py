import os
import requests
from typing import List
from fastapi import UploadFile, HTTPException
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

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
        """Helper function to get embeddings for a list of texts."""
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
            embeddings = response.json()
            # Ensure the output is a list of lists of floats
            if isinstance(embeddings, list) and all(isinstance(e, list) for e in embeddings):
                return embeddings
            else:
                raise ValueError("Unexpected response format from Hugging Face API")
        except requests.exceptions.RequestException as e:
            # Handle potential API errors, timeouts, etc.
            raise RuntimeError(f"Failed to get embeddings from Hugging Face API: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        # The API expects a list, so we wrap the single text in a list
        # and expect a list with one embedding in return.
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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks: List[str]):
    """
    Creates a FAISS vector store by fetching embeddings from the HF Inference API.
    """
    if not text_chunks:
        raise ValueError("Cannot create vector store from empty text chunks.")
    
    if not HF_TOKEN:
        raise ValueError("Hugging Face API token is not set in environment variables.")

    # Initialize our custom embedding class
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_url=HF_API_URL, token=HF_TOKEN)
    
    # Create the vector store using the custom embedding function
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store
