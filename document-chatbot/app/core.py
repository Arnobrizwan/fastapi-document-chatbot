from typing import List
from fastapi import UploadFile, HTTPException
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# LangChainDeprecationWarning: Updated imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_text_from_files(files: List[UploadFile]) -> str:
    """
    Extracts raw text from a list of uploaded files (PDFs and TXT).
    """
    text = ""
    for file in files:
        # Security check for file type
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
    """
    Splits a long text into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks: List[str]):
    """
    Creates a FAISS vector store from text chunks using HuggingFace embeddings.
    This is the core of the retrieval mechanism.
    """
    if not text_chunks:
        raise ValueError("Cannot create vector store from empty text chunks.")
        
    # Using a popular, lightweight sentence transformer model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create the vector store
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store
