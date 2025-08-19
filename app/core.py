import io
from typing import List
from fastapi import UploadFile, HTTPException
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_text_from_files(files: List[UploadFile]) -> str:
    """
    Extracts raw text from a list of uploaded files (PDFs and TXT).
    """
    text = ""
    for file in files:
        try:
            # Read file content into memory first
            file_content = file.file.read()
            
            # Reset file pointer for potential re-reading
            file.file.seek(0)
            
            # Security check for file type
            if file.content_type == "application/pdf" or file.filename.lower().endswith('.pdf'):
                try:
                    # Use BytesIO to handle the file content properly
                    pdf_file = io.BytesIO(file_content)
                    pdf_reader = PdfReader(pdf_file)
                    
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            
                except Exception as e:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Error reading PDF file '{file.filename}': {str(e)}"
                    )
                    
            elif file.content_type == "text/plain" or file.filename.lower().endswith('.txt'):
                try:
                    text += file_content.decode("utf-8") + "\n"
                except UnicodeDecodeError:
                    # Try different encodings
                    try:
                        text += file_content.decode("latin-1") + "\n"
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Error decoding text file '{file.filename}': {str(e)}"
                        )
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}. Only PDF and TXT files are allowed."
                )
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected error processing file '{file.filename}': {str(e)}"
            )
    
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text content could be extracted from the uploaded files."
        )
    
    return text

def get_text_chunks(text: str) -> List[str]:
    """
    Splits a long text into smaller chunks for processing.
    """
    if not text.strip():
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced chunk size for memory efficiency
        chunk_overlap=100,  # Reduced overlap
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
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
    
    try:
        # Use an even smaller, more memory-efficient model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Force CPU usage
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Process chunks in smaller batches to avoid memory issues
        batch_size = 50
        vector_store = None
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            
            if vector_store is None:
                # Create initial vector store
                vector_store = FAISS.from_texts(texts=batch, embedding=embeddings)
            else:
                # Add to existing vector store
                batch_store = FAISS.from_texts(texts=batch, embedding=embeddings)
                vector_store.merge_from(batch_store)
        
        return vector_store
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating vector store: {str(e)}"
        )