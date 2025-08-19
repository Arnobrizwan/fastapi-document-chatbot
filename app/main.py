import os
import gc
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from huggingface_hub import InferenceClient

# Import from our application modules
from . import core, models, security, database

# --- App Initialization with Lifespan Events ---
from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    try:
        database.init_db()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
    yield
    # Code to run on shutdown
    gc.collect()  # Force garbage collection on shutdown

app = FastAPI(
    title="Document Chatbot API",
    description="Upload documents and ask questions about them.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/", tags=["Status"])
def root():
    """Root endpoint to verify the API is running."""
    return {"message": "Document Chatbot API is running", "status": "OK"}

@app.get("/health", response_model=models.HealthCheck, tags=["Status"])
def health_check():
    """Endpoint to check if the API is running."""
    return {"status": "OK"}

@app.post("/upload", tags=["Chatbot"])
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload 1-5 files (PDF/TXT), process them, and create a new persistent session.
    """
    if not 1 <= len(files) <= 5:
        raise HTTPException(status_code=400, detail="Must upload between 1 and 5 files.")

    # Enhanced file validation
    allowed_types = ["application/pdf", "text/plain"]
    max_file_size = 5 * 1024 * 1024  # Reduced to 5MB for Render's memory limits
    
    for file in files:
        # Check file size
        if hasattr(file, 'size') and file.size and file.size > max_file_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File '{file.filename}' is too large (max 5MB)."
            )
        
        # Check file type
        if file.content_type not in allowed_types and not (
            file.filename.lower().endswith('.pdf') or file.filename.lower().endswith('.txt')
        ):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' has unsupported type. Only PDF and TXT files are allowed."
            )

    try:
        # Process files and create vector store
        raw_text = core.get_text_from_files(files)
        
        if not raw_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the uploaded files."
            )
        
        # Limit text size to prevent memory issues
        if len(raw_text) > 100000:  # 100KB limit
            raw_text = raw_text[:100000]
            print("Warning: Text truncated to 100KB to prevent memory issues")
        
        text_chunks = core.get_text_chunks(raw_text)
        
        if not text_chunks:
            raise HTTPException(
                status_code=400,
                detail="Could not create text chunks from the extracted text."
            )
        
        # Limit number of chunks to prevent memory issues
        if len(text_chunks) > 200:
            text_chunks = text_chunks[:200]
            print("Warning: Limited to 200 chunks to prevent memory issues")
        
        vector_store = core.get_vector_store(text_chunks)

        # Save the session to the database and get the new session_id
        session_id = database.save_session(vector_store=vector_store, sources=text_chunks)
        
        # Force garbage collection after processing
        gc.collect()
        
        return {
            "session_id": session_id, 
            "message": f"{len(files)} files uploaded successfully.",
            "chunks_created": len(text_chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        # Force garbage collection on error
        gc.collect()
        print(f"Error during upload: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred during file processing. Please try with smaller files."
        )

@app.post("/ask", response_model=models.AskResponse, tags=["Chatbot"])
async def ask_question(request: models.AskRequest):
    """
    Ask a question against a specific session's documents, loaded from the database.
    """
    try:
        session = database.load_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")

        vector_store = session["vector_store"]
        
        # Ensure Hugging Face API token is set
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise HTTPException(
                status_code=500, 
                detail="Hugging Face API token not configured."
            )

        # Retrieve relevant documents (reduced k for memory efficiency)
        docs = vector_store.similarity_search(request.question, k=2)
        
        # Create a more concise context to stay within token limits
        context_parts = []
        for doc in docs:
            content = doc.page_content.strip()
            # Limit each document content to prevent token overflow
            if len(content) > 500:
                content = content[:500] + "..."
            context_parts.append(content)
        
        context = "\n\n".join(context_parts)
        
        # More concise prompt to save tokens
        prompt = f"Context:\n{context}\n\nQuestion: {request.question}\n\nAnswer based on the context above:"

        # Use InferenceClient with error handling
        client = InferenceClient(token=hf_token)
        response = client.text_generation(
            prompt=prompt,
            model="google/flan-t5-large",  # Use smaller model for better reliability
            max_new_tokens=256,  # Reduced token count
            temperature=0.5,
            do_sample=True,
            return_full_text=False
        )
        
        # Clean up the response
        answer = response.strip() if response else "I couldn't generate an answer based on the provided context."
        
        return {
            "answer": answer,
            "sources": [doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content for doc in docs]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in ask_question: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An error occurred while processing your question. Please try again."
        )

@app.get("/sessions/{session_id}/sources", response_model=List[str], tags=["Sessions"])
def get_session_sources(session_id: str):
    """Retrieve the source text chunks for a given session from the database."""
    try:
        session = database.load_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        return session.get("sources", [])
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting session sources: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving session sources.")

# --- Bonus: JWT Admin Routes ---

@app.post("/token", response_model=models.Token, tags=["Admin"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate and get a JWT access token."""
    try:
        user = security.get_user(form_data.username)
        if not user or not security.verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = security.create_access_token(data={"sub": user["username"]})
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in login: {e}")
        raise HTTPException(status_code=500, detail="Authentication error occurred.")

@app.get("/admin/sessions", tags=["Admin"])
def get_all_sessions(current_user: models.User = Depends(security.get_current_active_user)):
    """Protected route to view all active session IDs from the database."""
    try:
        session_ids = database.get_all_session_ids()
        return {"sessions": session_ids}
    except Exception as e:
        print(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving sessions.")