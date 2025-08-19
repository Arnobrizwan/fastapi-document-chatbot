import os
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
# LangChainDeprecationWarning: Updated import
from langchain_community.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
# FIX: Import the new InferenceClient
from huggingface_hub import InferenceClient

# Import from our application modules
from . import core, models, security, database

# --- App Initialization with Lifespan Events ---
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    database.init_db()
    yield
    # Code to run on shutdown (if any)

app = FastAPI(
    title="Document Chatbot API",
    description="Upload documents and ask questions about them.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.get("/health", response_model=models.HealthCheck, tags=["Status"])
def health_check():
    """Endpoint to check if the API is running."""
    return {"status": "OK"}

@app.post("/upload", tags=["Chatbot"])
def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload 1-5 files (PDF/TXT), process them, and create a new persistent session.
    """
    if not 1 <= len(files) <= 5:
        raise HTTPException(status_code=400, detail="Must upload between 1 and 5 files.")

    # Security: Basic file size check (e.g., 10MB)
    for file in files:
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File '{file.filename}' is too large (max 10MB).")

    try:
        raw_text = core.get_text_from_files(files)
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the uploaded files.")
        
        text_chunks = core.get_text_chunks(raw_text)
        vector_store = core.get_vector_store(text_chunks)

        # Save the session to the database and get the new session_id
        session_id = database.save_session(vector_store=vector_store, sources=text_chunks)
        
        return {"session_id": session_id, "message": f"{len(files)} files uploaded successfully."}

    except Exception as e:
        # For debugging; in production, you'd have more robust logging
        print(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {e}")

@app.post("/ask", response_model=models.AskResponse, tags=["Chatbot"])
def ask_question(request: models.AskRequest):
    """
    Ask a question against a specific session's documents, loaded from the database.
    """
    session = database.load_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    vector_store = session["vector_store"]
    
    # Ensure Hugging Face API token is set
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=500, detail="Hugging Face API token not configured.")

    # Retrieve relevant documents (top-k=3)
    docs = vector_store.similarity_search(request.question, k=3)
    
    # Create the prompt that will be sent to the model
    # This is what the "stuff" chain does internally
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {request.question}\nHelpful Answer:"

    try:
        # FIX: Use the new InferenceClient directly
        client = InferenceClient(token=hf_token)
        response = client.text_generation(
            prompt=prompt,
            model="google/flan-t5-xxl",
            max_new_tokens=512,
            temperature=0.7,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Hugging Face API: {e}")
    
    return {
        "answer": response,
        "sources": [doc.page_content for doc in docs]
    }

@app.get("/sessions/{session_id}/sources", response_model=List[str], tags=["Sessions"])
def get_session_sources(session_id: str):
    """(Bonus) Retrieve the source text chunks for a given session from the database."""
    session = database.load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session.get("sources", [])

# --- Bonus: JWT Admin Routes ---

@app.post("/token", response_model=models.Token, tags=["Admin"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate and get a JWT access token."""
    user = security.get_user(form_data.username)
    if not user or not security.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = security.create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/admin/sessions", tags=["Admin"])
def get_all_sessions(current_user: models.User = Depends(security.get_current_active_user)):
    """Protected route to view all active session IDs from the database."""
    session_ids = database.get_all_session_ids()
    return {"sessions": session_ids}
