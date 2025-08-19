from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}

def test_upload_files():
    """Test file upload endpoint with a dummy text file."""
    # Create a dummy file for testing
    dummy_file_path = "test_doc.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a test document for the chatbot.")

    with open(dummy_file_path, "rb") as f:
        response = client.post("/upload", files={"files": ("test_doc.txt", f, "text/plain")})
    
    os.remove(dummy_file_path) # Clean up the dummy file

    assert response.status_code == 200
    json_response = response.json()
    assert "session_id" in json_response
    assert "session_1" in json_response["session_id"]
    assert "files uploaded successfully" in json_response["message"]

def test_ask_question_no_session():
    """Test asking a question with a non-existent session ID."""
    response = client.post("/ask", json={"session_id": "invalid_session", "question": "What is this?"})
    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found."}

