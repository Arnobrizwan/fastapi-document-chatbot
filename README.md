Document Chatbot API
This project provides a minimal, yet powerful, chatbot API built with FastAPI. Users can upload text-based documents (PDF and TXT) and ask questions about their content. The chatbot provides grounded answers with citations from the source documents.

Features
File Upload: Accepts 1-5 PDF or TXT files per session.

Q&A: Ask questions and get answers based only on the uploaded documents.

Source Citations: Responses include the actual text snippets used to generate the answer.

Session Handling: Manages document context within distinct sessions.

Health Check: A /health endpoint to monitor API status.

Secure Admin Route: A bonus JWT-protected endpoint to view active sessions.

Tech Stack
Backend: FastAPI

Text Processing: LangChain

Embeddings: sentence-transformers (specifically all-MiniLM-L6-v2)

Vector Store: FAISS (in-memory)

LLM for Q&A: Hugging Face Hub (e.g., google/flan-t5-xxl)

Setup and Installation
Prerequisites
Python 3.8+

A Hugging Face account and an API Token.

1. Clone the Repository
git clone <your-repo-url>
cd document-chatbot

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Configure Environment Variables
Create a .env file in the root directory and add your Hugging Face API token. You can also set a custom secret key for JWT.

HUGGINGFACEHUB_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
SECRET_KEY="your-super-secret-key-for-jwt"

How to Run the Application
Once the setup is complete, you can run the FastAPI server using Uvicorn:

uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000. You can access the interactive API documentation at http://127.0.0.1:8000/docs.

Example API Usage (cURL)
1. Health Check
curl -X GET "http://127.0.0.1:8000/health"

Expected Response:

{
  "status": "OK"
}

2. Upload Documents
Upload one or more files to create a new session.

curl -X POST "http://127.0.0.1:8000/upload" \
-H "Content-Type: multipart/form-data" \
-F "files=@/path/to/your/document1.pdf" \
-F "files=@/path/to/your/document2.txt"

Expected Response:

{
  "session_id": "session_1",
  "message": "Files uploaded successfully."
}

3. Ask a Question
Use the session_id from the upload step to ask a question.

curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{
  "session_id": "session_1",
  "question": "What is the main topic of the document?"
}'

Expected Response:

{
  "answer": "The main topic of the document is the impact of climate change on polar bears.",
  "sources": [
    "Polar bears rely heavily on sea ice for hunting seals...",
    "A 2021 study showed a significant decline in the polar bear population..."
  ]
}

4. Get Session Sources (Bonus)
Retrieve all the text chunks for a given session.

curl -X GET "http://127.0.0.1:8000/sessions/session_1/sources"

5. Admin: Get JWT Token (Bonus)
curl -X POST "http://127.0.0.1:8000/token" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "username=admin&password=adminpassword"

6. Admin: Access Protected Route (Bonus)
Use the token from the previous step to access the protected route.

curl -X GET "http://127.0.0.1:8000/admin/sessions" \
-H "Authorization: Bearer <your_access_token>"
