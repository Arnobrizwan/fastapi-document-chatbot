import sqlite3
import pickle
import uuid
from typing import List, Dict, Any, Optional

# --- Database Configuration ---
DB_FILE = "chatbot_sessions.db"

def init_db():
    """Initializes the database and creates the sessions table if it doesn't exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                vector_store BLOB NOT NULL,
                sources BLOB NOT NULL
            )
        """)
        conn.commit()

def save_session(vector_store: Any, sources: List[str]) -> str:
    """
    Saves a new session to the database.
    - Serializes the vector store and sources for storage.
    - Generates a unique session ID.
    - Returns the new session ID.
    """
    session_id = str(uuid.uuid4())
    
    # Serialize the Python objects into a byte stream (BLOB)
    pickled_vector_store = pickle.dumps(vector_store)
    pickled_sources = pickle.dumps(sources)
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, vector_store, sources) VALUES (?, ?, ?)",
            (session_id, pickled_vector_store, pickled_sources)
        )
        conn.commit()
    return session_id

def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads a session from the database by its ID.
    - Deserializes the vector store and sources back into Python objects.
    - Returns a dictionary with the session data or None if not found.
    """
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT vector_store, sources FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            vector_store = pickle.loads(row[0])
            sources = pickle.loads(row[1])
            return {"vector_store": vector_store, "sources": sources}
    return None

def get_all_session_ids() -> List[str]:
    """Retrieves a list of all session IDs from the database."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id FROM sessions")
        rows = cursor.fetchall()
        return [row[0] for row in rows]

