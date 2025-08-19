from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    session_id: str
    question: str

class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str
    sources: List[str]

class HealthCheck(BaseModel):
    """Response model for the /health endpoint."""
    status: str = "OK"

class Token(BaseModel):
    """Response model for the JWT /token endpoint."""
    access_token: str
    token_type: str

class User(BaseModel):
    """Pydantic model for user data (used in security), compatible with Pydantic v1."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
