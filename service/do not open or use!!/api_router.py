from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from new_service.pipeline import get_enhanced_pipeline
from new_service.chat import process_chat_request
import sqlite3
from pathlib import Path
import sys
import os

# Add root to path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from flight_fetcher import get, deserialize_flight, serialize_flight

router = APIRouter(prefix="/api/v2", tags=["enhanced_analysis"])

CACHE_DB_PATH = Path("flight_cache.db")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    flight_id: str
    analysis: Optional[Dict[str, Any]] = None
    points: Optional[List[Dict[str, Any]]] = None
    user_question: str

@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Process a chat request using OpenAI, keeping API keys secure on the backend.
    """
    try:
        response_text = process_chat_request(
            messages=request.messages,
            flight_id=request.flight_id,
            analysis=request.analysis,
            points=request.points,
            user_question=request.user_question
        )
        return {"response": response_text}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{flight_id}")
def analyze_flight_enhanced(flight_id: str):
    """
    Enhanced analysis endpoint including Glitch Analyzer and LLM Layer.
    """
    pipeline = get_enhanced_pipeline()
    
    try:
        flight = None
        
        # Check Cache (Reuse logic or import if possible, but replicating small cache logic is safer to avoid deps)
        if CACHE_DB_PATH.exists():
            conn = sqlite3.connect(str(CACHE_DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM flights WHERE flight_id = ?", (flight_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                flight = deserialize_flight(row[0])
        
        if not flight:
            # Fetch live
            flight = get(flight_id=flight_id)
            # We won't write to cache here to keep "read only" regarding side effects if possible,
            # but fetching usually implies caching in this system. 
            # For now let's just use it.
            
        if not flight or not flight.points:
             raise HTTPException(status_code=404, detail=f"Flight data not found for {flight_id}")

        results = pipeline.analyze(flight)
        return results
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

