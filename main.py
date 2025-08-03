# main.py
import os
import requests
import numpy as np
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Header, APIRouter
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from pypdf import PdfReader

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: Optional[str] = None
    questions: Optional[List[str]] = None

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App and Router Setup ---
app = FastAPI(title="Gemini 1.5 Pro Q&A Service")
router = APIRouter(prefix="/api/v1")

# --- Helper Functions ---
def process_document(url: str):
    """Downloads and chunks the document."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with BytesIO(response.content) as pdf_file:
            reader = PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        
        chunks = []
        chunk_size, chunk_overlap = 2000, 300
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - chunk_overlap
        return [chunk for chunk in chunks if chunk.strip()]
    except Exception as e:
        print(f"Error processing document: {e}")
        return []

def get_gemini_embeddings(texts: List[str], task_type: str):
   """Generates embeddings using Google's API."""
   try:
       response = genai.embed_content(model='models/embedding-001', content=texts, task_type=task_type)
       return response['embedding']
   except Exception as e:
       print(f"Gemini embedding API call failed: {e}")
       raise HTTPException(status_code=500, detail="Failed to generate embeddings from Google.")


def generate_answer_with_gemini(question: str, context: str):
    """Generates an answer using the Gemini 1.5 Pro model."""
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"""
    You are an expert Q&A system. Your answers must be based *only* on the provided context.
    If the answer cannot be found in the context, state that clearly and concisely. Don't mention that you have read the document, it should be a direct one liner answer.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        if "API_KEY_INVALID" in str(e).upper():
             raise HTTPException(status_code=401, detail="Authentication failed with Google. Check your API key.")
        raise HTTPException(status_code=500, detail="Failed to generate answer from Google.")

# --- API Endpoint using the Router ---
@router.post("/hackrx/run", response_model=HackRxResponse, tags=["HackRx"])
async def run_submission(
    request: HackRxRequest,
    authorization: Optional[str] = Header(None)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    
    api_key = authorization.split(" ")[1]
    if not api_key:
        raise HTTPException(status_code=401, detail="Bearer token is empty.")

    try:
        # Configure the Google client for this specific request
        genai.configure(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Google client: {e}")

    if not request.documents or not request.questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions' field in the request.")

    try:
        chunks = process_document(request.documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not process document from URL.")

        # Create embeddings for all chunks via Gemini API
        chunk_embeddings = get_gemini_embeddings(chunks, "RETRIEVAL_DOCUMENT")

        all_answers = []
        for question in request.questions:
            # Create embedding for the question
            question_embedding = get_gemini_embeddings([question], "RETRIEVAL_QUERY")[0]
            
            # Perform semantic search
            similarities = [np.dot(question_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
            top_k = 5
            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
            context = "\n\n---\n\n".join([chunks[i] for i in top_indices])
            
            # Generate answer with Gemini 1.5 Pro using the retrieved context
            answer = generate_answer_with_gemini(question, context)
            all_answers.append(answer)
            
        return HackRxResponse(answers=all_answers)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(router)

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "model_info": "This endpoint uses Google Gemini for embeddings and generation."}
