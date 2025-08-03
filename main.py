# main.py
import os
import requests
import numpy as np
import google.generativeai as genai
import asyncio  # <-- Import asyncio for parallel processing
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
app = FastAPI(title="Optimized Gemini Q&A Service")
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
   """Generates embeddings for a list of texts using Google's API."""
   try:
       response = genai.embed_content(model='models/embedding-001', content=texts, task_type=task_type)
       return response['embedding']
   except Exception as e:
       print(f"Gemini embedding API call failed: {e}")
       raise HTTPException(status_code=500, detail="Failed to generate embeddings from Google.")
# Replace the old generate_answer_async function with this new version

async def generate_answer_async(question: str, context: str):
    """Asynchronous function to generate a single answer using Gemini Flash with an improved prompt."""
    # Using gemini-1.5-flash-latest for speed
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # --- START OF NEW, IMPROVED PROMPT ---
    # The new 'prompt' variable inside your generate_answer_async function

    prompt = f"""
    You are an AI assistant specializing in detailed policy and contract analysis. 
    Your task is to provide a clear, brief and factual answer to the `QUESTION` based *only* on the `CONTEXT` provided.

    **Instructions for your response:**

    1.  **Be Subtle:** If the question can be answered in a single line, try to answer it in a single sentence only. Add lines only when necessary information about the points to answer the question aren't included in the first sentence. 

    2.  **Use Complete Sentences:** Always formulate your answer in formal, well-structured sentences. Do not use bullet points unless the source text uses them.

    3.  **Answer Directly:**
       * For questions that can be answered with a "yes" or "no", you must start your response immediately with "Yes," or "No," followed by a very short explanation of 1 or 2 lines.
       * **Crucially, do NOT use any introductory phrases or preambles.** Avoid phrases like "According to the provided document...", "The context states that...", or "Based on the text...".

    4.  **Handle Missing Information:** If the answer to the `QUESTION` absolutely cannot be found in the `CONTEXT`, you must respond with the single phrase: "The information for this question is not available in the provided text."

    CONTEXT:
    ---
    {context}
    ---

    QUESTION:
    {question}

    ANSWER:
    """
   
    # --- END OF NEW, IMPROVVED PROMPT ---
    
    try:
        # Use the async version of the generation method
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini async API call failed: {e}")
        # Return error message instead of crashing the whole request
        return f"Error generating answer for this question: {e}"

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
        genai.configure(api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Google client: {e}")

    if not request.documents or not request.questions:
        raise HTTPException(status_code=400, detail="Missing 'documents' or 'questions' field in the request.")

    # --- Main Logic ---
    # 1. Process document (still synchronous)
    chunks = process_document(request.documents)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not process document from URL.")

    # 2. Batch embed all document chunks at once (synchronous)
    chunk_embeddings = get_gemini_embeddings(chunks, "RETRIEVAL_DOCUMENT")

    # 3. Batch embed all questions at once (synchronous)
    questions = request.questions
    question_embeddings = get_gemini_embeddings(questions, "RETRIEVAL_QUERY")

    # 4. Create a list of async tasks to run in parallel
    tasks = []
    for i, question in enumerate(questions):
        question_embedding = question_embeddings[i]
        
        # Perform search (this is fast, so no need for async)
        similarities = [np.dot(question_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
        top_k = 5
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        context = "\n\n---\n\n".join([chunks[i] for i in top_indices])
        
        # Create an async task for the slow network call (LLM generation)
        task = generate_answer_async(question, context)
        tasks.append(task)
        
    # 5. Run all answer generation tasks concurrently
    all_answers = await asyncio.gather(*tasks)
            
    return HackRxResponse(answers=all_answers)

# Include the router in the main app
app.include_router(router)

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "info": "Optimized Gemini Q&A Endpoint"}
