import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag import search, generate_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Enterprise Search RAG Demo",
    description="AI-powered semantic search API with RAG architecture. Built by Jattin Shah - MSc Applied AI, TU Dresden",
    version="2.0.0"
)


# Request model — Pydantic validates incoming data
class SearchRequest(BaseModel):
    question: str
    top_k: int = 3


# Response model
class SearchResponse(BaseModel):
    question: str
    answer: str
    sources: list
    model: str


# ── ENDPOINT 1: Home ─────────────────────────────────
@app.get("/")
def home():
    """Home endpoint — confirms API is running"""
    return {
        "message": "Enterprise Search RAG Demo is running!",
        "version": "2.0.0",
        "author": "Jattin Shah — MSc Applied AI, TU Dresden",
        "endpoints": {
            "search": "POST /search",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }


# ── ENDPOINT 2: Search ───────────────────────────────
@app.post("/search")
def search_documents(request: SearchRequest):
    """
    Main RAG search endpoint.

    Full pipeline:
    1. RETRIEVE — find relevant documents using vector similarity
    2. AUGMENT  — inject retrieved docs as context into prompt
    3. GENERATE — Groq LLM generates grounded answer
    """
    try:
        logger.info(f"Processing query: {request.question}")

        # Step 1 — RETRIEVE relevant documents
        retrieved_docs = search(request.question, request.top_k)

        # Step 2 & 3 — AUGMENT + GENERATE answer using Groq LLM
        answer = generate_answer(request.question, retrieved_docs)

        logger.info("Answer generated successfully")

        return {
            "question": request.question,
            "answer": answer,
            "sources": retrieved_docs,
            "model": "llama-3.1-8b-instant"
        }

    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ── ENDPOINT 3: Health Check ─────────────────────────
@app.get("/health")
def health():
    """
    Health check endpoint.
    Used by Kubernetes liveness probes to verify service is running.
    """
    return {"status": "healthy", "version": "2.0.0"}