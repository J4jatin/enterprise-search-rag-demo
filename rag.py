import os
import logging
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from documents import documents

# Load environment variables from .env file
load_dotenv()

# Set up logging so we can see what is happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the embedding model
logger.info("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Embedding model loaded successfully!")

# Set up Groq client using API key from .env
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Pre-compute document vectors once at startup
# This is a key performance optimization
document_texts = [doc["content"] for doc in documents]
document_vectors = model.encode(document_texts)
logger.info(f"Indexed {len(documents)} documents successfully!")


def search(question: str, top_k: int = 3) -> list:
    """
    Retrieve most relevant documents using semantic similarity.
    
    This is the RETRIEVAL step of RAG:
    - Convert question to vector
    - Compare against all document vectors
    - Return top_k most similar documents
    """
    # Convert question to vector
    question_vector = model.encode([question])
    
    # Calculate cosine similarity between question and all documents
    similarities = cosine_similarity(question_vector, document_vectors)[0]
    
    # Get top_k most similar document indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build results list
    results = []
    for idx in top_indices:
        results.append({
            "title": documents[idx]["title"],
            "content": documents[idx]["content"],
            "similarity_score": round(float(similarities[idx]), 4)
        })
    
    return results


def generate_answer(question: str, context_chunks: list) -> str:
    """
    Generate a final answer using Groq LLM.
    
    This is the GENERATION step of RAG:
    - Take retrieved document chunks as context
    - Inject context into prompt (context-injection)
    - Send to Groq LLM for answer generation
    - Return grounded, accurate answer
    """
    # Build context from retrieved documents
    context = "\n\n".join([
        f"[{chunk['title']}]: {chunk['content']}"
        for chunk in context_chunks
    ])
    
    # Prompt engineering — explicit instructions to the LLM
    prompt = f"""You are an enterprise search assistant for SAP LeanIX.
Your job is to answer questions based ONLY on the provided context.
Do not use any outside knowledge. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    # Call Groq API with the prompt
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.1
    )
    
    return response.choices[0].message.content