import os
from dotenv import load_dotenv

load_dotenv()

class RAGConfig:
    PDF_PATH = "tm_5_692_1.pdf"
    PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME = "tm-5-692-1-v5" 
    PINECONE_ENV        = "us-east-1"

    EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
    RERANKER_MODEL_NAME  = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ← changed

    HYBRID_ALPHA  = 0.5
    TOP_K_HYBRID  = 30   
    TOP_K_FINAL   = 5   

    GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL      = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS = 2048