import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
USE_OPENAI = OPENAI_API_KEY is not None and OPENAI_API_KEY.strip() != ""

# Database
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/customer_support")

# ChromaDB
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./database/chroma_db")

# Embeddings
EMBEDDING_MODEL = "all-mpnet-base-v2"

# RAG Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 5


def get_llm():
    """Return the configured LLM instance (OpenAI or Ollama)."""
    if USE_OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


def get_embeddings():
    """Return the HuggingFace embeddings model."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
