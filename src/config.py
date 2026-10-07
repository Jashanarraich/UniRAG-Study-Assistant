import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# API Keys (Reads from local .env or Streamlit Cloud Secrets)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass

# Directory Paths
DATA_DIR = BASE_DIR / "data" / "docs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# RAG & Chunking Parameters
# Chunk size: Number of characters per split
CHUNK_SIZE = 1000

# Chunk overlap: Shared characters between consecutive chunks to prevent losing context
CHUNK_OVERLAP = 200

# Model configurations
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-3.5-flash-lite"

