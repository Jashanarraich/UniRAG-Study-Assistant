import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# API Key: Read from local .env or Streamlit Cloud Secrets
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            GOOGLE_API_KEY = str(st.secrets["GOOGLE_API_KEY"]).strip().strip('"').strip("'")
    except Exception:
        pass

# Ensure os.environ also has it set for any underlying Google SDKs
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Directory Paths
DATA_DIR = BASE_DIR / "data" / "docs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# RAG & Chunking Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Model configurations
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-3.5-flash-lite"
