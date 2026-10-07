import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file (if running locally)
load_dotenv(dotenv_path=BASE_DIR / ".env")


def get_google_api_key() -> str:
    """
    Dynamically retrieves the Google Gemini API key from:
    1. OS Environment / .env
    2. Streamlit Cloud Secrets (st.secrets)
    3. User interactive input stored in st.session_state
    """
    # 1. Environment variable
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if key:
        return key

    # 2. Streamlit context (Secrets and Session State)
    try:
        import streamlit as st
        # Check UI session state if user entered it in sidebar
        if "user_api_key" in st.session_state and st.session_state["user_api_key"]:
            return st.session_state["user_api_key"].strip()

        # Check Streamlit Cloud Secrets
        if hasattr(st, "secrets"):
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    return ""


# Keep for backward-compatibility
GOOGLE_API_KEY = get_google_api_key()

# Directory Paths
DATA_DIR = BASE_DIR / "data" / "docs"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# RAG & Chunking Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Model configurations
EMBEDDING_MODEL_NAME = "gemini-embedding-001"
LLM_MODEL_NAME = "gemini-3.5-flash-lite"
