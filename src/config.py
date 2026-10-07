import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file (if running locally)
load_dotenv(dotenv_path=BASE_DIR / ".env")


def clean_api_key(val) -> str:
    """Strips accidental quotes, spaces, and formatting characters from API key string."""
    if not val:
        return ""
    s = str(val).strip()
    for quote in ['"', "'", "`"]:
        if s.startswith(quote) and s.endswith(quote):
            s = s[1:-1].strip()
    return s


def get_google_api_key() -> str:
    """
    Dynamically retrieves the Google Gemini API key:
    1. User custom override in st.session_state (highest priority)
    2. Streamlit Cloud Secrets (st.secrets)
    3. OS Environment / .env
    """
    # 1. Custom user override entered in UI
    try:
        import streamlit as st
        if "user_api_key" in st.session_state and st.session_state["user_api_key"]:
            cleaned = clean_api_key(st.session_state["user_api_key"])
            if cleaned:
                return cleaned
    except Exception:
        pass

    # 2. Streamlit Cloud Secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            cleaned = clean_api_key(st.secrets["GOOGLE_API_KEY"])
            if cleaned:
                return cleaned
    except Exception:
        pass

    # 3. Environment variable / local .env
    key = clean_api_key(os.getenv("GOOGLE_API_KEY", ""))
    if key:
        return key

    return ""


def has_default_api_key() -> bool:
    """Returns True if a server-level default API key exists in secrets or .env."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            if str(st.secrets["GOOGLE_API_KEY"]).strip():
                return True
    except Exception:
        pass
    return bool(os.getenv("GOOGLE_API_KEY", "").strip())



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
