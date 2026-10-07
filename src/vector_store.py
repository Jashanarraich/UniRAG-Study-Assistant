from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    get_google_api_key,
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_DIR,
)


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Gemini embeddings model.
    Embeddings transform text into high-dimensional numerical vectors that capture meaning.
    """
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please add your Gemini API key in the sidebar, .env, or Streamlit secrets."
        )

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=api_key,
    )


def create_vector_store(
    chunks: List[Document],
    persist_directory: Optional[Path] = None,
    collection_name: str = "unirag_docs",
    batch_size: int = 15,
) -> Chroma:
    """
    Takes document chunks, cleans them, computes embeddings in safe batches of 15,
    and stores them in a local ChromaDB database.
    """
    # 1. Filter out empty or minimal whitespace chunks
    valid_chunks = [c for c in chunks if c.page_content and len(c.page_content.strip()) > 3]
    if not valid_chunks:
        raise ValueError("No readable text found in the uploaded documents to index.")

    persist_dir = str(persist_directory or VECTOR_DB_DIR)
    embeddings = get_embeddings_model()

    # 2. Initialize Chroma collection
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    # 3. Add documents in safe batches to avoid Google API request payload limits
    for i in range(0, len(valid_chunks), batch_size):
        batch = valid_chunks[i : i + batch_size]
        vector_store.add_documents(batch)

    return vector_store


def load_vector_store(
    persist_directory: Optional[Path] = None,
    collection_name: str = "unirag_docs",
) -> Chroma:
    """
    Loads an existing ChromaDB vector database from local storage.
    """
    persist_dir = str(persist_directory or VECTOR_DB_DIR)
    embeddings = get_embeddings_model()

    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    """
    Converts a Chroma vector store into a LangChain Retriever.
    'k' defines how many top-matching text chunks will be fetched for a student question.
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
