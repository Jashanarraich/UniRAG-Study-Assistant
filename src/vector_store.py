from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL_NAME,
    VECTOR_DB_DIR,
)


def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """
    Initializes and returns the Google Gemini embeddings model.
    Embeddings transform text into high-dimensional numerical vectors that capture meaning.
    """
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment or Streamlit Secrets. "
            "Please configure your Gemini API key."
        )

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
    )


def create_vector_store(
    chunks: List[Document],
    persist_directory: Optional[Path] = None,
    collection_name: str = "unirag_docs",
) -> Chroma:
    """
    Takes document chunks, computes their embeddings, and stores them in a local ChromaDB database.
    """
    # Filter out empty or whitespace-only chunks
    valid_chunks = [c for c in chunks if c.page_content and len(c.page_content.strip()) > 3]
    if not valid_chunks:
        raise ValueError("No readable text found in the uploaded documents to index.")

    persist_dir = str(persist_directory or VECTOR_DB_DIR)
    embeddings = get_embeddings_model()

    vector_store = Chroma.from_documents(
        documents=valid_chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )
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
