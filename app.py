import os
import shutil
from pathlib import Path
import streamlit as st

from src.config import DATA_DIR, VECTOR_DB_DIR, GOOGLE_API_KEY
from src.document_loader import load_and_split_document
from src.vector_store import create_vector_store, load_vector_store
from src.rag_pipeline import UniRAGPipeline

# Configure Streamlit page
st.set_page_config(
    page_title="UniRAG – AI University Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


def reset_knowledge_base():
    """Clears uploaded documents, local vector store, and chat history."""
    st.session_state.messages = []
    st.session_state.rag_pipeline = None
    st.session_state.indexed_files = []
    st.session_state.total_chunks = 0

    if VECTOR_DB_DIR.exists():
        try:
            shutil.rmtree(VECTOR_DB_DIR)
        except Exception:
            pass

    if DATA_DIR.exists():
        for file in DATA_DIR.glob("*"):
            if file.name != ".gitkeep":
                try:
                    file.unlink()
                except Exception:
                    pass


# --- SIDEBAR: Document Ingestion and Controls ---
with st.sidebar:
    st.title("🎓 UniRAG Controls")
    st.caption("AI-Powered Study Assistant using LangChain & RAG")

    st.markdown("---")

    # API Key Status indicator
    if GOOGLE_API_KEY:
        st.success("✅ Google Gemini API Key Active")
    else:
        st.error("❌ GOOGLE_API_KEY is missing in your .env file.")

    st.markdown("### 1. Upload Study Materials")
    uploaded_files = st.file_uploader(
        "Upload Course PDFs or Question Paper Photos:",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload lecture notes, textbooks (PDF) or photos of question papers (PNG, JPG).",
    )

    process_button = st.button("⚡ Process & Index Documents", type="primary", use_container_width=True)

    if process_button:
        if not GOOGLE_API_KEY:
            st.error("Please add your GOOGLE_API_KEY in the .env file first.")
        elif not uploaded_files:
            st.warning("Please upload at least one PDF or photo.")
        else:
            with st.spinner("Processing documents into chunks and embeddings (AI OCR enabled)..."):
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                all_chunks = []
                indexed_names = []

                for uploaded_file in uploaded_files:
                    # Save uploaded file to disk
                    file_path = DATA_DIR / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load and split into chunks (runs OCR for images/scans automatically)
                    chunks = load_and_split_document(str(file_path))
                    all_chunks.extend(chunks)
                    indexed_names.append(uploaded_file.name)

                if all_chunks:
                    # Create Chroma vector database and initialize RAG pipeline
                    vector_store = create_vector_store(all_chunks)
                    st.session_state.rag_pipeline = UniRAGPipeline(vector_store)
                    st.session_state.indexed_files = indexed_names
                    st.session_state.total_chunks = len(all_chunks)
                    st.success(f"Successfully indexed {len(all_chunks)} chunks from {len(indexed_names)} file(s)!")
                else:
                    st.error("Could not extract any readable text from the uploaded file(s).")

    st.markdown("---")
    st.markdown("### 📊 Index Status")
    if st.session_state.rag_pipeline is not None:
        st.info(
            f"**Indexed Files:** {len(st.session_state.indexed_files)}\n\n"
            f"**Total Chunks:** {st.session_state.total_chunks}\n\n"
            + "\n".join([f"- 📄 `{f}`" for f in st.session_state.indexed_files])
        )
    else:
        st.write("No documents indexed yet.")

    st.markdown("---")
    if st.button("🗑️ Reset All Notes & Chat", use_container_width=True):
        reset_knowledge_base()
        st.rerun()


# --- MAIN PANEL: Chat Interface ---
st.header("UniRAG – AI University Study Assistant")
st.markdown(
    "Ask conceptual questions, request summaries, or clarify difficult topics from your uploaded course materials. "
    "Every answer is grounded strictly in your notes with page citations."
)

# If no notes uploaded yet, show quick-start guide
if st.session_state.rag_pipeline is None:
    st.info(
        "👈 **Get Started:**\n"
        "1. Upload your course PDFs or photos of question papers using the sidebar.\n"
        "2. Click **'Process & Index Documents'**.\n"
        "3. Ask your questions here in the chat!"
    )

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display source citations accordion for assistant answers
        if message.get("sources"):
            with st.expander("📚 Referenced Sources & Citations"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(
                        f"**{idx}. {src['file_name']}** — *Page {src['page']}*\n\n"
                        f"> {src['preview']}"
                    )

# Chat Input
user_question = st.chat_input("Ask a question about your study material...")

if user_question:
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate assistant answer
    with st.chat_message("assistant"):
        if st.session_state.rag_pipeline is None:
            warning_msg = "Please upload and process your study material using the sidebar first."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        else:
            with st.spinner("Searching study material and generating answer..."):
                response = st.session_state.rag_pipeline.query(user_question)
                answer = response["answer"]
                sources = response.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Referenced Sources & Citations"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(
                                f"**{idx}. {src['file_name']}** — *Page {src['page']}*\n\n"
                                f"> {src['preview']}"
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
