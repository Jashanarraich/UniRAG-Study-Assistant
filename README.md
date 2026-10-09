# UniRAG – AI University Study Assistant

An intelligent, interactive question-answering study assistant designed for university students using **LangChain**, **Google Gemini**, **ChromaDB**, and **Retrieval-Augmented Generation (RAG)**.

---

## 📌 Project Overview
University students often struggle to locate exact concepts, formulas, or explanations buried within hundreds of pages of lecture slides and textbooks. Standard LLMs hallucinate or provide generic answers without referring to the specific curriculum.

**UniRAG** solves this problem by allowing students to:
1. Upload course materials in **PDF format** (text or scanned) or as **photos/images (`.png`, `.jpg`, `.jpeg`)** of question papers and handwritten notes.
2. Automatically perform **Multimodal AI OCR** using Google Gemini Vision when photos or scanned documents are uploaded.
3. Chunk and index materials into a high-dimensional vector space using ChromaDB.
4. Query the materials using natural language.
5. Receive answers strictly grounded in their syllabus along with source citations.

---

## 🏗️ Architecture & RAG Pipeline

```text
[Course PDF] 
      │
      ▼
1. Document Loader (pypdf: PyPDFLoader)
      │
      ▼
2. Text Splitter (RecursiveCharacterTextSplitter: chunk_size=1000, overlap=200)
      │
      ▼
3. Vector Embeddings (Google Gemini: gemini-embedding-001)
      │
      ▼
4. Vector Database (ChromaDB: vector_db/)
      │
      ├── Student Question ────────┐
      ▼                            ▼
5. Similarity Retriever ──────> Relevant Document Chunks
                                   │
                                   ▼
6. Prompt Template + LLM (Google Gemini: gemini-3.5-flash-lite)
                                   │
                                   ▼
7. Grounded Answer + Source Citations (Streamlit UI)
```

---

## 🛠️ Technology Stack

| Layer | Technology | Role |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Pure Python web interface with drag-and-drop file upload and chat UI. |
| **Orchestration** | [LangChain](https://www.langchain.com/) | Connects document loaders, vector stores, prompt templates, and the LLM. |
| **PDF Processing** | [PyPDF](https://pypi.org/project/pypdf/) | Extracts text and metadata (page numbers) from PDF documents. |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Local, serverless vector database for storing and querying embeddings. |
| **Embeddings & LLM** | [Google Gemini](https://ai.google.dev/) | `gemini-embedding-001` for vector representations and `gemini-3.5-flash-lite` for generation. |
| **Configuration** | [python-dotenv](https://pypi.org/project/python-dotenv/) | Securely loads API credentials from `.env`. |

---

## 📂 Project Directory Structure

```text
Minor Project/
├── data/
│   └── docs/            # Storage directory for uploaded student PDFs
├── src/
│   ├── __init__.py      # Package initializer
│   ├── config.py        # Centralized configurations, paths, and model names
│   ├── document_loader.py # PDF loading and recursive text chunking
│   ├── vector_store.py  # Chroma vector database and embedding logic
│   └── rag_pipeline.py  # LangChain prompt template, retriever, and LLM chain
├── app.py               # Streamlit user interface
├── requirements.txt     # Python package dependencies
├── .env                 # Secret API keys (ignored by Git)
├── .env.example         # Template for environment configuration
├── .gitignore           # Git ignore rules
└── README.md            # Comprehensive project documentation
```

---

## 🚀 How to Run the Project

### 1. Activate the Virtual Environment
Open PowerShell in the project directory:
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Verify Your API Key
Ensure your `GOOGLE_API_KEY` is present in `.env`:
```env
GOOGLE_API_KEY=AIzaSy...
```

### 3. Launch the Application
Run the Streamlit application:
```powershell
.\venv\Scripts\streamlit run app.py
```
Streamlit will automatically open `http://localhost:8501` in your browser.

---

## 🎓 FAQ

**Q1: What is Retrieval-Augmented Generation (RAG)?**
> **Answer:** RAG combines information retrieval with text generation. Instead of asking an LLM to rely only on its pre-trained memory, we retrieve relevant text chunks from custom documents (like our course PDF) and inject them into the LLM's prompt as context.

**Q2: Why do we split documents into chunks instead of passing the whole PDF?**
> **Answer:** Passing entire textbooks exceeds token limits, increases latency, and dilutes the LLM's focus. Chunking allows us to retrieve only the most relevant paragraphs with high precision.

**Q3: Why do we need chunk overlap?**
> **Answer:** If an important concept or sentence spans across the boundary of two chunks, overlap (e.g. 200 characters) ensures that context is not lost or broken in half.

**Q4: What is a vector embedding?**
> **Answer:** A vector embedding is a list of numbers representing the semantic meaning of a piece of text. Texts with similar meanings have high cosine similarity in vector space.

**Q5: How does your application handle scanned PDFs or photos of question papers?**
> **Answer:** Standard PDF libraries only extract selectable text and return blank results for scanned images. UniRAG implements Multimodal Vision OCR using Google Gemini. When an image or scanned page is detected, Gemini Vision transcribes all questions, mathematical formulas, and diagrams into text before passing them into the LangChain RAG pipeline.

