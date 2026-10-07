from typing import List, Dict, Any
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import get_google_api_key, LLM_MODEL_NAME
from src.vector_store import get_retriever


def get_llm(model_name: str = LLM_MODEL_NAME, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Initializes the Gemini LLM for question answering.
    Temperature is kept low (0.2) to ensure factual, grounded responses based on student notes.
    """
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def format_context_docs(docs: List[Document]) -> str:
    """
    Formats retrieved document chunks into a single readable string for the prompt.
    Includes page numbers so the LLM understands the document context.
    """
    formatted_chunks = []
    for doc in docs:
        page = doc.metadata.get("page", 0)
        # PyPDF uses 0-based page indexing; add 1 for human-friendly page numbers
        human_page = page + 1 if isinstance(page, int) else page
        source = Path(doc.metadata.get("source", "Document")).name
        formatted_chunks.append(
            f"[Source: {source}, Page: {human_page}]\n{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


def extract_text_from_llm_response(content: Any) -> str:
    """
    Safely extracts plain string text from LLM response content,
    handling both string outputs and structured list formats.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts) if parts else str(content)
    return str(content)


class UniRAGPipeline:
    """
    End-to-end RAG Pipeline using LangChain.
    Coordinates between:
    1. Chroma Retriever (finding relevant chunks)
    2. Prompt Template (structuring the university assistant instructions)
    3. Google Gemini LLM (generating answers grounded in context)
    """

    def __init__(self, vector_store: Chroma, k: int = 4):
        self.vector_store = vector_store
        self.retriever = get_retriever(vector_store, k=k)
        self.llm = get_llm()

        # Prompt template instructing the model to act as a university study assistant
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are UniRAG, an intelligent AI University Study Assistant.\n"
                "Your role is to help university students understand their course material accurately.\n\n"
                "GUIDELINES:\n"
                "1. Answer the question strictly using the provided context from the student's study material.\n"
                "2. If the context does not contain enough information to answer the question, clearly state: "
                "'I could not find the answer to this question in the uploaded study materials.' "
                "Do NOT hallucinate or extrapolate beyond the provided text.\n"
                "3. Use a clear, well-structured format (bullet points, numbered lists, or bold highlights) to explain academic concepts.\n"
                "4. Be concise, polite, and encouraging.\n\n"
                "CONTEXT FROM STUDY MATERIALS:\n"
                "{context}"
            ),
            ("human", "{question}"),
        ])

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline for a given question:
        1. Retrieves the top relevant chunks.
        2. Formats the prompt with context and question.
        3. Invokes the LLM through LangChain.
        4. Extracts and returns the answer alongside source references.
        """
        # 1. Retrieve relevant chunks using the retriever
        retrieved_docs = self.retriever.invoke(question)

        if not retrieved_docs:
            return {
                "answer": "No relevant study material found in the database. Please make sure a PDF is uploaded and processed.",
                "sources": [],
                "raw_docs": [],
            }

        # 2. Format chunks into context
        context_str = format_context_docs(retrieved_docs)

        # 3. Create the prompt messages and invoke the LLM
        messages = self.prompt_template.format_messages(
            context=context_str,
            question=question,
        )
        llm_response = self.llm.invoke(messages)
        answer_text = extract_text_from_llm_response(llm_response.content)

        # 4. Prepare structured source citations for the UI
        sources = []
        for doc in retrieved_docs:
            page = doc.metadata.get("page", 0)
            human_page = page + 1 if isinstance(page, int) else page
            sources.append({
                "file_name": Path(doc.metadata.get("source", "Document")).name,
                "page": human_page,
                "preview": doc.page_content[:250].strip() + ("..." if len(doc.page_content) > 250 else ""),
            })

        return {
            "answer": answer_text,
            "sources": sources,
            "raw_docs": retrieved_docs,
        }
