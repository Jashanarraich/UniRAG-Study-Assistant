from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import GOOGLE_API_KEY, LLM_MODEL_NAME
from src.vector_store import get_retriever


def get_llm(model_name: str = LLM_MODEL_NAME, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Initializes the Gemini LLM for question answering.
    Temperature is kept low (0.2) to ensure factual, grounded responses based on student notes.
    """
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
        vertexai=False,
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


def solve_direct_question(question: str) -> str:
    """
    Answers and solves student questions directly even when no course documents
    are uploaded, acting as a comprehensive academic tutor.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are UniRAG, an intelligent AI University Study Assistant and Exam Tutor.\n"
            "Your role is to help students solve complex exam questions, clarify difficult concepts, "
            "and provide complete, step-by-step academic solutions.\n\n"
            "GUIDELINES:\n"
            "1. Provide a comprehensive, rigorous, and step-by-step solution.\n"
            "2. If it is a mathematical, algorithmic, or theoretical problem, show all working and derivations.\n"
            "3. Format cleanly using bold headings, numbered steps, or bullet points.\n"
            "4. Be encouraging, clear, and academic."
        ),
        ("human", "{question}"),
    ])
    messages = prompt.format_messages(question=question)
    response = llm.invoke(messages)
    return extract_text_from_llm_response(response.content)


class UniRAGPipeline:
    """
    End-to-end RAG Pipeline using LangChain.
    Coordinates between:
    1. Chroma Retriever (finding relevant chunks)
    2. Prompt Template (structuring the university assistant instructions)
    3. Google Gemini LLM (generating answers grounded in context and solving questions)
    """

    def __init__(self, vector_store: Chroma, k: int = 4):
        self.vector_store = vector_store
        self.retriever = get_retriever(vector_store, k=k)
        self.llm = get_llm()

        # Prompt template instructing the model to act as a university study assistant
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are UniRAG, an expert AI University Study Assistant and Problem Solver.\n"
                "Your role is to help students understand their course materials and solve exam questions.\n\n"
                "INSTRUCTIONS:\n"
                "1. ALIGN WITH COURSE NOTES: If the provided context covers the question, base your explanation "
                "primarily on the student's uploaded material and syllabus conventions.\n"
                "2. ALWAYS SOLVE AND EXPLAIN: If the question asks to solve a problem (e.g. from an uploaded question paper, "
                "past exam, numerical, or conceptual query) and the complete solution is NOT written out in the notes, "
                "DO NOT refuse to answer! Provide the FULL, step-by-step solution, derivation, code, or explanation using "
                "rigorous academic principles.\n"
                "3. ACADEMIC STRUCTURE: Structure your answer logically with bold headings, numbered steps, or bullet points.\n"
                "4. Be thorough, accurate, and encouraging.\n\n"
                "CONTEXT FROM UPLOADED STUDY MATERIALS:\n"
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

        # 2. Format chunks into context (if empty, solve directly)
        if not retrieved_docs:
            direct_ans = solve_direct_question(question)
            return {
                "answer": direct_ans,
                "sources": [],
                "raw_docs": [],
            }

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
