import io
import os
from typing import List, Union
from pathlib import Path
from PIL import Image
import pypdf
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, get_google_api_key, LLM_MODEL_NAME


def get_genai_client() -> genai.Client:
    """Initializes Google GenAI Client for vision and OCR tasks."""
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please provide it in the sidebar, .env, or Streamlit secrets.")
    return genai.Client(api_key=api_key)


def extract_text_from_image(image_input: Union[Image.Image, bytes, Path]) -> str:
    """
    Uses Google Gemini's multimodal vision model to accurately transcribe
    text, questions, formulas, and diagrams from an image or photo.
    """
    client = get_genai_client()

    if isinstance(image_input, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input))
    else:
        img = image_input

    prompt = (
        "You are an expert academic assistant and OCR scanner.\n"
        "Transcribe ALL readable text, questions, options, headings, numbers, and mathematical equations "
        "from this photo/document verbatim.\n"
        "Maintain the original structure and question numbering (e.g. Q1, Q2, a, b, c) accurately.\n"
        "If there are tables or diagrams, transcribe their text and provide a concise textual explanation."
    )

    try:
        response = client.models.generate_content(
            model=LLM_MODEL_NAME,
            contents=[img, prompt],
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        print(f"OCR transcription warning: {e}")
        return ""


def load_image_file(file_path: Union[str, Path]) -> List[Document]:
    """
    Loads a standalone image (JPG, PNG, JPEG) of notes or a question paper,
    runs Gemini Vision OCR, and returns it as a LangChain Document.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    text = extract_text_from_image(path)
    if not text:
        text = f"[No readable text could be recognized from image {path.name}]"

    return [
        Document(
            page_content=text,
            metadata={"source": path.name, "page": 1, "type": "image"},
        )
    ]


def load_pdf(file_path: Union[str, Path]) -> List[Document]:
    """
    Loads a PDF file. If a page has digital selectable text, it extracts it directly.
    If a page is scanned/photo (empty or minimal text), it automatically extracts the
    embedded page image and performs Gemini Vision OCR.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    documents: List[Document] = []
    reader = pypdf.PdfReader(str(path))

    for page_idx, page in enumerate(reader.pages):
        # 1. Attempt digital text extraction
        extracted_text = (page.extract_text() or "").strip()

        # 2. Check if page appears to be a scanned photo (very short or empty text)
        if len(extracted_text) < 40 and len(page.images) > 0:
            # Scanned page detected: extract the largest image and run OCR
            ocr_texts = []
            for img_obj in page.images:
                try:
                    ocr_res = extract_text_from_image(img_obj.data)
                    if ocr_res:
                        ocr_texts.append(ocr_res)
                except Exception as img_err:
                    print(f"Error processing page {page_idx + 1} image: {img_err}")

            if ocr_texts:
                extracted_text = "\n\n".join(ocr_texts)

        if extracted_text:
            documents.append(
                Document(
                    page_content=extracted_text,
                    metadata={"source": path.name, "page": page_idx},
                )
            )

    return documents


def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Dispatches file loading based on extension:
    Supports PDFs (.pdf) and Photos/Images (.png, .jpg, .jpeg, .webp).
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        return load_image_file(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported: PDF, PNG, JPG, JPEG.")


def split_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Splits loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
    Preserves page number and file metadata so sources can be properly cited.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def load_and_split_document(
    file_path: Union[str, Path],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Loads any supported document or photo and splits it into searchable chunks.
    """
    docs = load_document(file_path)
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# Backward-compatible alias for existing code
load_and_split_pdf = load_and_split_document
