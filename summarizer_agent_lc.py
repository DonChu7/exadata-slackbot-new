# summarizer_agent_lc.py
from __future__ import annotations
import io, os
import base64 as b64
from typing import Dict, Any, Optional, Tuple, List

from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from pypdf import PdfReader

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
CHUNK_SIZE   = int(os.getenv("LC_SUM_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP= int(os.getenv("LC_SUM_CHUNK_OVERLAP", "200"))
SHORT_THRESH = int(os.getenv("LC_SUM_SHORT_THRESH", "1200"))

_llm: Optional[OllamaLLM] = None

def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model=OLLAMA_MODEL)
    return _llm

def _split_docs(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return [Document(page_content=c) for c in splitter.split_text(text)]

def lc_summarize_text(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"error": "empty text"}
    docs = _split_docs(text)
    chain_type = "stuff" if (len(text) <= SHORT_THRESH or len(docs) <= 1) else "map_reduce"
    chain = load_summarize_chain(_get_llm(), chain_type=chain_type)
    out = chain.invoke({"input_documents": docs})
    return {"summary": (out.get("output_text") or "").strip(), "chain_type": chain_type}

def _pdf_to_text_pages(pdf_bytes: bytes) -> Tuple[str, int]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = len(reader.pages)
    parts = [(p.extract_text() or "") for p in reader.pages]
    return ("\n".join(parts).strip(), pages)

def lc_summarize_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    text, pages = _pdf_to_text_pages(pdf_bytes)
    if len(text) < 200:
        return {"summary": "[No extractable text — likely a scanned/image-only PDF.]", "pages": pages, "chain_type": "n/a"}
    res = lc_summarize_text(text)
    res["pages"] = pages
    return res

def lc_summarize_pdf_path(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {"error": f"file not found: {path}"}
    with open(path, "rb") as f:
        return lc_summarize_pdf_bytes(f.read())

def summarize_pdf_b64(b64_str: str, style_hint: str | None = None) -> Dict[str, Any]:
    try:
        data = base64.b64decode(b64_str)
    except Exception as e:
        return {"error": f"invalid base64: {e}"}
    return lc_summarize_pdf_bytes(data)

def health() -> Dict[str, Any]:
    try:
        _ = _get_llm()
        return {"ok": True, "model": OLLAMA_MODEL}
    except Exception as e:
        return {"ok": False, "error": str(e)}
