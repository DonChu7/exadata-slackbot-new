from __future__ import annotations
import io, os, traceback
from typing import Dict, Any, Tuple, List

# choose your provider; if you previously used OCI here, swap in ChatOCIGenAI
from langchain_ollama import OllamaLLM
from langchain_community.chat_models import ChatOCIGenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from pypdf import PdfReader

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
CHUNK_SIZE   = int(os.getenv("LC_SUM_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP= int(os.getenv("LC_SUM_CHUNK_OVERLAP", "200"))
SHORT_THRESH = int(os.getenv("LC_SUM_SHORT_THRESH", "1200"))

_llm = None
_llm_info = {"backend": "ollama", "model_id": OLLAMA_MODEL}


def _resolve_provider(model_id: str, explicit: str | None) -> str:
    if explicit:
        if explicit in {"xai"}:
            return "meta"
        return explicit
    mid = (model_id or "").lower()
    if mid.startswith("cohere"):
        return "cohere"
    if mid.startswith("xai") or mid.startswith("meta"):
        return "meta"
    if mid.startswith("openai"):
        return "meta"
    return "meta"


def _build_oci_llm() -> tuple[ChatOCIGenAI, Dict[str, str]] | None:
    model_id = os.getenv("SUMMARIZER_OCI_MODEL_ID")
    if not model_id:
        return None
    provider_override = os.getenv("SUMMARIZER_OCI_PROVIDER")
    provider_override = (provider_override or "").strip().lower() or None

    endpoint = (
        os.getenv("SUMMARIZER_OCI_ENDPOINT")
        or os.getenv("OCI_SUMMARIZER_ENDPOINT")
        or os.getenv("OCI_GENAI_ENDPOINT")
    )
    compartment = (
        os.getenv("SUMMARIZER_OCI_COMPARTMENT_ID")
        or os.getenv("OCI_SUMMARIZER_COMPARTMENT_ID")
        or os.getenv("OCI_COMPARTMENT_ID")
    )
    if not (endpoint and compartment):
        return None

    auth_type = os.getenv("SUMMARIZER_OCI_AUTH_TYPE") or os.getenv("OCI_AUTH_TYPE", "API_KEY")
    profile = os.getenv("SUMMARIZER_OCI_CONFIG_PROFILE") or os.getenv("OCI_CONFIG_PROFILE", "DEFAULT")
    temperature = float(os.getenv("SUMMARIZER_OCI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("SUMMARIZER_OCI_MAX_TOKENS", os.getenv("OCI_MAX_TOKENS", "2048")))

    if not (endpoint and compartment):
        return None
    provider = _resolve_provider(model_id, provider_override)
    model_kwargs = {"temperature": temperature, "max_tokens": max_tokens}
    print(
        "[Summarizer] Using OCI chat model:",
        f"provider={provider}",
        f"model_id={model_id}",
        f"endpoint={endpoint}",
        f"compartment={compartment}",
    )
    llm = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment,
        provider=provider,
        auth_type=auth_type,
        auth_profile=profile,
        model_kwargs=model_kwargs,
    )
    info = {"backend": "oci_chat", "provider": provider, "model_id": model_id}
    return llm, info


def _get_llm():
    global _llm, _llm_info
    if _llm is None:
        try:
            built = _build_oci_llm()
            if built:
                oci_llm, info = built
                _llm = oci_llm
                _llm_info = info
        except Exception as exc:
            print(f"[Summarizer] Failed to initialize OCI chat model ({exc}); falling back to Ollama.")
        if _llm is None:
            print(f"[Summarizer] Using Ollama model '{OLLAMA_MODEL}' for LangChain summarizer.")
            _llm = OllamaLLM(model=OLLAMA_MODEL)
            _llm_info = {"backend": "ollama", "model_id": OLLAMA_MODEL}
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
    try:
        chain = load_summarize_chain(_get_llm(), chain_type=chain_type)
        out = chain.invoke({"input_documents": docs})
        return {"summary": (out.get("output_text") or "").strip(), "chain_type": chain_type}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[:4000]}


def health() -> Dict[str, Any]:
    llm = _get_llm()
    info = dict(_llm_info)
    info["class"] = llm.__class__.__name__
    return info

def _pdf_to_text_pages(pdf_bytes: bytes) -> Tuple[str, int]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = len(reader.pages)
    parts = [(p.extract_text() or "") for p in reader.pages]
    return ("\n".join(parts).strip(), pages)

def lc_summarize_pdf_path(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {"error": f"file not found: {path}"}
    with open(path, "rb") as f:
        data = f.read()
    text, pages = _pdf_to_text_pages(data)
    if len(text) < 200:
        return {"summary": "[No extractable text — likely a scanned/image-only PDF.]", "pages": pages, "chain_type": "n/a"}
    res = lc_summarize_text(text); res["pages"] = pages; return res
