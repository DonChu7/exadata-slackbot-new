from __future__ import annotations
import os, oracledb
from typing import Any, Dict, List, Optional

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOCIGenAI
from langchain_community.embeddings.oracleai import OracleEmbeddings
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy

_CONN: Optional[oracledb.Connection] = None
_EMB: Optional[OracleEmbeddings] = None
_VS: Optional[OracleVS] = None
_RETRIEVER = None
_QA = None
_CONFIG: Dict[str, Any] = {}

_DEFAULT_TABLE = "SLACKBOT_VECTORS"
_DEFAULT_MODEL = "ALL_MINILM_L12_V2"


def _make_rag_llm() -> ChatOCIGenAI:
    explicit_provider = (
        os.getenv("RAG_OCI_PROVIDER")
        or os.getenv("OCI_RAG_PROVIDER")
        or os.getenv("OCI_GENAI_PROVIDER")
        or ""
    )
    model_id = (
        os.getenv("RAG_OCI_MODEL_ID")
        or os.getenv("OCI_RAG_MODEL_ID")
        or os.getenv("OCI_GENAI_MODEL_ID")
    )
    if not model_id:
        raise RuntimeError("Missing RAG_OCI_MODEL_ID / OCI_GENAI_MODEL_ID for RAG LLM.")

    provider = explicit_provider.strip().lower()
    if not provider:
        mid = model_id.lower()
        if mid.startswith("cohere"):
            provider = "cohere"
        elif mid.startswith("xai"):
            provider = "meta"  # mimic ExaCopilot fallback
        elif mid.startswith("openai"):
            provider = "meta"  # OCI currently maps OpenAI models via meta bridge
        else:
            provider = "meta"

    endpoint = (
        os.getenv("RAG_OCI_ENDPOINT")
        or os.getenv("OCI_RAG_ENDPOINT")
        or os.getenv("OCI_GENAI_ENDPOINT")
    )
    if not endpoint:
        raise RuntimeError("Missing RAG_OCI_ENDPOINT / OCI_GENAI_ENDPOINT for RAG LLM.")

    compartment = (
        os.getenv("RAG_OCI_COMPARTMENT_ID")
        or os.getenv("OCI_RAG_COMPARTMENT_ID")
        or os.getenv("OCI_COMPARTMENT_ID")
    )
    auth_type = os.getenv("RAG_OCI_AUTH_TYPE") or os.getenv("OCI_AUTH_TYPE", "API_KEY")
    auth_profile = (
        os.getenv("RAG_OCI_CONFIG_PROFILE")
        or os.getenv("OCI_RAG_CONFIG_PROFILE")
        or os.getenv("OCI_CONFIG_PROFILE")
        or "DEFAULT"
    )

    temperature = float(
        os.getenv("RAG_OCI_TEMPERATURE", os.getenv("OCI_TEMPERATURE", "1"))
    )
    max_tokens = int(
        os.getenv("RAG_OCI_MAX_TOKENS", os.getenv("OCI_MAX_TOKENS", "2048"))
    )

    model_kwargs = {"temperature": temperature, "max_tokens": max_tokens}

    print(
        "[RAG] Using OCI chat model:",
        f"provider={provider}",
        f"model_id={model_id}",
        f"endpoint={endpoint}",
        f"compartment={compartment}",
    )

    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=compartment,
        provider=provider,
        auth_type=auth_type,
        auth_profile=auth_profile,
        model_kwargs=model_kwargs,
    )

def init_once() -> Dict[str, Any]:
    global _CONN, _EMB, _VS, _RETRIEVER, _QA, _CONFIG
    if _QA is not None:
        return {"ok": True, "note": "already initialized"}

    user = os.getenv("ORA_USER")
    password = os.getenv("ORA_PASSWORD")
    dsn = os.getenv("ORA_DSN")
    table = os.getenv("ORA_TABLE", _DEFAULT_TABLE)
    model_name = os.getenv("ORA_MODEL_NAME", _DEFAULT_MODEL)
    rag_model_id = (
        os.getenv("RAG_OCI_MODEL_ID")
        or os.getenv("OCI_RAG_MODEL_ID")
        or os.getenv("OCI_GENAI_MODEL_ID")
    )
    rag_provider = (
        os.getenv("RAG_OCI_PROVIDER")
        or os.getenv("OCI_RAG_PROVIDER")
        or os.getenv("OCI_GENAI_PROVIDER")
        or "cohere"
    )

    _CONFIG = {
        "user": user,
        "dsn": dsn,
        "table": table,
        "model_name": model_name,
        "rag_model": rag_model_id,
        "rag_provider": rag_provider,
    }

    if not user or not password or not dsn:
        raise RuntimeError("Missing ORA_USER/ORA_PASSWORD/ORA_DSN.")

    try:
        _CONN = oracledb.connect(user=user, password=password, dsn=dsn)
    except Exception as exc:
        raise RuntimeError(f"Oracle connection failed: {exc}") from exc
    _EMB = OracleEmbeddings(conn=_CONN, params={"provider": "database", "model": model_name})
    _VS = OracleVS(client=_CONN, table_name=table, embedding_function=_EMB,
                   distance_strategy=DistanceStrategy.COSINE)
    _RETRIEVER = _VS.as_retriever(search_kwargs={"k": 3})

    llm = _make_rag_llm()
    print("[RAG] LLM created:", type(llm).__name__, "provider=", _CONFIG.get("rag_provider"))
    _QA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                      retriever=_RETRIEVER, return_source_documents=True)

    try:
        _ = _QA.invoke({"query": "health check"})
    except Exception as e:
        print(f"[RAG] Warning: health probe failed: {e}")
        return {"ok": True, "warning": f"health probe failed: {e}"}
    return {"ok": True}

def health() -> Dict[str, Any]:
    return {
        "oracle_connected": bool(_CONN),
        "table": _CONFIG.get("table", _DEFAULT_TABLE),
        "embed_model": _CONFIG.get("model_name", _DEFAULT_MODEL),
        "initialized": bool(_QA),
        "llm_provider": _CONFIG.get("rag_provider"),
        "oci_model_id": _CONFIG.get("rag_model"),
    }

def query(question: str, k: int = 3) -> Dict[str, Any]:
    if not question or not question.strip():
        return {"error": "empty question"}
    if _QA is None:
        init_once()
    if _RETRIEVER and k:
        _RETRIEVER.search_kwargs["k"] = int(k)

    result = _QA.invoke({"query": question})
    answer = result.get("result", "") or ""
    sources = result.get("source_documents", []) or []
    src: List[Dict[str, Any]] = []
    for d in sources:
        meta = d.metadata or {}
        src.append({
            "title": meta.get("title") or os.path.basename(meta.get("source","") or "") or "untitled",
            "source": meta.get("source"),
            "score": meta.get("score"),
            "chunk_preview": (d.page_content or "")[:300]
        })
    return {"answer": answer, "sources": src}
