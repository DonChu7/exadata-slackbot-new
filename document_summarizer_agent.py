# document_summarizer_agent.py
# -------------------------------------------------------------
# PDF/text summarizer with:
# - Proxy-immune local Ollama client (ignores corporate proxies)
# - Warm-up hook to pre-load the model
# - Single-pass fast path for short texts
# - Manual map→reduce for long texts
# - Timing breadcrumbs in `notes`
# - Verbose debug logs when enabled
#
# Deps:
#   pip install pypdf langchain langchain-community requests
# (langchain is used only for the text splitter; no langchain_ollama dependency)
#
# Recommended env:
#   export OLLAMA_BASE_URL=http://127.0.0.1:11434
#   export OLLAMA_MODEL=mistral
#   export NO_PROXY=127.0.0.1,localhost,::1   # optional
#   export DEBUG_SUMMARIZER=1                 # to see detailed logs
#   export SUMMARIZER_LOG_LEVEL=DEBUG
#
# Public API:
#   summarize_pdf_bytes(pdf_bytes, title="...", style_hint=None) -> SummaryResult
#   summarize_plain_text(text, title="...", style_hint=None) -> SummaryResult
#   warm_model()  # optional: pre-load the model
#   probe_llm()   # tiny test call

from __future__ import annotations

import io
import os
import time
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =========================
# Logging
# =========================
_LOG_LEVEL = os.getenv("SUMMARIZER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.INFO))
logger = logging.getLogger("document_summarizer")

DEBUG_SUMMARIZER = os.getenv("DEBUG_SUMMARIZER", "0").lower() in ("1", "true", "yes")


def _dbg(msg: str, *args):
    if DEBUG_SUMMARIZER or logger.isEnabledFor(logging.DEBUG):
        logger.debug(msg, *args)


# =========================
# Config
# =========================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_STYLE_HINT = (
    "Executive-friendly, precise, technical. Avoid fluff. "
    "Include key steps, parameters, decisions, risks, and next actions."
)

# Safety bounds for prompts fed to the model
MAX_MAP_INPUT_CHARS = 4000
MAX_REDUCE_INPUT_CHARS = 8000

# Chunking defaults (tuned for Mistral context headroom)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Include notes directly inside summary text when a failure occurs
INCLUDE_NOTES_IN_SUMMARY_ON_FAIL = os.getenv("INCLUDE_NOTES_IN_SUMMARY_ON_FAIL", "1").lower() in ("1","true","yes")


# =========================
# Data types
# =========================
@dataclass
class SummaryResult:
    title: str
    page_count: int
    char_count: int
    summary: str
    notes: Optional[str] = None  # e.g., OCR used, timings, or error details


# =========================
# Proxy-immune Ollama HTTP client
# =========================
class OllamaHTTP:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 180,
        num_ctx: int = 8192,
        temperature: float = 0.2,
        num_predict: int = 512,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.num_predict = num_predict

        s = requests.Session()
        s.trust_env = False                       # ignore HTTP(S)_PROXY / NO_PROXY
        s.proxies.update({"http": None, "https": None})  # hard bypass
        self.sess = s

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        t0 = time.perf_counter()
        r = self.sess.post(url, json=payload, timeout=self.timeout)
        dt = time.perf_counter() - t0
        _dbg("OllamaHTTP.generate: status=%s time=%.2fs", getattr(r, "status_code", "?"), dt)
        r.raise_for_status()
        data = r.json()
        # With stream=False, Ollama returns the full text in "response"
        return data.get("response", "") or data.get("text", "")


@lru_cache(maxsize=1)
def get_local_ollama() -> OllamaHTTP:
    _dbg("Init OllamaHTTP model=%s base=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    return OllamaHTTP(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        timeout=180,
        num_ctx=8192,
        temperature=0.2,
        num_predict=512,
    )


def _llm_call(prompt: str, retries: int = 2, backoff: float = 0.8) -> str:
    last_err = None
    for i in range(retries + 1):
        try:
            if DEBUG_SUMMARIZER:
                _dbg("[LLM] try=%d prompt_len=%d", i + 1, len(prompt))
                _dbg("[LLM] prompt_head:\n%s", prompt[:600])
            out = get_local_ollama().generate(prompt)
            if DEBUG_SUMMARIZER:
                _dbg("[LLM] ok out_len=%d", len(out))
                _dbg("[LLM] out_head:\n%s", out[:600])
            return out
        except Exception as e:
            last_err = e
            logger.warning("[LLM] fail try=%d/%d: %s: %s", i + 1, retries + 1, type(e).__name__, e)
            if i < retries:
                time.sleep(backoff * (2 ** i))
    raise last_err


# =========================
# Extraction
# =========================
def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, int]:
    """
    Returns (text, page_count) using pypdf.
    For image-only PDFs, text will likely be short/empty.
    """
    t0 = time.perf_counter()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = len(reader.pages)
    text_parts: List[str] = []
    for idx, p in enumerate(reader.pages, start=1):
        txt = p.extract_text() or ""
        _dbg("Extracted page %d: %d chars", idx, len(txt))
        text_parts.append(txt)
    text = "\n".join(text_parts).strip()
    dt = time.perf_counter() - t0
    logger.info("PDF extracted: pages=%d, total_chars=%d, time=%.2fs", pages, len(text), dt)
    if DEBUG_SUMMARIZER:
        _dbg("Extracted text head:\n%s", text[:800])
    return text, pages


# Optional OCR fallback (commented)
# from pdf2image import convert_from_bytes
# import pytesseract
# def _extract_text_via_ocr(pdf_bytes: bytes) -> str:
#     images = convert_from_bytes(pdf_bytes, dpi=300)
#     pieces = []
#     for img in images:
#         pieces.append(pytesseract.image_to_string(img))
#     return "\n".join(pieces).strip()


# =========================
# Chunking
# =========================
def _chunk_text(text: str, max_chunk_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    t0 = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    dt = time.perf_counter() - t0
    logger.info("Chunking: %d chunks (size=%d, overlap=%d), time=%.2fs", len(chunks), max_chunk_chars, overlap, dt)
    if DEBUG_SUMMARIZER:
        for i, c in enumerate(chunks[:5], start=1):
            _dbg("Chunk %d len=%d head:\n%s", i, len(c), c[:300])
        if len(chunks) > 5:
            _dbg("... (%d more chunks not shown)", len(chunks) - 5)
    return chunks


# =========================
# Prompts
# =========================
SINGLE_PROMPT_TMPL = """Summarize the passage *using only the information provided*.
- Be concise.
- If something is not stated, say "not specified".
- Prefer bullets when appropriate.

Passage:
{chunk}
"""
MAP_PROMPT_TMPL = """You are a precise technical summarizer.
Summarize the following excerpt into 5-8 concise bullet points.
Focus on: purpose, key steps, parameters, commands/paths, decisions, and risks.
Avoid fluff; keep bullets short.

Excerpt:
{chunk}
"""


REDUCE_PROMPT_TMPL = """You are producing an executive summary for "{title}".
Synthesize the bullet lists below into:
1) A short overview paragraph (2-4 sentences).
2) 6-10 bullets of key points (facts, steps, decisions).
3) A brief "Risks / Gaps" section (if applicable).
4) "Next Steps" with owners/dates if present.

Bullet inputs:
{bullets}
"""
# =========================
# Summarization
# =========================
def _single_pass_summary(text: str, title: str, style_hint: Optional[str]) -> str:
    prompt = SINGLE_PROMPT_TMPL.format(chunk=text[:MAX_MAP_INPUT_CHARS])
    if style_hint:
        prompt = f"(Style: {style_hint})\n\n{prompt}"
    return _llm_call(prompt).strip()


def _manual_map_reduce_summary(chunks: List[str], title: str, style_hint: Optional[str]) -> str:
    # MAP: per-chunk bullets
    bullets_all = []
    for idx, c in enumerate(chunks, 1):
        snippet = c[:MAX_MAP_INPUT_CHARS]
        prompt = MAP_PROMPT_TMPL.format(chunk=snippet)
        if style_hint:
            prompt = f"(Style: {style_hint})\n\n{prompt}"
        try:
            out = _llm_call(prompt)
        except Exception as e:
            prompt_head = prompt[:400].replace("```", "`ˋ`")  # keep Slack formatting safe
            raise RuntimeError(
                f"MAP step failed at chunk {idx}: {type(e).__name__}: {e}\n"
                f"prompt_head:\n{prompt_head}"
            ) from e
        bullets = (out or "").strip()
        bullets_all.append(f"--- chunk {idx} ---\n{bullets}")

    # REDUCE: synthesize bullets
    combined = "\n".join(bullets_all)
    combined_trimmed = combined[:MAX_REDUCE_INPUT_CHARS]  # keep within context
    reduce_prompt = REDUCE_PROMPT_TMPL.format(title=title, bullets=combined_trimmed)
    if style_hint:
        reduce_prompt = f"(Style: {style_hint})\n\n{reduce_prompt}"
    try:
        final = _llm_call(reduce_prompt)
    except Exception as e:
        prompt_head = reduce_prompt[:400].replace("```", "`ˋ`")
        raise RuntimeError(
            f"REDUCE step failed: {type(e).__name__}: {e}\n"
            f"prompt_head:\n{prompt_head}"
        ) from e

    return (final or "").strip()


# =========================
# Public API
# =========================
def summarize_pdf_bytes(
    pdf_bytes: bytes,
    title: str = "document.pdf",
    style_hint: Optional[str] = DEFAULT_STYLE_HINT,
) -> SummaryResult:
    """
    Extract -> (single-pass or map-reduce) summary.
    Adds timing breadcrumbs to notes.
    """
    t0 = time.perf_counter()
    try:
        text, pages = _extract_text_from_pdf_bytes(pdf_bytes)
    except PdfReadError as e:
        logger.exception("PDF parse error")
        return SummaryResult(title=title, page_count=0, char_count=0, summary="[PDF parse error]", notes=str(e))
    except Exception as e:
        logger.exception("Unexpected read error")
        return SummaryResult(title=title, page_count=0, char_count=0, summary="[Unexpected read error]", notes=f"{type(e).__name__}: {e}")
    t1 = time.perf_counter()

    notes = None
    if len(text) < 200:
        notes = "Extracted text is very short; PDF may be image-only. Enable OCR if needed."

    if not text:
        return SummaryResult(
            title=title,
            page_count=pages,
            char_count=0,
            summary="[No extractable text found.]",
            notes=notes + (f" | timing: extract={t1-t0:.2f}s" if notes else f"timing: extract={t1-t0:.2f}s"),
        )

    try:
        if len(text) <= 2000:  # fast path
            summary = _single_pass_summary(text, title, style_hint)
            t2 = time.perf_counter()
            timing = f"timing: extract={t1-t0:.2f}s, single={t2-t1:.2f}s"
            notes = (notes + " | " + timing) if notes else timing
        else:
            chunks = _chunk_text(text)
            t2 = time.perf_counter()
            summary = _manual_map_reduce_summary(chunks, title=title, style_hint=style_hint)
            t3 = time.perf_counter()
            timing = f"timing: extract={t1-t0:.2f}s, chunk={t2-t1:.2f}s, mapreduce={t3-t2:.2f}s"
            notes = (notes + " | " + timing) if notes else timing
    except Exception as e:
        err = f"{e}"
        summary_text = "[Summarization failed]"
        if INCLUDE_NOTES_IN_SUMMARY_ON_FAIL:
            summary_text = f"[Summarization failed]\n{err[:1800]}"
        return SummaryResult(
            title=title,
            page_count=pages,
            char_count=len(text),
            summary=summary_text,
            notes=(notes + " | " + err) if notes else err,
        )

    return SummaryResult(
        title=title,
        page_count=pages,
        char_count=len(text),
        summary=summary if summary else "[No summary produced]",
        notes=notes,
    )


def summarize_plain_text(
    text: str,
    title: str = "document",
    style_hint: Optional[str] = DEFAULT_STYLE_HINT,
) -> SummaryResult:
    """
    Summarize plain text (no PDF extraction).
    Adds timing breadcrumbs to notes.
    """
    if not text or not text.strip():
        return SummaryResult(title=title, page_count=0, char_count=0, summary="[Empty text]")

    t0 = time.perf_counter()
    try:
        if len(text) <= 800:  # fast path
            summary = _single_pass_summary(text, title, style_hint)
            t1 = time.perf_counter()
            notes = f"timing: single={t1-t0:.2f}s"
        else:
            chunks = _chunk_text(text)
            t1 = time.perf_counter()
            summary = _manual_map_reduce_summary(chunks, title=title, style_hint=style_hint)
            t2 = time.perf_counter()
            notes = f"timing: chunk={t1-t0:.2f}s, mapreduce={t2-t1:.2f}s"
    except Exception as e:
        err = f"{e}"
        summary_text = "[Summarization failed]"
        if INCLUDE_NOTES_IN_SUMMARY_ON_FAIL:
            summary_text = f"[Summarization failed]\n{err[:1800]}"
        return SummaryResult(
            title=title,
            page_count=0,
            char_count=len(text),
            summary=summary_text,
            notes=err,
        )

    return SummaryResult(
        title=title,
        page_count=0,
        char_count=len(text),
        summary=summary if summary else "[No summary produced]",
        notes=notes,
    )


# =========================
# Warm-up & Diag
# =========================
def warm_model():
    """Force-load the model to avoid cold-start latency."""
    try:
        t0 = time.perf_counter()
        _ = get_local_ollama().generate("Reply with OK.")
        dt = time.perf_counter() - t0
        logger.info("warm_model: completed in %.2fs", dt)
    except Exception as e:
        logger.warning("warm_model error: %s: %s", type(e).__name__, e)


def probe_llm() -> str:
    """Quick connectivity probe for Ollama."""
    out = get_local_ollama().generate("Reply with OK.")
    return out.strip()[:100]
