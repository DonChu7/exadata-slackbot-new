# llm_provider.py — 0.2 line, OCI (community) or Ollama
import os
from langchain_ollama import OllamaLLM
from langchain_community.chat_models import ChatOCIGenAI

TOOL_FRIENDLY_DEFAULT = "cohere.command-r-plus-latest"

def _resolve_oci_provider(model_id: str | None):
    explicit = os.getenv("OCI_GENAI_PROVIDER")
    if explicit:
        provider = explicit.strip().lower()
    elif model_id:
        low = model_id.lower()
        if low.startswith("cohere"):
            provider = "cohere"
        elif low.startswith("meta"):
            provider = "meta"
        elif low.startswith("xai") or "grok" in low:
            provider = "xai"
        else:
            provider = "cohere"
    else:
        provider = "cohere"

    if provider == "meta":
        fallback = os.getenv("OCI_GENAI_TOOL_MODEL_ID", TOOL_FRIENDLY_DEFAULT)
        print("[llm_provider] 'meta' not tool-friendly; falling back to", fallback)
        return "cohere", fallback
    return provider, model_id

def make_llm() -> object:
    provider = os.getenv("LLM_PROVIDER", "oci").lower()
    if provider == "oci":
        model_id = os.getenv("OCI_GENAI_MODEL_ID", TOOL_FRIENDLY_DEFAULT)
        endpoint = os.getenv(
            "OCI_GENAI_ENDPOINT",
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        )
        compartment = os.getenv("OCI_COMPARTMENT_ID")
        auth_type = os.getenv("OCI_AUTH_TYPE", "API_KEY")
        profile = os.getenv("OCI_CONFIG_PROFILE", "DEFAULT")

        provider_tag, model_id = _resolve_oci_provider(model_id)
        temperature = float(os.getenv("OCI_TEMPERATURE", "1"))
        max_ct = int(os.getenv("OCI_MAX_COMPLETION_TOKENS", os.getenv("OCI_MAX_TOKENS", "1024")))
        model_kwargs = {"temperature": temperature, "max_completion_tokens": max_ct}

        print(f"[llm_provider] Using provider={provider_tag} model={model_id} kwargs={model_kwargs}")
        return ChatOCIGenAI(
            model_id=model_id,
            service_endpoint=endpoint,
            compartment_id=compartment,
            provider=provider_tag,
            model_kwargs=model_kwargs,
            auth_type=auth_type,
            auth_profile=profile,
        )

    # default: local Ollama
    return OllamaLLM(model=os.getenv("OLLAMA_MODEL", "mistral"))