#!/usr/bin/env python3
from __future__ import annotations
import os, json, re, traceback
import requests
from typing import Any, Dict, Optional
from mcp.server.fastmcp import FastMCP
from urllib.parse import quote
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import quote, urlparse, urljoin
import base64

# --- Config via env ---
BASE_URL = os.getenv("GENAI4TEST_BASE_URL",
    "https://phoenix228912.dev3sub3phx.databasede3phx.oraclevcn.com:8000")
VERIFY_SSL = os.getenv("GENAI4TEST_VERIFY_SSL", "false").lower() == "true" \
             or os.getenv("GENAI4TEST_CA_BUNDLE")  # path to CA file if provided
CA_BUNDLE = os.getenv("GENAI4TEST_CA_BUNDLE")  # e.g. /etc/ssl/certs/your-ca.pem
TIMEOUT_S = float(os.getenv("GENAI4TEST_TIMEOUT_S", "600"))
DEFAULT_EMAIL = os.getenv("GENAI4TEST_EMAIL", "dongyang.zhu@oracle.com")
DEFAULT_AGENT = os.getenv("GENAI4TEST_AGENT", "bug_agent_dynamic")

app = FastMCP("genai4test-mcp")


@app.tool()
def health() -> Dict[str, Any]:
    """Quick reachability probe for the genai4test service."""
    try:
        verify_arg = CA_BUNDLE if CA_BUNDLE else bool(VERIFY_SSL)
        r = requests.get(f"{BASE_URL}/docs", timeout=30, verify=verify_arg)
        return {"ok": r.ok, "status_code": r.status_code}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.tool()
def run_bug_test(bug_no: str, email: str | None = None, agent: str | None = None) -> dict:
    """
    Call genai4test to generate a test from a bug.
    Returns: {ok, summary, sql, file_url, request_url, status, error?}
    """
    try:
        bug_no = (bug_no or "").strip()
        if not bug_no:
            return {"ok": False, "error": "bug_no is required"}

        email = (email or DEFAULT_EMAIL).strip()
        agent = (agent or DEFAULT_AGENT).strip()

        # URL-encode path segments safely
        email_q = quote(email, safe="")
        bug_q   = quote(bug_no, safe="")
        agent_q = quote(agent, safe="")

        url = f"{BASE_URL}/genai4test/run-bug/{email_q}/{bug_q}/{agent_q}"
        kwargs = {"timeout": TIMEOUT_S}
        if CA_BUNDLE:
            kwargs["verify"] = CA_BUNDLE
        else:
            kwargs["verify"] = bool(VERIFY_SSL)

        url = f"{BASE_URL}/genai4test/run-bug/{email_q}/{bug_q}/{agent_q}"

        sess = requests.Session()
        sess.trust_env = False                          # ignore proxy env
        sess.proxies = {"http": None, "https": None}    # force no proxy

        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10, pool_block=True)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)

        verify_arg = CA_BUNDLE if CA_BUNDLE else bool(VERIFY_SSL)
        timeout_arg = (10, TIMEOUT_S)   # (connect, read). e.g. (10, 600+)

        resp = sess.get(url, timeout=timeout_arg, verify=verify_arg)
        if resp.status_code != 200:
            # include a short body preview to help debug
            body = resp.text[:500]
            return {"ok": False, "status": resp.status_code, "request_url": url,
                    "error": f"HTTP {resp.status_code}", "body": body}

        data = resp.json()

        script = data.get("sql") or data.get("code") or data.get("script")
        file_url = data.get("file_url")
        abs_url = None
        if isinstance(file_url, str) and file_url:
            # Ensure file_url is correctly rooted under /genai4test/
            prefix = "/genai4test/"
            if not file_url.startswith(prefix):
                # Strip leading slashes and enforce /genai4test/ prefix
                file_url = f"genai4test/{file_url.lstrip('/')}"
            abs_url = urljoin(BASE_URL.rstrip("/") + "/", file_url)

        return {
            "ok": True,
            "request_url": url,
            "summary": data.get("summary"),
            "script": script,
            "sql": script,                    # <-- key your Slack code already expects
            "file_url": file_url,
            "absolute_file_url": abs_url,     # <-- optional convenience
        }
    except requests.exceptions.ReadTimeout as e:
        return {"ok": False, "error": f"ReadTimeout: {e}", "request_url": url}
    except requests.exceptions.SSLError as e:
        return {"ok": False, "error": f"SSLError: {e}", "request_url": url}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "request_url": url}

if __name__ == "__main__":
    app.run()