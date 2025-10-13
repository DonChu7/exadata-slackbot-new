#!/usr/bin/env python3
"""
MCP server for OEDA generation.

Tools
- generate_minconfig(request: str)
    -> {"minconfig_json": {...}}

- generate_oedaxml(request: str, genoedaxml_path?: str=None, return_xml?: bool=False)
    -> {
         "minconfig_json": {...},
         "es_xml_path": "<path or null>",
         "es_xml_b64": "<base64 or null>"  # only if return_xml=True and file exists
       }

Safety
- Optional allowlist for genoedaxml path via env OEDA_GENOEDAXML_ALLOWLIST (colon/semicolon separated prefixes)
"""

from __future__ import annotations
import os, json, base64
from typing import Dict, Any, Optional, List
import re
_RACK_PAT1 = re.compile(r"rack\s*description:\s*(?P<desc>.+)", re.I)
_RACK_PAT2 = re.compile(r"deduced\s+rackDescription\s+to:\s*(?P<desc>.+)", re.I)
_XVER_ANY = re.compile(r"\bX\s*(\d+)\b", re.I)

from mcp.server.fastmcp import FastMCP

# your existing agent
from oeda_agent import call_llm_to_generate_json, run_genoedaxml

app = FastMCP("oeda-mcp")


import re

LIVE_MIG_PAT = re.compile(r"\blive[-\s]*migration\b", re.I)

def _is_live_migration_req(text: str) -> bool:
    return bool(LIVE_MIG_PAT.search(text or ""))

def _apply_live_migration_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce required knobs for live-migration-capable env."""
    cfg = dict(cfg or {})
    cfg["virtualCluster"] = True
    cfg["exascale"] = True
    cfg.setdefault("clusterCount", 1)
    # Prefer uniform celldisk unless user already specified per-cluster
    if "clusterGuestStorage" not in cfg and "clusterStorage" not in cfg:
        if int(cfg.get("clusterCount", 1)) > 1:
            cfg["clusterGuestStorage"] = ",".join(["celldisk"] * int(cfg["clusterCount"]))
        else:
            cfg["guestStorage"] = "celldisk"
    return cfg


def _parse_rack_description(log_text: str) -> Optional[str]:
    """
    Try common variants:
      - 'Rack description: X11M-2 ...'
      - 'Setting deduced rackDescription to: X8M-2 ...'
      - 'Rack Description = X10 ...'
      - fallback: first line that looks like it starts with X<number>
    """
    if not log_text:
        return None
    patterns = [
        re.compile(r"^\s*rack\s*description\s*[:=\-]\s*(.+)$", re.I),
        re.compile(r"^\s*setting\s+deduced\s+rack\s*description\s+to\s*[:=\-]\s*(.+)$", re.I),
        re.compile(r"^\s*deduced\s+rack\s*description\s*[:=\-]\s*(.+)$", re.I),
        re.compile(r"^\s*rack\s*model\s*[:=\-]\s*(.+)$", re.I),
    ]
    for line in log_text.splitlines():
        for pat in patterns:
            m = pat.search(line)
            if m:
                return m.group(1).strip()
    # Fallback: look for a standalone 'X<number>' line segment and return the whole line
    for line in log_text.splitlines():
        if _XVER_ANY.search(line):
            return line.strip()
    return None
    

def _extract_x_version(text: str) -> Optional[int]:
    if not text:
        return None
    m = _XVER_ANY.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def _rack_version_ok(rack_desc: str) -> bool:
    m = re.match(r"\s*X\s*(\d+)", (rack_desc or "").upper())
    return bool(m and int(m.group(1)) >= 10)

def validate_hw_support_from_log(log_text: str) -> tuple[str, Optional[str], str]:
    """
    Returns (status, rack_desc, reason)
      status in {"ok", "fail", "unknown"}
    """
    desc = _parse_rack_description(log_text)
    ver  = _extract_x_version(desc) or _extract_x_version(log_text)

    if ver is None:
        return "unknown", desc, "rackDescription/version not found in genoedaxml output"
    if ver >= 10:
        return "ok", desc, "OK"
    return "fail", desc, "The hardware version doesn't support live migration. Pick hardware that's X10 or above."

def _is_allowed_path(path: str) -> bool:
    """
    If OEDA_GENOEDAXML_ALLOWLIST is set, path must start with one of its prefixes.
    e.g. OEDA_GENOEDAXML_ALLOWLIST=/net/dbdevfssmnt-shared01.dev3fss1phx.databasede3phx.oraclevcn.com/exadata_dev_image_oeda
    """
    if not path:
        return False
    allow = os.getenv("OEDA_GENOEDAXML_ALLOWLIST", "")
    if not allow.strip():
        return True  # permissive if unset
    seps = ";" if ";" in allow else ":"
    prefixes = [p.strip() for p in allow.split(seps) if p.strip()]
    path_abs = os.path.abspath(path)
    for pref in prefixes:
        if path_abs.startswith(os.path.abspath(pref)):
            return True
    return False

@app.tool()
def generate_minconfig(request: str) -> Dict[str, Any]:
    """
    Return only minconfig.json from a natural-language request.
    """
    if not isinstance(request, str) or not request.strip():
        return {"error": "Missing or empty 'request' string."}
    cfg = call_llm_to_generate_json(request.strip())
    live_mig = _is_live_migration_req(request)
    if live_mig:
        cfg = _apply_live_migration_defaults(cfg)
    return {"minconfig_json": cfg, "live_migration": live_mig}

@app.tool()
def generate_oedaxml(
    request: str,
    genoedaxml_path: Optional[str] = None,
    return_xml: bool = False,
    force_mock: bool = False,
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Build minconfig via LLM/mock and optionally run genoedaxml to produce es.xml.
    If debug=True, include run logs and discovery notes in the response.
    """
    if not isinstance(request, str) or not request.strip():
        return {"error": "Missing or empty 'request' string."}

    # choose generator
    try:
        if force_mock:
            from oeda_agent import mock_llm_response
            cfg = mock_llm_response(request.strip())
            gen_used = "mock"
        else:
            cfg = call_llm_to_generate_json(request.strip())
            gen_used = "llm"
    except Exception as e:
        return {"error": f"minconfig generation failed: {e}"}

    # Enforce live-migration defaults if requested
    live_mig = _is_live_migration_req(request)
    if live_mig:
        cfg = _apply_live_migration_defaults(cfg)

    out: Dict[str, Any] = {
        "minconfig_json": cfg,
        "generator": gen_used,
        "es_xml_path": None,
        "live_migration": live_mig,
        # the three new fields the app will read:
        "live_mig_check": "n/a",
        "live_mig_reason": None,
        "rack_desc": None,
    }

    # If no genoedaxml path, return JSON only
    if not genoedaxml_path:
        if debug:
            out["note"] = "genoedaxml_path not provided; returning JSON only."
        return out

    # allowlist
    if not _is_allowed_path(genoedaxml_path):
        out["error"] = "genoedaxml_path not allowed by OEDA_GENOEDAXML_ALLOWLIST"
        return out

    # Run genoedaxml and capture stdout/stderr
    import io, sys, contextlib
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            es_xml_path = run_genoedaxml(cfg, genoedaxml_path)
        out["es_xml_path"] = es_xml_path
    except Exception as e:
        out["error"] = f"genoedaxml failed: {e}"
    finally:
        stdout_text = buf_out.getvalue()
        stderr_text = buf_err.getvalue()
        if debug:
            out["genoedaxml_stdout"] = stdout_text
            out["genoedaxml_stderr"] = stderr_text

    # Parse rackDescription only if relevant; gate HW check only for live-migration requests
    log_text = (stdout_text or "") + ("\n" + stderr_text if stderr_text else "")
    rack_desc = _parse_rack_description(log_text)
    out["rack_desc"] = rack_desc

    if live_mig:
        status, rack_desc, reason = validate_hw_support_from_log(log_text)
        out["rack_desc"]       = rack_desc
        out["live_mig_check"]  = status
        out["live_mig_reason"] = reason
        if status == "fail":   # Only block when we’re sure it’s < X10
            return out
    # else: leave live_mig_check as "n/a" and do not mention anything in Slack
        


    # If status == "unknown", do NOT block; return XML if available and include debug so we can investigate.


    # Optionally attach es.xml content
    if return_xml and out.get("es_xml_path") and os.path.isfile(out["es_xml_path"]):
        try:
            with open(out["es_xml_path"], "rb") as f:
                out["es_xml_b64"] = base64.b64encode(f.read()).decode("ascii")
        except Exception as e:
            out["es_xml_b64_error"] = f"Failed to read es.xml: {e}"

    return out

@app.tool()
def tool_manifest() -> Dict[str, Any]:
    return {
        "service": "oeda-mcp",
        "tools": [
            {
                "name": "generate_oedaxml",
                "description": "Generate OEDA es.xml and/or minconfig.json from natural language.",
                "intents": ["oeda", "oedaxml", "es.xml", "exadata xml", "generate xml"],
                "patterns": [r"\bgenerate\s+oedaxml\b", r"\bconfig+xml\b", r"\boeda\b", r"\bxml\b"]
            }
        ]
    }


if __name__ == "__main__":
    app.run()