# app.py — slim Slack bot wired to MCP tools

import os as OS
import re
import json
import time
import base64
import tempfile as TF
import threading
import subprocess
import contextvars
from collections import defaultdict
import requests
from typing import Any, List
import traceback
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import html

from dotenv import load_dotenv
# from slack_bolt import App
# from slack_bolt.adapter.socket_mode import SocketModeHandler
import asyncio
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from mcp_client import PersistentMCPClient

import uuid
from metrics_utils import feedback_blocks, record_feedback_click, append_jsonl, utc_iso

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage, AnyMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from llm_provider import make_llm

# ---------------------------------------------------------------------------
# Boot / env
# ---------------------------------------------------------------------------
load_dotenv()

SLACK_BOT_TOKEN = OS.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = OS.getenv("SLACK_APP_TOKEN", "")
FEEDBACK_PATH = OS.getenv("FEEDBACK_PATH", "./metrics/feedback.jsonl")

JENKINS_URL        = OS.getenv("JENKINS_URL", "https://ci-cloud.us.oracle.com/jenkins/escs-test")
JENKINS_USER       = OS.getenv("JENKINS_USER", "jiahao.l.li@oracle.com")
JENKINS_API_TOKEN  = OS.getenv("JENKINS_API_TOKEN", "")
JENKINS_FOLDER     = OS.getenv("JENKINS_FOLDER", "UPGRADE_LOOP_RUN")
JENKINS_JOB        = OS.getenv("JENKINS_JOB", "01_PRE_SETUP_FOR_SM")
JENKINS_VERIFY_SSL = OS.getenv("JENKINS_VERIFY_SSL", "true").lower() == "true"

# MCP servers (commands to launch each tool server)
RUNINTEG_CMD = OS.getenv("RUNINTEG_CMD", "python /scratch/dongyzhu/exadata-slackbot/runintegration_server.py").split()
OEDA_CMD     = OS.getenv("OEDA_CMD",     "python /scratch/dongyzhu/exadata-slackbot/oeda_server.py").split()
RAG_CMD      = OS.getenv("RAG_CMD",      "python /scratch/dongyzhu/exadata-slackbot/exa23ai_rag_server.py").split()
SUM_CMD      = OS.getenv("SUM_CMD",      "python /scratch/dongyzhu/exadata-slackbot/summarizer_server.py").split()
GENAI4TEST_CMD = OS.getenv("GENAI4TEST_CMD","python /scratch/dongyzhu/exadata-slackbot/genai4test_server.py").split()
LABELHEALTH_CMD = OS.getenv("LABELHEALTH_CMD","python /scratch/dongyzhu/exadata-slackbot/label_health_server.py").split()

# Default genoedaxml path (allowlisted in oeda_server)
DEFAULT_GENXML = OS.getenv("GENOEDAXML_PATH",
    "/net/dbdevfssmnt-shared01.dev3fss1phx.databasede3phx.oraclevcn.com/exadata_dev_image_oeda/genoeda/genoedaxml"
)

# ---------------------------------------------------------------------------
# feedbacks 
# ---------------------------------------------------------------------------
async def post_with_feedback(app, channel_id: str, thread_ts: str | None, text: str, *,
                       context: dict | None = None, user_id: str | None = None, client: AsyncWebClient | None = None,) -> str:
    """
    Posts a message with thumbs up/down buttons. Returns the message_ts.
    'context' is saved alongside feedback (tool name, question, etc).
    """
    # resolve user label if possible
    user_label = None
    if client and user_id:
        user_label = await _get_user_label(client, user_id)
    if not user_label:
        user_label = "unknown"

    uid = str(uuid.uuid4())
    record = {
        "user": user_label,
        "uuid": uid,
        "ts": utc_iso(),
        "context": context or {},
        "original_text": text,
    }
    # Persist the full payload once 
    append_jsonl(FEEDBACK_PATH, record)

    tiny_payload = json.dumps({"uuid": uid})

    res = await app.client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text=text,
        blocks=feedback_blocks(text, voted=None, payload_json=tiny_payload),
    )
    return res["ts"]

# ---------------------------------------------------------------------------
# Helpers (Jenkins + file transfer)
# ---------------------------------------------------------------------------

def _jenkins_session():
    if not (JENKINS_URL and JENKINS_USER and JENKINS_API_TOKEN):
        raise RuntimeError("Missing JENKINS_URL/JENKINS_USER/JENKINS_API_TOKEN")
    s = requests.Session()
    s.trust_env = False
    s.auth = (JENKINS_USER, JENKINS_API_TOKEN)
    s.verify = JENKINS_VERIFY_SSL
    try:
        r = s.get(f"{JENKINS_URL}/crumbIssuer/api/json", timeout=10)
        if r.ok:
            j = r.json()
            s.headers[j.get("crumbRequestField", "Jenkins-Crumb")] = j.get("crumb")
    except Exception:
        pass
    return s

async def trigger_upgrade_loop_run(params=None):
    s = _jenkins_session()
    base = f"{JENKINS_URL}/job/{JENKINS_FOLDER}/job/{JENKINS_JOB}"
    endpoint = f"{base}/buildWithParameters" if params else f"{base}/build"
    resp = s.post(endpoint, data=(params or {}), timeout=20)
    if resp.status_code not in (200, 201, 202, 302):
        raise RuntimeError(f"Jenkins returned {resp.status_code}: {resp.text[:300]}")
    return {"queued": True, "queue_url": resp.headers.get("Location"), "job_url": base}

def _fmt_duration(ms: int) -> str:
    s = int(ms) // 1000 if ms else 0
    h, m, sec = s // 3600, (s % 3600)//60, s % 60
    return f"{h}h {m}m {sec}s" if h else (f"{m}m {sec}s" if m else f"{sec}s")

def _resolve_build_url_from_queue(queue_url: str, session) -> str:
    if not queue_url: return ""
    url = queue_url.rstrip("/") + "/api/json"
    for _ in range(300):
        try:
            r = session.get(url, timeout=15)
            if r.ok:
                data = r.json()
                if data.get("cancelled"):
                    return ""
                exe = data.get("executable") or {}
                burl = (exe.get("url") or "").rstrip("/")
                if burl:
                    return burl
        except Exception:
            pass
        time.sleep(5)
    return ""

def _monitor_and_notify(queue_url: str, base_job_url: str, channel_id: str, thread_ts: str):
    try:
        s = _jenkins_session()
    except Exception as e:
        try:
            app.client.chat_postMessage(channel=channel_id, thread_ts=thread_ts,
                                        text=f"Unable to monitor Jenkins build: {e}")
        except Exception:
            pass
        return

    build_url = _resolve_build_url_from_queue(queue_url, s)
    if not build_url and base_job_url:
        try:
            r = s.get(f"{base_job_url}/lastBuild/api/json", timeout=15)
            if r.ok:
                u = (r.json().get("url") or "").rstrip("/")
                build_url = u or build_url
        except Exception:
            pass
    if not build_url:
        try:
            app.client.chat_postMessage(channel=channel_id, thread_ts=thread_ts,
                                        text="Unable to resolve build URL for monitoring.")
        except Exception:
            pass
        return

    # poll until complete
    api = build_url.rstrip("/") + "/api/json"
    last_result = None
    for _ in range(360):
        try:
            r = s.get(api, timeout=20)
            if r.ok:
                j = r.json()
                building = j.get("building", False)
                result = j.get("result")
                if not building and result:
                    dur = _fmt_duration(j.get("duration", 0))
                    final = f"Job finished: {result}\nBuild: {build_url}\nDuration: {dur}"
                    app.client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=final)
                    return
                last_result = result
        except Exception:
            pass
        time.sleep(10)
    app.client.chat_postMessage(
        channel=channel_id, thread_ts=thread_ts,
        text=f"Monitoring timed out. Latest known status: {last_result or 'BUILDING'}\nBuild: {build_url}"
    )

def _parse_env_params_from_text(text: str):
    t = (text or "").lower()
    if re.search(r"\br1x\b", t): return {"ENV": "r1x"}
    if re.search(r"\br1\b",  t): return {"ENV": "r1"}
    return None

def scp_file_with_key(file_path: str, destination: str, ssh_key_path: str) -> bool:
    try:
        cmd = ["scp", "-i", ssh_key_path, file_path, destination]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] SCP failed:", e.stderr.decode())
        return False

def _pick_genai4test_filename(res: dict) -> str:
    """
    Derive a reasonable filename from genai4test result.
    Prefer URL basename; otherwise infer from script; default to .txt.
    """
    # Prefer URL basename if available
    url = (res.get("absolute_file_url") or res.get("file_url") or "").strip()
    if url:
        name = OS.path.basename(urlparse(url).path)
        if name:
            return name

    # Infer from inline script (sql/sh) if present
    script = (res.get("sql") or "").lstrip()
    if script.startswith("#"):  # looks like shell
        return "genai4test_script.sh"
    # crude SQL heuristic
    if script.startswith("Rem"):
        return "genai4test_script.sql"
    return "genai4test_output.txt"

def _download_bytes(url: str, verify) -> bytes:
    s = requests.Session()
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    r = s.get(url, timeout=(10, 600), verify=verify)
    r.raise_for_status()
    return r.content

async def _get_first_name(client, user_id: str) -> str:
    """
    Try multiple profile fields (first_name/given_name/display/real_name),
    fall back to the local-part of email, then to 'there'.
    """
    try:
        ui = await client.users_info(user=user_id)   # requires 'users:read' scope
        user = (ui or {}).get("user", {}) or {}
        prof = user.get("profile", {}) or {}

        candidates = [
            prof.get("first_name"),                 # legacy; often empty
            prof.get("given_name"),                 # some workspaces use this
            prof.get("display_name_normalized"),
            prof.get("display_name"),
            prof.get("real_name_normalized"),
            prof.get("real_name"),
            user.get("name"),                       # legacy handle
        ]
        first = next((c for c in candidates if c), None)
        if first and " " in first:
            first = first.split()[0]

        if not first:
            email = prof.get("email")
            if email and "@" in email:
                first = email.split("@", 1)[0]

        return first or "there"
    except Exception as e:
        print("[users_info] error:", type(e).__name__, e)
        return "there"
    

async def _get_user_label(client: AsyncWebClient, user_id: str) -> str:
    """
    Return a readable label for a Slack user:
    'Full Name (@username)' if available,
    else '@username',
    else '<@U12345>' as a last resort.
    """
    if not user_id:
        return "unknown"
    try:
        ui = await client.users_info(user=user_id)  # needs users:read
        user = (ui or {}).get("user", {}) or {}
        prof = user.get("profile", {}) or {}
        real = prof.get("real_name_normalized") or prof.get("real_name") or ""
        handle = user.get("name") or prof.get("display_name") or ""
        if real and handle:
            return f"{real} (@{handle})"
        if real:
            return real
        if handle:
            return f"@{handle}"
        return f"<@{user_id}>"
    except SlackApiError as e:
        # Scope missing or other error — don’t break metrics
        print("[users_info] scope/error:", getattr(e, "response", {}).data if getattr(e, "response", None) else e)
        return f"<@{user_id}>"
    except Exception as e:
        print("[users_info] error:", type(e).__name__, e)
        return f"<@{user_id}>"

def feedback_thanks_blocks(text: str, sentiment: str) -> list[dict]:
    """
    A grey 'thanks' note below the original text.
    sentiment: "up" or "down"
    """
    icon = "👍" if sentiment == "up" else "👎"
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": text}},
        {"type": "context", "elements": [
            {"type": "mrkdwn", "text": f"_Thanks for your feedback {icon}_"}
        ]},
    ]

def _format_lrg_table(lrgs: list[str], max_rows: int = 50) -> tuple[str, str]:
    """Build a monospace table for LRG names."""
    lrgs = [l for l in (lrgs or []) if l]
    shown = lrgs[:max_rows]
    w = len(str(len(shown))) + 1
    header = f"{'#':<{w}} LRG"
    underline = "-" * len(header)
    lines = [header, underline]
    for i, l in enumerate(shown, 1):
        lines.append(f"{i:<{w}} {l}")
    extra = f"\n(+{len(lrgs)-max_rows} more)" if len(lrgs) > max_rows else ""
    return "\n".join(lines), extra

def _format_env_table(rows: list[dict], max_rows: int = 30) -> tuple[str, str]:
    """
    Build a monospace table for RunIntegration envs.
    rows: [{"rack":"...","type":"..."}...]
    Returns (table_text, suffix_message_if_truncated)
    """
    rows = [r for r in (rows or []) if r.get("rack")]
    shown = rows[:max_rows]
    w = len(str(len(shown))) + 1  # width for '#'
    header = f"{'#':<{w}} RACK                         TYPE"
    underline = "-" * len(header)
    lines = [header, underline]
    for i, r in enumerate(shown, 1):
        rack = r.get("rack","")
        typ  = r.get("type","")
        lines.append(f"{i:<{w}} {rack:<28} {typ}")
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return "\n".join(lines), extra

def _unwrap_env_list(res: Any, keys: list[str]) -> list:
    """
    From a response that can be:
      - dict with one of keys pointing to a list
      - list directly
      - anything else
    return a list (or []).
    """
    if isinstance(res, dict):
        for k in keys:
            v = res.get(k)
            if isinstance(v, list):
                return v
    if isinstance(res, list):
        return res
    return []

def _parse_env_item(it: Any) -> dict | None:
    """
    Accept:
      - dict {"rack_name": "...", "deploy_type": "..."} or {"rack":"...","type":"..."} etc.
      - string "rack : type" (or "rack:type")
    Return: {"rack":"...","type":"..."} or None
    """
    if isinstance(it, dict):
        rack = it.get("rack") or it.get("rack_name") or it.get("full_rack_name")
        typ  = it.get("type") or it.get("deploy_type") or ""
        if rack:
            return {"rack": str(rack).strip(), "type": str(typ).strip()}
        return None
    if isinstance(it, str):
        parts = it.split(":", 1)
        if parts:
            rack = parts[0].strip()
            typ  = parts[1].strip() if len(parts) > 1 else ""
            if rack:
                return {"rack": rack, "type": typ}
        return None
    return None

def _normalize_idle_envs(res: Any) -> list[dict]:
    """
    Handle dict{"idle_envs":[...]} or list[...] (dicts or strings).
    """
    raw_list = _unwrap_env_list(res, keys=["idle_envs", "result", "items", "data"])
    rows: list[dict] = []
    for it in raw_list:
        parsed = _parse_env_item(it)
        if parsed:
            rows.append(parsed)
    return rows

def _normalize_disabled_envs(res: Any) -> list[dict]:
    """
    Handle dict{"disabled_envs":[...]} or list[...] (dicts or strings).
    """
    raw_list = _unwrap_env_list(res, keys=["disabled_envs", "result", "items", "data"])
    rows: list[dict] = []
    for it in raw_list:
        parsed = _parse_env_item(it)
        if parsed:
            rows.append(parsed)
    return rows

def _unwrap_list(res: Any, keys: list[str]) -> list:
    """Return a list from dict containers or pass through a list; else []."""
    if isinstance(res, dict):
        for k in keys:
            v = res.get(k)
            if isinstance(v, list):
                return v
    if isinstance(res, list):
        return res
    return []

def _normalize_lrg_with_difs(res: Any) -> list[dict]:
    """
    Expect: {"lrgs_with_difs":[{"lrg":...,"sucs":...,"difs":...,"nwdif":...,"intdif":...,"szdif":...,"comments":...}]}
    Return rows as [{"lrg","sucs","difs","nwdif","intdif","szdif","comments"}]
    """
    items = _unwrap_list(res, keys=["lrgs_with_difs", "items", "result", "data"])
    rows = []
    for it in items:
        if not isinstance(it, dict): 
            continue
        rows.append({
            "lrg": str(it.get("lrg","")).strip(),
            "sucs": str(it.get("sucs","")).strip(),
            "difs": str(it.get("difs","")).strip(),
            "nwdif": str(it.get("nwdif","")).strip(),
            "intdif": str(it.get("intdif","")).strip(),
            "szdif": str(it.get("szdif","")).strip(),
            "comments": (str(it.get("comments","")).strip() or ""),
        })
    # keep only rows with an LRG
    return [r for r in rows if r["lrg"]]

def _normalize_dif_details(res: Any) -> list[dict]:
    """
    Expect: {"dif_details":[{"lrg","name","rti_number","rti_assigned_to","rti_status","text","comments"}...]}
    """
    items = _unwrap_list(res, keys=["dif_details", "items", "result", "data"])
    rows = []
    for it in items:
        if not isinstance(it, dict): 
            continue
        rows.append({
            "lrg": str(it.get("lrg","")).strip(),
            "name": str(it.get("name","")).strip(),
            "rti": str(it.get("rti_number","")).strip(),
            "assignee": str(it.get("rti_assigned_to","")).strip(),
            "status": str(it.get("rti_status","")).strip(),
            "text": str(it.get("text","")).strip(),
            "comments": str(it.get("comments","")).strip(),
        })
    return [r for r in rows if r["lrg"] or r["name"]]

def _format_lrg_with_difs_table(rows: list[dict], max_rows: int = 30) -> tuple[str, str]:
    """Monospace table for lrg+counts."""
    rows = rows or []
    shown = rows[:max_rows]
    w = len(str(len(shown))) + 1
    header = f"{'#':<{w}} LRG                         SUCS  DIFS  NWDIF  INTDIF  SZDIF"
    underline = "-" * len(header)
    lines = [header, underline]
    for i, r in enumerate(shown, 1):
        lines.append(f"{i:<{w}} {r['lrg']:<28} {r['sucs']:>4}  {r['difs']:>4}  {r['nwdif']:>5}  {r['intdif']:>6}  {r['szdif']:>5}")
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return "\n".join(lines), extra

def _format_dif_details_table(rows: list[dict], max_rows: int = 20) -> tuple[str, str]:
    """Monospace table for dif details (compact)."""
    rows = rows or []
    shown = rows[:max_rows]
    w = len(str(len(shown))) + 1
    header = f"{'#':<{w}} LRG                         NAME                        RTI        STATUS   ASSIGNEE"
    underline = "-" * len(header)
    lines = [header, underline]
    def clip(s, n): return (s[:n-1] + "…") if len(s) > n else s
    for i, r in enumerate(shown, 1):
        lines.append(
            f"{i:<{w}} {clip(r['lrg'],28):<28} {clip(r['name'],26):<26} {clip(r['rti'],10):<10} {clip(r['status'],7):<7} {clip(r['assignee'],12):<12}"
        )
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return "\n".join(lines), extra

def _unwrap_list_generic(res: Any, keys: list[str]) -> list:
    """Return a list from a dict container (by keys) or pass through list; else []."""
    if isinstance(res, dict):
        for k in keys:
            v = res.get(k)
            if isinstance(v, list):
                return v
    if isinstance(res, list):
        return res
    return []

def _normalize_dif_occurrences(res: Any) -> list[dict]:
    """
    Server shape (find_dif_occurrence):
      {"dif_occurrences":[{"label","lrg","name","rti_number","rti_assigned_to","comments","text"?}], ...}
    Normalize to rows: [{"label","lrg","name","rti","assignee"}]
    """
    items = _unwrap_list_generic(res, keys=["dif_occurrences", "items", "result", "data"])
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append({
            "label":      str(it.get("label","")).strip(),
            "lrg":        str(it.get("lrg","")).strip(),
            "name":       str(it.get("name","")).strip(),
            "rti":        str(it.get("rti_number","")).strip(),
            "assignee":   str(it.get("rti_assigned_to","")).strip(),
            # "comments": str(it.get("comments","")).strip(),   # optional column, can be long
        })
    return [r for r in rows if r["label"] or r["lrg"] or r["name"]]

def _normalize_widespread_issues(res: Any) -> list[dict]:
    """
    Server shape (find_widespread_issues):
      {"widespread_issues":[{"name":"dif_name","lrgs":"l1,l2,..."}], ...}
    Normalize to rows: [{"name","lrgs","count"}]
    """
    items = _unwrap_list_generic(res, keys=["widespread_issues", "items", "result", "data"])
    rows = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("name","")).strip()
        lrgs = str(it.get("lrgs","")).strip()
        cnt = len([x for x in (lrgs.split(",") if lrgs else []) if x.strip()])
        rows.append({"name": name, "lrgs": lrgs, "count": cnt})
    return [r for r in rows if r["name"]]

def _format_dif_occ_table(rows: list[dict], max_rows: int = 25) -> tuple[str, str]:
    """Monospace table for dif occurrences."""
    rows = rows or []
    shown = rows[:max_rows]
    w = len(str(len(shown))) + 1
    header = f"{'#':<{w}} LABEL                       LRG                         DIF                         RTI        ASSIGNEE"
    underline = "-" * len(header)
    def clip(s, n): 
        s = s or ""
        return (s[:n-1] + "…") if len(s) > n else s
    lines = [header, underline]
    for i, r in enumerate(shown, 1):
        lines.append(f"{i:<{w}} {clip(r['label'],27):<27} {clip(r['lrg'],27):<27} {clip(r['name'],27):<27} {clip(r['rti'],10):<10} {clip(r['assignee'],12):<12}")
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return "\n".join(lines), extra

def _format_widespread_table(rows: list[dict], max_rows: int = 20) -> tuple[str, str]:
    """Monospace table for widespread issues (dif name + LRG list + count)."""
    rows = rows or []
    shown = rows[:max_rows]
    w = len(str(len(shown))) + 1
    header = f"{'#':<{w}} DIF NAME                    COUNT  LRGs"
    underline = "-" * len(header)
    def clip(s, n): 
        s = s or ""
        return (s[:n-1] + "…") if len(s) > n else s
    lines = [header, underline]
    for i, r in enumerate(shown, 1):
        lines.append(f"{i:<{w}} {clip(r['name'],26):<26} {str(r['count']):>5}  {clip(r['lrgs'],60)}")
    extra = f"\n(+{len(rows)-max_rows} more)" if len(rows) > max_rows else ""
    return "\n".join(lines), extra

def _split_for_slack(text: str, max_chars: int = 3500) -> list[str]:
    """
    Split long Slack messages into parts under Slack's message limit (~4000 chars safe).
    Preserves code blocks and avoids cutting mid-block or mid-line.
    """
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    parts, current, current_len = [], [], 0

    def flush():
        if current:
            parts.append("".join(current).rstrip())

    for ln in lines:
        if current_len + len(ln) > max_chars:
            flush()
            current, current_len = [], 0
        current.append(ln)
        current_len += len(ln)

    flush()
    return parts

def _humanize_html(text: str) -> str:
    """
    Convert any embedded HTML (```html ... ```, or raw <tags>) to human-readable plain text.
    """
    if not text:
        return text

    # Detect ```html blocks
    html_blocks = re.findall(r"```html(.*?)```", text, re.DOTALL)
    for block in html_blocks:
        readable = _strip_html_to_text(block)
        text = text.replace(f"```html{block}```", readable)

    # If there are remaining raw tags outside code fences, sanitize them too
    if "<" in text and ">" in text:
        text = _strip_html_to_text(text)

    return text.strip()

def _strip_html_to_text(fragment: str) -> str:
    """
    Strip HTML tags, decode entities, preserve headings/lists as readable text.
    """
    try:
        soup = BeautifulSoup(fragment, "html.parser")

        # Replace <br> with newlines
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Headings → uppercase section titles
        for h in soup.find_all(["h1","h2","h3","h4","h5","h6","div"]):
            if "heading" in (h.get("class") or []):
                h.insert_before("\n" + h.get_text(strip=True).upper() + "\n")
                h.decompose()

        # List items → table-like numbered lines
        for ul in soup.find_all("ul"):
            items = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
            if items:
                ul.replace_with("\n# | ITEM\n--|-----\n" + "\n".join(f"{i+1} | {itm}" for i, itm in enumerate(items)))

        plain = soup.get_text("\n", strip=True)
        return html.unescape(plain)
    except Exception as e:
        print("[HTML sanitize error]", e)
        return fragment

# ---------------------------------------------------------------------------
# Slack app
# ---------------------------------------------------------------------------
#app = App(token=SLACK_BOT_TOKEN)
app = AsyncApp(token=SLACK_BOT_TOKEN)

# MCP clients (persistent stdio)
RUNINTEG_CLIENT = PersistentMCPClient(RUNINTEG_CMD)
OEDA_CLIENT     = PersistentMCPClient(OEDA_CMD)
RAG_CLIENT      = PersistentMCPClient(RAG_CMD)
SUM_CLIENT      = PersistentMCPClient(SUM_CMD)
GENAI4TEST_CLIENT = PersistentMCPClient(GENAI4TEST_CMD)
LABELHEALTH_CLIENT = PersistentMCPClient(LABELHEALTH_CMD)

# LangGraph agent setup
CURRENT_THREAD_ID = contextvars.ContextVar("current_thread_id", default=None)
TOOL_RUN_RESULTS: defaultdict[str, List[dict[str, Any]]] = defaultdict(list)

THREAD_HISTORY: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
def _append_history(thread_id: str, role: str, content: str | None):
    if not thread_id or not content:
        return
    history = THREAD_HISTORY[thread_id]
    history.append({"role": role, "content": content})
    if len(history) > 20:
        del history[:-20]

def _format_json(data: Any) -> str:
    try:
        return json.dumps(data, indent=2, default=str)
    except Exception:
        return str(data)

# Maps thread_id -> list of artifacts (most recent last)
THREAD_ARTIFACTS: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
def _record_artifact(thread_id: str, filename: str, local_path: str, source: str):
    """
    Remember a local file created/obtained in this thread for later reuse.
    We keep paths to temp files; do not unlink them until done uploading.
    """
    if not (thread_id and local_path and OS.path.exists(local_path)):
        return
    THREAD_ARTIFACTS[thread_id].append({
        "filename": filename,
        "local_path": local_path,
        "source": source,
        "ts": time.time(),
    })

def _latest_artifact(thread_id: str) -> dict | None:
    items = THREAD_ARTIFACTS.get(thread_id) or []
    return items[-1] if items else None


class GenerateOedaArgs(BaseModel):
    request: str = Field(..., description="Full natural-language request describing the desired Exadata configuration.")


@tool("generate_oedaxml", args_schema=GenerateOedaArgs)
async def generate_oedaxml_tool(request: str) -> str:
    "Generate Exadata configuration artifacts (minconfig.json and es.xml). Always pass the full user request."
    payload = {
        "request": request,
        "genoedaxml_path": DEFAULT_GENXML,
        "return_xml": True,
        "force_mock": True,
    }
    res = await asyncio.to_thread(OEDA_CLIENT.call_tool, "generate_oedaxml", payload)
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "generate_oedaxml", "result": res})
    if res.get("error"):
        return _format_json({"status": "error", "message": res["error"]})
    summary = {
        "status": "ok",
        "live_migration_check": res.get("live_mig_check"),
        "live_migration_reason": res.get("live_mig_reason"),
        "rack_description": res.get("rack_desc"),
        "minconfig_json": res.get("minconfig_json"),
        "es_xml_path": res.get("es_xml_path"),
        "es_xml_available": bool(res.get("es_xml_b64")),
        "notes": res.get("note") or res.get("generator"),
        "post_steps": "cd oss/test/tsage/sosd && run doimageoeda.sh -xml [your/xml/path] -error_report -skip_ahf -remote -skip_qinq_checks_cell",
    }
    return _format_json(summary)


class RunintegrationStatusArgs(BaseModel):
    rack: str = Field(..., description="Rack identifier such as scaXXXadmYY.")


@tool("runintegration_status", args_schema=RunintegrationStatusArgs)
async def runintegration_status_tool(rack: str) -> str:
    "Check RunIntegration status for a specific rack."
    res = await asyncio.to_thread(RUNINTEG_CLIENT.status, rack)
    return _format_json(res)


@tool("runintegration_idle_envs")
async def runintegration_idle_envs_tool() -> str:
    "List idle RunIntegration environments."
    res = await asyncio.to_thread(RUNINTEG_CLIENT.idle_envs)
    print("[DEBUG idle_envs raw]", type(res), (res[:2] if isinstance(res, list) else res))
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "runintegration_idle_envs", "result": res})
    return _format_json(res)

@tool("runintegration_disabled_envs")
async def runintegration_disabled_envs_tool() -> str:
    "List disabled RunIntegration environments."
    res = await asyncio.to_thread(RUNINTEG_CLIENT.disabled_envs)
    print("[DEBUG idle_envs raw]", type(res), (res[:2] if isinstance(res, list) else res))
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "runintegration_disabled_envs", "result": res})
    return _format_json(res)


class BugTestArgs(BaseModel):
    bug_no: str = Field(..., description="Bug number, e.g. 35123456")
    email: str | None = Field(None, description="Email to use (optional)")
    agent: str | None = Field(None, description="genai4test agent (optional)")

@tool("run_bug_test", args_schema=BugTestArgs)
async def run_bug_test_tool(bug_no: str, email: str | None = None, agent: str | None = None) -> str:
    """
    Generate a shell or sql test for a bug via genai4test.
    Returns a summary with script/code and optional file URL.
    """
    args = {"bug_no": bug_no}
    if email: args["email"] = email
    if agent: args["agent"] = agent

    # offload MCP call to a thread (PersistentMCPClient is sync)
    res = await asyncio.to_thread(GENAI4TEST_CLIENT.call_tool, "run_bug_test", args)

    # keep for later slack rendering / debugging
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "run_bug_test", "result": res})

    # If failed, surface the error details directly so Slack shows them
    if not isinstance(res, dict) or not res.get("ok"):
        err = (isinstance(res, dict) and (res.get("error") or res.get("status"))) or "unknown error"
        body = isinstance(res, dict) and res.get("body")
        details = f":x: genai4test failed for bug {bug_no}: {err}"
        if body:
            details += "\n```" + str(body)[:500] + "```"
        # still return text so the agent can show it verbatim
        return details

    # success path: return compact JSON the agent can read
    return _format_json(res) if isinstance(res, dict) else str(res)

class LabelSeriesArgs(BaseModel):
    series: str = Field(..., description="Name of the series. The series can be 'OSS_MAIN','OSS_25.2' etc..")
    n: int | None = Field(None, description="Number of labels")

@tool("get_labels_from_series", args_schema=LabelSeriesArgs)
async def get_labels_from_series(series: str, n: int = 10) -> dict:
    """
    Get n recent labels from the given series. The series can be "OSS_MAIN", "OSS_25.2" etc..
    """
    args = {"series": series}
    if n: args["n"] = n

    # offload MCP call to a thread (PersistentMCPClient is sync)
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_labels_from_series", args)
    print("res",res)
    print("args",args)

    # keep for later slack rendering / debugging
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_labels_from_series", "result": res})
        

    # ✅ Only treat as failure if the server returned an 'error'
    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        details = f":x: label health app failed for series `{series}`: {err}"
        return details

    # success path: return compact JSON text so the agent can echo nicely
    return _format_json(res)

class LrgsFromRegressArgs(BaseModel):
    regress: str = Field(..., description="Regress name, e.g. 'SAGE_FC' or 'EXAC_REGRESS'")

@tool("get_lrgs_from_regress", args_schema=LrgsFromRegressArgs)
async def get_lrgs_from_regress_tool(regress: str) -> str:
    """
    Return LRGs associated with the given regress. Example regress: 'SAGE_FC', 'EXAC_REGRESS'.
    """
    regress = (regress or "").strip()
    if not regress:
        return ":x: Please provide a non-empty regress name."

    # offload MCP call to a thread (PersistentMCPClient is sync)
    res = await asyncio.to_thread(
        LABELHEALTH_CLIENT.call_tool,
        "get_lrgs_from_regress",
        {"regress": regress},
    )

    # keep for Slack enrichment / debugging
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_lrgs_from_regress", "result": res})

    # Treat only explicit 'error' as failure
    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for regress `{regress}`: {err}"

    # success: return compact JSON so the agent can echo nicely
    return _format_json(res)

class LrgWithDifsArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrgs: str | None = Field(None, description="Optional comma-separated LRGs to filter")
    regress: str | None = Field(None, description="Optional regress filter")

@tool("find_lrg_with_difs", args_schema=LrgWithDifsArgs)
async def find_lrg_with_difs_tool(label: str, lrgs: str | None = None, regress: str | None = None) -> str:
    """
    List LRGs with difs/failures for a label (optionally filtered by specific LRGs).
    """
    args = {"label": label}
    if lrgs: args["lrgs"] = lrgs
    if regress: args["regress"] = regress
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_lrg_with_difs", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_lrg_with_difs", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for label `{label}`: {err}"

    return _format_json(res)

class DifDetailsArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrgs: str | None = Field(None, description="Comma-separated LRGs to filter")
    name: str | None = Field(None, description="Filter: dif name")
    status: str | None = Field(None, description="Filter: status")
    text: str | None = Field(None, description="Filter: text/description")
    rti_number: str | None = Field(None, description="Filter: RTI number")
    rti_assigned_to: str | None = Field(None, description="Filter: RTI assignee")
    rti_status: str | None = Field(None, description="Filter: RTI status")
    comments: str | None = Field(None, description="Filter: comment content")
    regress: str | None = Field(None, description="Filter: regress name")

@tool("find_dif_details", args_schema=DifDetailsArgs)
async def find_dif_details_tool(
    label: str,
    lrgs: str | None = None,
    name: str | None = None,
    status: str | None = None,
    text: str | None = None,
    rti_number: str | None = None,
    rti_assigned_to: str | None = None,
    rti_status: str | None = None,
    comments: str | None = None,
    regress: str | None = None,
) -> str:
    """
    Detailed dif/failure info for a label with optional filters.
    """
    args = {
        "label": label,
        **({} if lrgs is None else {"lrgs": lrgs}),
        **({} if name is None else {"name": name}),
        **({} if status is None else {"status": status}),
        **({} if text is None else {"text": text}),
        **({} if rti_number is None else {"rti_number": rti_number}),
        **({} if rti_assigned_to is None else {"rti_assigned_to": rti_assigned_to}),
        **({} if rti_status is None else {"rti_status": rti_status}),
        **({} if comments is None else {"comments": comments}),
        **({} if regress is None else {"regress": regress}),
    }
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_dif_details", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_dif_details", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for label `{label}`: {err}"

    return _format_json(res)

class DifOccurrenceArgs(BaseModel):
    dif: str = Field(..., description="Dif name to search for")
    series: str = Field(..., description="Series, e.g. 'OSS_MAIN' or 'OSS_25.1'")

@tool("find_dif_occurrence", args_schema=DifOccurrenceArgs)
async def find_dif_occurrence_tool(dif: str, series: str) -> str:
    """
    Find occurrences of a dif in a given series.
    """
    args = {"dif": dif, "series": series}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_dif_occurrence", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_dif_occurrence", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for dif `{dif}` in series `{series}`: {err}"

    return _format_json(res)

class WidespreadArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    n: int = Field(3, description="Minimum occurrences to consider widespread (default: 3)")

@tool("find_widespread_issues", args_schema=WidespreadArgs)
async def find_widespread_issues_tool(label: str, n: int = 3) -> str:
    """
    List widespread issues (dif name + lrgs) for a label.
    """
    args = {"label": label, "n": n}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_widespread_issues", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_widespread_issues", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed for label `{label}`: {err}"

    return _format_json(res)

class FindCrashesArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrgs: str | None = Field(None, description="Optional comma-separated LRGs filter (e.g. 'lrg1,lrg2')")
    lrg: str | None = Field(None, description="Optional single LRG filter (alternative to lrgs)")
    regress: str | None = Field(None, description="Optional regress filter")

@tool("find_crashes", args_schema=FindCrashesArgs)
async def find_crashes_tool(label: str, lrgs: str | None = None, lrg: str | None = None, regress: str | None = None) -> str:
    """
    Get crash information for a specific label from the Label Health service.
    """
    args = {"label": label}
    if lrgs:   args["lrgs"] = lrgs
    if lrg:    args["lrg"] = lrg
    if regress: args["regress"] = regress
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "find_crashes", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "find_crashes", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get crashes for `{label}`: {err}"

    return _format_json(res)

class LrgHistoryArgs(BaseModel):
    lrg: str = Field(..., description="LRG identifier, e.g. 'lrgrhexaprovcluster'")
    series: str | None = Field(None, description="Optional series filter, e.g. 'OSS_MAIN'")
    n: int | None = Field(10, description="Number of history labels to return (default: 20)")

@tool("get_lrg_history", args_schema=LrgHistoryArgs)
async def get_lrg_history_tool(lrg: str, series: str | None = None, n: int | None = 20) -> str:
    """
    Get LRG history for a given LRG, optionally filtered by series and number of labels.
    """
    args = {"lrg": lrg}
    if series:
        args["series"] = series
    if n is not None:
        args["n"] = n

    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_lrg_history", args)

    # record for debugging / optional enrichments
    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_lrg_history", "result": res})

    # error handling
    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get LRG history for `{lrg}`: {err}"

    # success: return compact JSON the agent can render (prompt tables)
    return _format_json(res)

# --- query_ai_crash_summary -------------------------------------------------
class AiCrashSummaryArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    lrg: str = Field(..., description="LRG identifier")
    dif_name: str = Field(..., description="Dif name")

@tool("query_ai_crash_summary", args_schema=AiCrashSummaryArgs)
async def query_ai_crash_summary_tool(label: str, lrg: str, dif_name: str) -> str:
    """
    Get AI-generated crash summary for a specific crash (label + LRG + dif).
    """
    args = {"label": label, "lrg": lrg, "dif_name": dif_name}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "query_ai_crash_summary", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "query_ai_crash_summary", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get AI crash summary for `{label}` / `{lrg}` / `{dif_name}`: {err}"

    return _format_json(res)

# --- get_se_rerun_details ---------------------------------------------------
class SeRerunArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")
    se_job_id: str | None = Field(None, description="Optional SE job id (7–9 digits)")

@tool("get_se_rerun_details", args_schema=SeRerunArgs)
async def get_se_rerun_details_tool(label: str, se_job_id: str | None = None) -> str:
    """
    Get SE rerun details for a label (or a specific SE job id if provided).
    """
    args = {"label": label}
    if se_job_id:
        args["se_job_id"] = se_job_id

    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_se_rerun_details", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_se_rerun_details", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        target = f"`{label}` (SE job `{se_job_id}`)" if se_job_id else f"`{label}`"
        return f":x: label health app failed to get SE rerun details for {target}: {err}"

    return _format_json(res)

# --- get_regress_summary ----------------------------------------------------
class RegressSummaryArgs(BaseModel):
    regress: str = Field(..., description="Regress name, e.g. 'SAGE_FC'")
    series: str = Field(..., description="Series, e.g. 'OSS_MAIN'")

@tool("get_regress_summary", args_schema=RegressSummaryArgs)
async def get_regress_summary_tool(regress: str, series: str) -> str:
    """
    Get regress summary for a regress/series pair.
    """
    args = {"regress": regress, "series": series}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_regress_summary", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_regress_summary", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get regress summary for `{regress}` in `{series}`: {err}"

    return _format_json(res)

# --- get_label_info ---------------------------------------------------------
class LabelInfoArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")

@tool("get_label_info", args_schema=LabelInfoArgs)
async def get_label_info_tool(label: str) -> str:
    """
    Get detailed information about a specific label.
    """
    args = {"label": label}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_label_info", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_label_info", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get label info for `{label}`: {err}"

    return _format_json(res)

# --- get_ai_label_summary ---------------------------------------------------
class AiLabelSummaryArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")

@tool("get_ai_label_summary", args_schema=AiLabelSummaryArgs)
async def get_ai_label_summary_tool(label: str) -> str:
    """
    Get AI-generated summary for a label (if previously created).
    """
    args = {"label": label}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_ai_label_summary", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_ai_label_summary", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get AI label summary for `{label}`: {err}"

    return _format_json(res)

# --- generate_ai_label_summary ----------------------------------------------
class GenerateAiLabelSummaryArgs(BaseModel):
    label: str = Field(..., description="Label name, e.g. 'OSS_MAIN_LINUX.X64_250929'")

@tool("generate_ai_label_summary", args_schema=GenerateAiLabelSummaryArgs)
async def generate_ai_label_summary_tool(label: str) -> str:
    """
    Generate an AI label summary for the given label (long-running call).
    """
    args = {"label": label}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "generate_ai_label_summary", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "generate_ai_label_summary", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to generate AI label summary for `{label}`: {err}"

    return _format_json(res)

# --- get_delta_diffs_between_labels -----------------------------------------
class DeltaDiffsArgs(BaseModel):
    label_1: str = Field(..., description="Source label")
    compare_labels: str = Field(..., description="Comma-separated list of labels to compare against")
    show_common: str = Field(..., description="Whether to show common diffs ('Y' or 'N')")

@tool("get_delta_diffs_between_labels", args_schema=DeltaDiffsArgs)
async def get_delta_diffs_between_labels_tool(label_1: str, compare_labels: str, show_common: str) -> str:
    """
    Get delta diffs between labels (source vs. compare set).
    """
    args = {"label_1": label_1, "compare_labels": compare_labels, "show_common": show_common}
    res = await asyncio.to_thread(LABELHEALTH_CLIENT.call_tool, "get_delta_diffs_between_labels", args)

    thread_id = CURRENT_THREAD_ID.get()
    if thread_id:
        TOOL_RUN_RESULTS[thread_id].append({"name": "get_delta_diffs_between_labels", "result": res})

    if not isinstance(res, dict) or res.get("error"):
        err = (isinstance(res, dict) and res.get("error")) or "unknown error"
        return f":x: label health app failed to get delta diffs for `{label_1}`: {err}"

    return _format_json(res)



class RagQueryArgs(BaseModel):
    question: str = Field(..., description="Question to answer using the Exadata knowledge base.")
    k: int = Field(3, description="Number of supporting documents to retrieve (default 3).")


@tool("rag_query", args_schema=RagQueryArgs)
async def rag_query_tool(question: str, k: int = 3) -> str:
    "Retrieve grounded answers about Exadata and Oracle topics."
    res = await asyncio.to_thread(RAG_CLIENT.call_tool, "rag_query", {"question": question, "k": k})
    return _format_json(res)


class SummarizeTextArgs(BaseModel):
    text: str = Field(..., description="Text that should be summarized.")


@tool("summarize_text", args_schema=SummarizeTextArgs)
async def summarize_text_tool(text: str) -> str:
    "Summarize a block of text."
    res = await asyncio.to_thread(SUM_CLIENT.call_tool, "lc_summarize_text", {"text": text})
    return _format_json(res)


def slack_recurring_prompt(state: AgentState) -> List[AnyMessage]:
    prompt = f"""You are Exadata Slack Assistant. You help users operate Oracle Exadata environments.
OUTPUT & FORMATTING RULES (STRICT)
- Lists → table: When ANY tool returns a LIST, reply with ONE short sentence of context and ALWAYS render the list as a compact monospace table in a fenced code block (```), with a header row. Never use bullets for tool outputs.
- Summaries → table-like block: When ANY tool returns a SUMMARY (single object or narrative), ALWAYS render the summary in a fenced code block (```), with a simple header row. Keep the lead-in sentence short.
- Attachments note: Mention when attachments (such as es.xml) will appear if relevant. Only call tools when needed. Keep prose Slack-short and precise. If no tool is appropriate, answer directly.

TOOL SELECTION
Use the following tools when applicable:
- generate_oedaxml: turn a natural-language hardware request into minconfig.json and es.xml. Default genoedaxml path is {DEFAULT_GENXML}. The tool output includes live migration checks and whether an es.xml attachment will be provided. Mention any failures clearly.
- runintegration_status: report the status of a specific rack in RunIntegration.
- runintegration_idle_envs: list idle RunIntegration environments that are ready to use.
- runintegration_disabled_envs: list disabled RunIntegration environments.
- summarize_text: summarize provided text when the user explicitly asks for a summary without a PDF attachment.
- rag_query: answer general Exadata or Oracle questions with sourced references. When you use this tool, include the cited titles in your reply. For general informational queries (“how to…”, “what is…”, “explain…”, guides, steps) about Exadata/Exascale, ALWAYS call the rag_query tool first to retrieve grounded sources before answering. Do not answer from memory.
- get_labels_from_series: Get n recent labels from the given series. The series can be "OSS_MAIN", "OSS_25.2" etc..
- get_lrgs_from_regress: list all LRGs associated with a regress (e.g., 'SAGE_FC').
- find_lrg_with_difs: list LRGs with dif counts for a label (optionally filtered).
- find_dif_details: detailed dif entries for a label with filters (rti, status, name, etc).
- find_dif_occurrence: find occurrences of a given dif across a series (returns label, LRG, dif name, RTI, assignee).
- find_widespread_issues: list widespread difs for a label (dif name, affected LRGs, count).
- find_crashes: list crash entries for a given label (LRG, name, status, RTI number, assignee, comments).
- get_lrg_history: list history entries for an LRG (optionally filter by series and number of labels).
- query_ai_crash_summary: AI-generated crash summary for a specific crash (label + LRG + dif).
- get_se_rerun_details: SE rerun details for a label (optionally filter by SE job id).
- get_regress_summary: weekly regress summary for a regress in a series.
- get_label_info: detailed information for a label (raw fields).
- get_ai_label_summary: fetch an existing AI-generated summary for a label.
- generate_ai_label_summary: generate an AI summary for a label (may take time).
- get_delta_diffs_between_labels: delta diffs for label_1 vs. compare_labels (show_common = 'Y' or 'N').

TABLE FORMATTING GUIDELINES (TOOL-SPECIFIC)

RunIntegration:
- runintegration_idle_envs: Input may be list of dicts with “rack_name” and “deploy_type”. Table columns: “# | RACK | TYPE”.
- runintegration_disabled_envs: Input may be list of strings “rack : type” or dicts. Normalize to columns “# | RACK | TYPE”.

Label Health – Series/Regress:
- get_labels_from_series: Table columns “# | LABEL”.
- get_lrgs_from_regress: Table columns “# | LRG”.
- find_lrg_with_difs: Each row has “lrg, sucs, difs, nwdif, intdif, szdif”. Table columns “# | LRG | SUCS | DIFS | NWDIF | INTDIF | SZDIF”.
- get_lrg_history: Table with columns like “# | LABEL | LRG | (other fields if present)”.

Label Health – Difs:
- find_dif_details: Rows with “lrg, name, rti_number, rti_assigned_to, rti_status, text/comments”. Table columns “# | LRG | NAME | RTI | STATUS | ASSIGNEE”. Truncate NAME to ~26 chars.
- find_dif_occurrence: Rows with “label, lrg, name, rti_number, rti_assigned_to”. Table columns “# | LABEL | LRG | DIF | RTI | ASSIGNEE”.
- find_widespread_issues: Rows with “name, lrgs (comma-separated), count=number of LRGs”. Table columns “# | DIF NAME | COUNT | LRGs”.

Label Health – Crashes:
- find_crashes: Rows with “lrg, name, status, rti_number, rti_assigned_to”. Table columns “# | LRG | NAME | STATUS | RTI | ASSIGNEE”.

Also print out the actual error message if you encounter.
"""
    return [{"role": "system", "content": prompt}] + state["messages"]


LLM = make_llm()
AGENT_TOOLS = [
    generate_oedaxml_tool,
    runintegration_status_tool,
    runintegration_idle_envs_tool,
    runintegration_disabled_envs_tool,
    summarize_text_tool,
    rag_query_tool,
    run_bug_test_tool, 
    get_labels_from_series,
    get_lrgs_from_regress_tool,
    find_lrg_with_difs_tool,
    find_dif_details_tool, 
    find_dif_occurrence_tool,
    find_widespread_issues_tool,
    find_crashes_tool,
    get_lrg_history_tool,  
    query_ai_crash_summary_tool,
    get_se_rerun_details_tool,
    get_regress_summary_tool, 
    get_label_info_tool,
    get_ai_label_summary_tool,
    generate_ai_label_summary_tool,
    get_delta_diffs_between_labels_tool,
]
AGENT = create_react_agent(
    model=LLM,
    tools=AGENT_TOOLS,
    prompt=slack_recurring_prompt,
)


def _extract_message_text(message: AnyMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _collect_tool_names(messages: List[AnyMessage]) -> List[str]:
    names: List[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None)
            if name:
                names.append(name)
    return names


async def _handle_tool_side_effects(thread_id: str, channel_id: str, thread_ts: str | None, slack_client) -> None:
    runs = TOOL_RUN_RESULTS.pop(thread_id, []) if thread_id else []
    for entry in runs:
        name = entry.get("name")
        result = entry.get("result") or {}
        if name == "generate_oedaxml":
            es_b64 = result.get("es_xml_b64")
            tmp_path = None
            if es_b64:
                try:
                    xml_bytes = base64.b64decode(es_b64)
                    with TF.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
                        tmp.write(xml_bytes)
                        tmp_path = tmp.name
                    await slack_client.files_upload_v2(
                        channels=[channel_id],
                        thread_ts=thread_ts,
                        initial_comment="Attached is the generated `es.xml` file from `generate_oedaxml`.",
                        file=tmp_path,
                        filename="es.xml",
                        title="es.xml",
                    )
                    _record_artifact(thread_ts or str(thread_id), "es.xml", tmp_path, source="oeda")
                except Exception as upload_err:
                    await slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=f":warning: Failed to upload es.xml: {upload_err}",
                    )
            live_check = result.get("live_mig_check")
            if live_check == "fail":
                reason = result.get("live_mig_reason") or "Live migration validation failed."
                rack_desc = result.get("rack_desc") or "unknown"
                await slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f":no_entry: Live migration check failed: {reason}\nRack description: `{rack_desc}`",
                )


# ---------------------------------------------------------------------------
# Slack event handler
# ---------------------------------------------------------------------------
@app.action("fb_up")
async def handle_fb_up(ack, body, client, say):
    await ack()

    # pull tiny payload with the uuid we stored in the buttons' value
    raw = (body.get("actions") or [{}])[0].get("value") or "{}"
    try:
        meta = json.loads(raw)
    except Exception:
        meta = {}
    uid = meta.get("uuid")

    # who / where
    user_id = (body.get("user") or {}).get("id")
    user_label = await _get_user_label(client, user_id)
    channel_id = (body.get("channel") or {}).get("id") or (body.get("container") or {}).get("channel_id")
    msg_ts = (body.get("message") or {}).get("ts") or (body.get("container") or {}).get("message_ts")
    original_text = (body.get("message") or {}).get("text") or "Thanks for the feedback."

    # metrics: append a click line with user
    try:
        append_jsonl(FEEDBACK_PATH, {
            "user": user_label,
            "uuid": uid,
            "ts": utc_iso(),
            "event": "thumb_up",
        })
    except Exception as e:
        print("[feedback up] append_jsonl error:", e)

    # UI: replace buttons with 'thanks' (prefer your feedback_blocks if it supports voted="up")
    try:
        if 'feedback_blocks' in globals():
            blocks = feedback_blocks(original_text, voted="up", payload_json=raw)
        else:
            blocks = feedback_thanks_blocks(original_text, "up")
        await client.chat_update(channel=channel_id, ts=msg_ts, text=original_text, blocks=blocks)
    except Exception as e:
        print("[feedback up] chat_update error:", e)

@app.action("fb_down")
async def handle_fb_down(ack, body, client, say):
    await ack()

    raw = (body.get("actions") or [{}])[0].get("value") or "{}"
    try:
        meta = json.loads(raw)
    except Exception:
        meta = {}
    uid = meta.get("uuid")

    # who / where
    user_id = (body.get("user") or {}).get("id")
    user_label = await _get_user_label(client, user_id)
    channel_id = (body.get("channel") or {}).get("id") or (body.get("container") or {}).get("channel_id")
    msg_ts = (body.get("message") or {}).get("ts") or (body.get("container") or {}).get("message_ts")
    original_text = (body.get("message") or {}).get("text") or "Thanks for the feedback."
    thread_ts = (body.get("message") or {}).get("thread_ts") or msg_ts

    # metrics: click with user
    try:
        append_jsonl(FEEDBACK_PATH, {
            "user": user_label,
            "uuid": uid,
            "ts": utc_iso(),
            "event": "thumb_down",
        })
    except Exception as e:
        print("[feedback down] append_jsonl error:", e)

    # UI: remove buttons now (same visual as before)
    try:
        if 'feedback_blocks' in globals():
            blocks = feedback_blocks(original_text, voted="down", payload_json=raw)
        else:
            blocks = feedback_thanks_blocks(original_text, "down")
        await client.chat_update(channel=channel_id, ts=msg_ts, text=original_text, blocks=blocks)
    except Exception as e:
        print("[feedback down] chat_update error:", e)

    # open modal to collect comment (keep the uuid and context)
    try:
        private_meta = json.dumps({
            "uuid": uid,
            "channel": channel_id,
            "thread_ts": thread_ts,
            "user": user_id,
        })
        await client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "callback_id": "fb_comment_modal",
                "private_metadata": private_meta,
                "title": {"type": "plain_text", "text": "Help us improve"},
                "submit": {"type": "plain_text", "text": "Send"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "fb_comment_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "fb_comment",
                            "multiline": True,
                            "placeholder": {"type": "plain_text", "text": "What didn’t work in this answer?"}
                        },
                        "label": {"type": "plain_text", "text": "Your feedback"}
                    }
                ],
            },
        )
    except Exception as e:
        print("[feedback modal] open error:", e)

@app.view("fb_comment_modal")
async def handle_fb_comment_submission(ack, body, client, view):
    await ack()
    # Extract comment
    try:
        comment = (view["state"]["values"]["fb_comment_block"]["fb_comment"]["value"] or "").strip()
    except Exception:
        comment = ""

    # Restore metadata
    meta_raw = view.get("private_metadata") or "{}"
    try:
        meta = json.loads(meta_raw)
    except Exception:
        meta = {}

    uid = meta.get("uuid")
    channel_id = meta.get("channel")
    thread_ts = meta.get("thread_ts")
    user_id = (body.get("user") or {}).get("id")

    # Record to feedback.jsonl
    user_label = await _get_user_label(client, user_id)
    append_jsonl(FEEDBACK_PATH, {
        "user": user_label,
        "uuid": uid,
        "ts": utc_iso(),
        "event": "feedback_comment",
        "comment": comment,
        "channel": channel_id,
        "thread_ts": thread_ts,
    })

    # Ephemeral thank-you in the same thread (no extra scopes needed)
    try:
        await client.chat_postEphemeral(
            channel=channel_id,
            user=user_id,
            text="Thanks for the feedback! 🙏 We’ll use it to improve.",
            thread_ts=thread_ts,
        )
    except Exception as e:
        print("[feedback modal] ephemeral thank-you error:", e)

@app.event("app_mention")
async def handle_app_mention(event, say, client):
    user_question = event.get("text", "")
    cleaned = " ".join(tok for tok in user_question.split() if not tok.startswith("<@"))
    lower = cleaned.lower()
    channel_id = event["channel"]
    base_ts = event.get("ts") or str(uuid.uuid4())
    thread_ts = event.get("thread_ts") or base_ts
    thread_key = thread_ts or base_ts
    pending_uploads: list[dict[str, str]] = []   # [{"path": "...", "filename": "...", "comment": "..."}]
    
    # Immediate feedback message
    user_id = event.get("user")  # Slack user ID 
    first_name = await _get_first_name(client, user_id)
    await say(f"Hi {first_name}, I received your request.\nPlease wait while I generate a response… :hourglass_flowing_sand:",
          thread_ts=thread_ts)
 

    if "summarize" in lower:
        try:
            files = event.get("files") or []
            had_pdf = False
            for f in files:
                name = f.get("name", "document.pdf")
                mt = f.get("mimetype", "")
                if name.lower().endswith(".pdf") or mt in ("application/pdf", "application/octet-stream"):
                    had_pdf = True
                    await say(f":page_facing_up: Got `{name}` — summarizing…", thread_ts=thread_ts)

                    # Download securely (use Slack token)
                    headers = {"Authorization": f"Bearer {OS.getenv('SLACK_BOT_TOKEN', '')}"}
                    def _dl(u,h):
                        r = requests.get(u, headers=h, timeout=60)
                        r.raise_for_status()
                        return r.headers.get("content-type",""), r.content
                    try:
                        ctype, content = await asyncio.to_thread(_dl, f["url_private_download"], headers)
                    except Exception as e:
                        await say(f":x: Download error for `{name}`: {e}", thread_ts=thread_ts)
                        continue

                    # Basic sanity: ensure it's a PDF
                    if not (content.startswith(b"%PDF") or "pdf" in ctype.lower()):
                        await say(
                            f":x: `{name}` does not look like a valid PDF (content-type `{ctype}` / magic {content[:4]!r}).",
                            thread_ts=thread_ts,
                        )
                        continue

                    # Write temp file
                    with TF.NamedTemporaryFile(delete=False, suffix=f"_{name}") as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name

                    # Call summarizer (wrap in thread; PersistentMCPClient is sync)
                    try:
                        res = await asyncio.to_thread(
                            SUM_CLIENT.call_tool, "lc_summarize_pdf_file", {"path": tmp_path}
                        )
                    except Exception as e:
                        await say(f":x: Summarizer call failed: {e}", thread_ts=thread_ts)
                        # Clean up file
                        try: OS.unlink(tmp_path)
                        except Exception: pass
                        continue

                    # Clean up temp file (we don't need it after the tool returns)
                    try:
                        OS.unlink(tmp_path)
                    except Exception:
                        pass

                    # Log raw for diagnosis (remove after stabilizing)
                    print("[SUM] raw:", res)

                    # Surface errors & missing summary with reasons
                    if not isinstance(res, dict):
                        await say(":x: Summarizer returned a non-JSON response.", thread_ts=thread_ts)
                        continu

                    if res.get("error"):
                        err = res["error"]
                        trace = res.get("trace") or res.get("details") or ""
                        msg = f":x: Summarizer error for `{name}`: {err}"
                        if trace:
                            msg += f"\n```text\n{str(trace)[:1200]}\n```"
                        await say(msg, thread_ts=thread_ts)
                        continue

                    summary = res.get("summary")
                    notes = res.get("notes") or res.get("reason") or res.get("chain_type") or res.get("message")
                    pages = res.get("pages") or res.get("num_pages")

                    if not summary:
                        # No summary but we have a reason—show it
                        msg = f":warning: Summarizer returned no summary for `{name}`."
                        if notes:
                            msg += f"\n*Reason:* {notes}"
                        await say(msg, thread_ts=thread_ts)
                    else:
                        # Success — show summary (code-fenced) + optional notes
                        out = f"*Summary for* `{name}` ({pages if pages is not None else '?'} pages):\n```\n{summary.strip()}\n```"
                        if notes:
                            out += f"\n_Note_: {notes}"
                        await say(out, thread_ts=thread_ts)

            if had_pdf:
                return
        except Exception as e:
            await say(f":x: MCP error (summarizer): {e}", thread_ts=thread_ts)
            return

    if ("jenkins" in lower and ("upgrade loop run" in lower or "upgrade_loop_run" in lower) and
        any(x in lower for x in ["submit", "build", "kick", "start"])):
        try:
            params = _parse_env_params_from_text(lower)
            await say(text="Got it ✅ submitting Jenkins build: UPGRADE_LOOP_RUN / 01_PRE_SETUP_FOR_SM", thread_ts=thread_ts)
            result = await trigger_upgrade_loop_run(params=params)
            msg = "\n".join(filter(None, [
                f"Params: {params}" if params else "",
                f"Queue: {result.get('queue_url')}",
                f"Job: {result.get('job_url')}",
            ]))
            await say(text=msg, thread_ts=thread_ts)
            threading.Thread(
                target=_monitor_and_notify,
                args=(result.get("queue_url"), result.get("job_url"), channel_id, thread_ts),
                daemon=True
            ).start()
        except Exception as e:
            await say(text=f"Trigger failed: `{e}`", thread_ts=thread_ts)
        return

    if any(w in lower for w in ["send", "transfer", "upload"]) and any(w in lower for w in ["file", "attachment", "generated", "test"]):
        # parse destination
        match = re.search(r'\b[\w.-]+@[\d.]+:[\w/\-_.]+\b', user_question)
        if not match:
            await say("❌ Please include a destination like `user@host:/path`.", thread_ts=thread_ts)
            return
        dest = match.group()

        # 1) Prefer explicit attachments (if user provided)
        files = event.get("files") or []
        if files:
            try:
                for f in files:
                    name = f["name"]
                    url = f["url_private_download"]
                    headers = {"Authorization": f"Bearer {OS.getenv('SLACK_BOT_TOKEN', '')}"}
                    # download in a thread
                    def _dl(u,h):
                        r = requests.get(u, headers=h, timeout=60)
                        r.raise_for_status()
                        return r.content
                    content = await asyncio.to_thread(_dl, url, headers)
                    with TF.NamedTemporaryFile(delete=False, suffix=f"_{name}") as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    ok = scp_file_with_key(tmp_path, dest, ssh_key_path="/net/10.32.19.91/export/exadata_images/ImageTests/.pxeqa_connect")
                    if ok:
                        await say(f"✅ Sent `{name}` to `{dest}`", thread_ts=thread_ts)
                    else:
                        await say(f"❌ Failed to send `{name}` to `{dest}`", thread_ts=thread_ts)
                    # keep? usually safe to unlink explicit uploads
                    try: OS.unlink(tmp_path)
                    except Exception: pass
            except Exception as e:
                await say(f"⚠️ Error sending file: {e}", thread_ts=thread_ts)
            return

        # 2) No attachments — reuse the latest artifact generated in this thread
        art = _latest_artifact(thread_ts or str(thread_key))
        if not art or not OS.path.exists(art["local_path"]):
            await say("⚠️ I couldn’t find a recent generated file in this thread. Please attach the file or re-run generation.", thread_ts=thread_ts)
            return

        try:
            ok = scp_file_with_key(art["local_path"], dest, ssh_key_path="/net/10.32.19.91/export/exadata_images/ImageTests/.pxeqa_connect")
            if ok:
                await say(f"✅ Sent `{art['filename']}` to `{dest}`", thread_ts=thread_ts)
            else:
                await say(f"❌ Failed to send `{art['filename']}` to `{dest}`", thread_ts=thread_ts)
        except Exception as e:
            await say(f"⚠️ Error sending file: {e}", thread_ts=thread_ts)
        return

    config = {"configurable": {"thread_id": thread_key}}
    prior_messages = list(THREAD_HISTORY[thread_key])
    agent_messages = prior_messages + [{"role": "user", "content": cleaned}]
    token = CURRENT_THREAD_ID.set(thread_key)
    try:
        result = await AGENT.ainvoke({"messages": agent_messages}, config=config)
    except Exception as agent_err:
        print(f"[AGENT ERROR] {agent_err}")
        traceback.print_exc()
        await say(":x: I hit an error while routing that request—trying a direct search fallback.", thread_ts=thread_ts)
        try:
            fallback = await asyncio.to_thread(RAG_CLIENT.call_tool, "rag_query", {"question": cleaned, "k": 3})
            _append_history(thread_key, "user", cleaned)
            if fallback.get("error"):
                err_msg = f":warning: RAG fallback errored: {fallback['error']}"
                await say(err_msg, thread_ts=thread_ts)
                _append_history(thread_key, "assistant", err_msg)
            else:
                ans = fallback.get("answer", "[no answer]")
                srcs = fallback.get("sources", []) or []
                src_lines = "\n".join(
                    f"• {s.get('title', 'untitled')} ({s.get('source') or 'n/a'})" for s in srcs
                )
                fallback_text = f"{ans}\n\n*Sources:*\n{src_lines or '—'}"
                await say(fallback_text, thread_ts=thread_ts)
                _append_history(thread_key, "assistant", fallback_text)
        except Exception as fallback_err:
            err_msg = f":x: MCP error (RAG fallback): {fallback_err}"
            await say(err_msg, thread_ts=thread_ts)
            _append_history(thread_key, "assistant", err_msg)
        return
    finally:
        CURRENT_THREAD_ID.reset(token)

    messages = result.get("messages") or []
    final_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            final_message = msg
            break
    if final_message is None and messages:
        final_message = messages[-1]
    final_text = _extract_message_text(final_message) if final_message else ""
    tool_names = _collect_tool_names(messages)

    _append_history(thread_key, "user", cleaned)
    if final_text.strip():
        _append_history(thread_key, "assistant", final_text.strip())

    # --- genai4test enrichment: try to attach the generated test file, or at least show a link ---
    # --- genai4test enrichment: summary + inline test + zip, no script in final_text ---
    if "run_bug_test" in tool_names:
        try:
            # get the most recent run_bug_test result for this thread
            recent = None
            for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
                if e.get("name") == "run_bug_test":
                    recent = e.get("result") or {}
                    break

            if isinstance(recent, dict):
                summary = (recent.get("summary") or "").strip()
                code_text = (recent.get("code") or recent.get("sql") or "").strip()
                url = (recent.get("absolute_file_url") or recent.get("file_url") or "").strip()

                # 1) Inline code/script attachment (with summary as the file's initial comment)
                if code_text:
                    fname = _pick_genai4test_filename(recent)
                    # choose suffix from content; keep filename consistent
                    suffix = ".sh" if code_text.startswith("#!") or code_text.startswith("#") else ".sql"
                    with TF.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(code_text.encode("utf-8", errors="replace"))
                        tmp_path = tmp.name

                    comment = f"```text\n{summary}\n```" if summary else "Generated test file"
                    await client.files_upload_v2(
                        channels=[channel_id],
                        thread_ts=thread_ts,
                        initial_comment=comment,
                        file=tmp_path,
                        filename=fname,
                        title=fname,
                    )
                    _record_artifact(thread_ts or str(thread_key), fname, tmp_path, source="genai4test")

                # 2) ZIP attachment (separate upload). No duplicate summary here.
                if url:
                    verify_env = OS.getenv("GENAI4TEST_CA_BUNDLE") or (
                        OS.getenv("GENAI4TEST_VERIFY_SSL", "false").lower() == "true"
                    )
                    try:
                        content = await asyncio.to_thread(_download_bytes, url, verify_env)
                        with TF.NamedTemporaryFile(delete=False, suffix=".zip") as tmpzip:
                            tmpzip.write(content)
                            zip_path = tmpzip.name
                        await client.files_upload_v2(
                            channels=[channel_id],
                            thread_ts=thread_ts,
                            initial_comment="ZIP package from GenAI4Test:",
                            file=zip_path,
                            filename=OS.path.basename(urlparse(url).path),
                            title="Generated ZIP File",
                        )
                    except Exception as zip_err:
                        print("[genai4test] ZIP upload failed:", zip_err)

                # 3) Ensure the posted message does NOT contain the code again.
                # Keep final_text short and let attachments speak.
                final_text = "GenAI4Test: generated test file(s) attached."

        except Exception as e:
            print("[genai4test] enrichment error:", e)
    # --- end enrichment ---


    # # --- get_lrgs_from_regress enrichment: simple table-only version ---
    # if "get_lrgs_from_regress" in set(tool_names):
    #     try:
    #         recent = None
    #         for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
    #             if e.get("name") == "get_lrgs_from_regress":
    #                 recent = e.get("result") or {}
    #                 break

    #         if isinstance(recent, dict):
    #             lrgs = [d.get("lrg") for d in (recent.get("lrgs") or []) if d.get("lrg")]
    #             regress_name = recent.get("regress") or "?"

    #             if lrgs:
    #                 # Always append a single table at the end
    #                 table, extra = _format_lrg_table(lrgs, max_rows=50)
    #                 block = (
    #                     f"*LRGs associated with regress* `{regress_name}` "
    #                     f"({len(lrgs)}):\n```{table}```{extra}"
    #                 )
    #                 final_text = block
    #     except Exception as e:
    #         print("[get_lrgs_from_regress] enrichment error:", e)
    # # --- runintegration_idle_envs
    # if "runintegration_idle_envs" in set(tool_names):
    #     try:
    #         recent = None
    #         for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
    #             if e.get("name") == "runintegration_idle_envs":
    #                 recent = e.get("result")  # can be dict or list
    #                 break

    #         rows = _normalize_idle_envs(recent)  # -> [{"rack","type"}...]
    #         if rows:
    #             table, extra = _format_env_table(rows, max_rows=30)
    #             block = (
    #                 f"*Idle RunIntegration environments* "
    #                 f"({len(rows)}):\n```{table}```{extra}"
    #             )
    #             final_text = block  # table only, same as LRG format
    #         else:
    #             final_text = "*Idle RunIntegration environments:* none found."
    #     except Exception as e:
    #         print("[runintegration_idle_envs] enrichment error:", e)

    # # --- runintegration_disabled_envs
    # if "runintegration_disabled_envs" in set(tool_names):
    #     try:
    #         recent = None
    #         for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
    #             if e.get("name") == "runintegration_disabled_envs":
    #                 recent = e.get("result")  # can be dict or list
    #                 break

    #         rows = _normalize_disabled_envs(recent)  # -> [{"rack","type"}...]
    #         if rows:
    #             table, extra = _format_env_table(rows, max_rows=30)
    #             block = (
    #                 f"*Disabled RunIntegration environments* "
    #                 f"({len(rows)}):\n```{table}```{extra}"
    #             )
    #             final_text = block  # table only, same as LRG format
    #         else:
    #             final_text = "*Disabled RunIntegration environments:* none."
    #     except Exception as e:
    #         print("[runintegration_disabled_envs] enrichment error:", e)
    # # --- find_lrg_with_difs: table-only like LRGs ---
    # if "find_lrg_with_difs" in set(tool_names):
    #     try:
    #         recent = None
    #         for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
    #             if e.get("name") == "find_lrg_with_difs":
    #                 recent = e.get("result") or {}
    #                 break
    #         rows = _normalize_lrg_with_difs(recent)
    #         label_name = (recent.get("label") if isinstance(recent, dict) else "") or "?"
    #         if rows:
    #             table, extra = _format_lrg_with_difs_table(rows, max_rows=30)
    #             block = f"*LRGs with difs for label* `{label_name}` ({len(rows)}):\n```{table}```{extra}"
    #             final_text = block
    #         else:
    #             final_text = f"*LRGs with difs for label* `{label_name}`: none."
    #     except Exception as e:
    #         print("[find_lrg_with_difs] enrichment error:", e)

    # # --- find_dif_details: table-only like LRGs ---
    # if "find_dif_details" in set(tool_names):
    #     try:
    #         recent = None
    #         for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
    #             if e.get("name") == "find_dif_details":
    #                 recent = e.get("result") or {}
    #                 break
    #         rows = _normalize_dif_details(recent)
    #         label_name = (recent.get("label") if isinstance(recent, dict) else "") or "?"
    #         if rows:
    #             table, extra = _format_dif_details_table(rows, max_rows=20)
    #             block = f"*Dif details for label* `{label_name}` ({len(rows)}):\n```{table}```{extra}"
    #             final_text = block
    #         else:
    #             final_text = f"*Dif details for label* `{label_name}`: none."
    #     except Exception as e:
    #         print("[find_dif_details] enrichment error:", e)
    # # --- find_dif_occurrence: table-only ---
    # if "find_dif_occurrence" in set(tool_names):
    #     try:
    #         recent = None
    #         for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
    #             if e.get("name") == "find_dif_occurrence":
    #                 recent = e.get("result") or {}
    #                 break
    #         rows = _normalize_dif_occurrences(recent)
    #         dif = (recent.get("dif") if isinstance(recent, dict) else "") or "?"
    #         series = (recent.get("series") if isinstance(recent, dict) else "") or "?"
    #         if rows:
    #             table, extra = _format_dif_occ_table(rows, max_rows=25)
    #             block = f"*Occurrences of dif* `{dif}` *in series* `{series}` ({len(rows)}):\n```{table}```{extra}"
    #             final_text = block
    #         else:
    #             final_text = f"*Occurrences of dif* `{dif}` *in series* `{series}`: none."
    #     except Exception as e:
    #         print("[find_dif_occurrence] enrichment error:", e)

    # # --- find_widespread_issues: table-only ---
    # if "find_widespread_issues" in set(tool_names):
    #     try:
    #         recent = None
    #         for e in reversed(TOOL_RUN_RESULTS.get(thread_key, [])):
    #             if e.get("name") == "find_widespread_issues":
    #                 recent = e.get("result") or {}
    #                 break
    #         rows = _normalize_widespread_issues(recent)
    #         label_name = (recent.get("label") if isinstance(recent, dict) else "") or "?"
    #         if rows:
    #             table, extra = _format_widespread_table(rows, max_rows=20)
    #             block = f"*Widespread difs for label* `{label_name}` ({len(rows)}):\n```{table}```{extra}"
    #             final_text = block
    #         else:
    #             final_text = f"*Widespread difs for label* `{label_name}`: none."
    #     except Exception as e:
    #         print("[find_widespread_issues] enrichment error:", e)
    # # --- end enrichment ---

    feedback_context = {
        "feature": "langgraph_agent",
        "tools": tool_names,
        "question": cleaned[:300],
    }
    
    if final_text.strip():
        final_text = _humanize_html(final_text)
        chunks = _split_for_slack(final_text.strip(), max_chars=3500)
        total = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            suffix = f" (Part {i}/{total})" if total > 1 else ""
            text_out = (chunk + suffix).rstrip()
            if i == 1:
                text_out = text_out if text_out.endswith("```") else (text_out + "\n```")
            try:
                if i == total:
                    # ✅ Only the LAST chunk gets the feedback buttons
                    await post_with_feedback(
                        app, channel_id, thread_ts, text_out,
                        context=feedback_context,
                        user_id=event.get("user"),
                        client=client,
                    )
                else:
                    # Earlier parts: plain message in the same thread
                    await say(text_out, thread_ts=thread_ts)
            except Exception:
                await say(text_out, thread_ts=thread_ts)
    else:
        await say(":grey_question: I couldn't produce a response for that.", thread_ts=thread_ts)

    # --- DO THE UPLOADS *AFTER* POSTING THE SUMMARY ---
    if pending_uploads:
        for item in pending_uploads:
            try:
                await client.files_upload_v2(
                    channels=[channel_id],
                    thread_ts=thread_ts,                  # keep in the same thread
                    initial_comment=item.get("comment") or "Attachment:",
                    file=item["path"],
                    filename=item["filename"],
                    title=item["filename"],
                )
            finally:
                # clean up temp file
                try:
                    OS.unlink(item["path"])
                except Exception:
                    pass
    await _handle_tool_side_effects(thread_key, channel_id, thread_ts, client)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def _run():
    print("[BOOT] Starting Slack bot...")
    print("  LLM_PROVIDER:", OS.getenv("LLM_PROVIDER"))
    print("  RAG_CMD:", OS.getenv("RAG_CMD"))
    handler = AsyncSocketModeHandler(app, SLACK_APP_TOKEN)
    await handler.start_async()

if __name__ == "__main__":
    try:
        asyncio.run(_run())
    finally:
        # graceful MCP shutdown
        for cli in (RUNINTEG_CLIENT, OEDA_CLIENT, RAG_CLIENT, SUM_CLIENT):
            try: cli.close()
            except Exception: pass
