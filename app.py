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

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from mcp_client import PersistentMCPClient

import uuid
from metrics_utils import feedback_blocks, record_feedback_click, append_jsonl, utc_iso

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, AIMessage, AnyMessage
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

# Default genoedaxml path (allowlisted in oeda_server)
DEFAULT_GENXML = OS.getenv("GENOEDAXML_PATH",
    "/net/dbdevfssmnt-shared01.dev3fss1phx.databasede3phx.oraclevcn.com/exadata_dev_image_oeda/genoeda/genoedaxml"
)

# ---------------------------------------------------------------------------
# feedbacks 
# ---------------------------------------------------------------------------
def post_with_feedback(app, channel_id: str, thread_ts: str | None, text: str, *,
                       context: dict | None = None) -> str:
    """
    Posts a message with thumbs up/down buttons. Returns the message_ts.
    'context' is saved alongside feedback (tool name, question, etc).
    """
    uid = str(uuid.uuid4())
    record = {
        "uuid": uid,
        "ts": utc_iso(),
        "context": context or {},
        "original_text": text,
    }
    # Persist the full payload once 
    append_jsonl(FEEDBACK_PATH, record)

    tiny_payload = json.dumps({"uuid": uid})

    res = app.client.chat_postMessage(
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

def trigger_upgrade_loop_run(params=None):
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

# ---------------------------------------------------------------------------
# Slack app
# ---------------------------------------------------------------------------
app = App(token=SLACK_BOT_TOKEN)

# MCP clients (persistent stdio)
RUNINTEG_CLIENT = PersistentMCPClient(RUNINTEG_CMD)
OEDA_CLIENT     = PersistentMCPClient(OEDA_CMD)
RAG_CLIENT      = PersistentMCPClient(RAG_CMD)
SUM_CLIENT      = PersistentMCPClient(SUM_CMD)

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


class GenerateOedaArgs(BaseModel):
    request: str = Field(..., description="Full natural-language request describing the desired Exadata configuration.")


@tool("generate_oedaxml", args_schema=GenerateOedaArgs)
def generate_oedaxml_tool(request: str) -> str:
    "Generate Exadata configuration artifacts (minconfig.json and es.xml). Always pass the full user request."
    payload = {
        "request": request,
        "genoedaxml_path": DEFAULT_GENXML,
        "return_xml": True,
        "force_mock": True,
    }
    res = OEDA_CLIENT.call_tool("generate_oedaxml", payload)
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
def runintegration_status_tool(rack: str) -> str:
    "Check RunIntegration status for a specific rack."
    res = RUNINTEG_CLIENT.status(rack)
    return _format_json(res)


@tool("runintegration_idle_envs")
def runintegration_idle_envs_tool() -> str:
    "List idle RunIntegration environments."
    res = RUNINTEG_CLIENT.idle_envs()
    return _format_json(res)


@tool("runintegration_disabled_envs")
def runintegration_disabled_envs_tool() -> str:
    "List disabled RunIntegration environments."
    res = RUNINTEG_CLIENT.disabled_envs()
    return _format_json(res)


class RagQueryArgs(BaseModel):
    question: str = Field(..., description="Question to answer using the Exadata knowledge base.")
    k: int = Field(3, description="Number of supporting documents to retrieve (default 3).")


@tool("rag_query", args_schema=RagQueryArgs)
def rag_query_tool(question: str, k: int = 3) -> str:
    "Retrieve grounded answers about Exadata and Oracle topics."
    res = RAG_CLIENT.call_tool("rag_query", {"question": question, "k": k})
    return _format_json(res)


class SummarizeTextArgs(BaseModel):
    text: str = Field(..., description="Text that should be summarized.")


@tool("summarize_text", args_schema=SummarizeTextArgs)
def summarize_text_tool(text: str) -> str:
    "Summarize a block of text."
    res = SUM_CLIENT.call_tool("lc_summarize_text", {"text": text})
    return _format_json(res)


def slack_recurring_prompt(state: AgentState) -> List[AnyMessage]:
    prompt = f"""You are Exadata Slack Assistant. You help users operate Oracle Exadata environments.
You have access to structured tools and should pick the right one based on the user's request:
- generate_oedaxml: turn a natural-language hardware request into minconfig.json and es.xml. Default genoedaxml path is {DEFAULT_GENXML}. The tool output includes live migration checks and whether an es.xml attachment will be provided. Mention any failures clearly.
- runintegration_status: report the status of a specific rack in RunIntegration.
- runintegration_idle_envs: list idle RunIntegration environments that are ready to use.
- runintegration_disabled_envs: list disabled RunIntegration environments.
- summarize_text: summarize provided text when the user explicitly asks for a summary without a PDF attachment.
- rag_query: answer general Exadata or Oracle questions with sourced references. When you use this tool, include the cited titles in your reply.

Every tool returns JSON; read the fields before answering. Mention when attachments (such as es.xml) will appear. Only call tools when they are needed, and respond concisely in Slack-ready prose. If no tool is appropriate, answer directly."""
    return [{"role": "system", "content": prompt}] + state["messages"]


LLM = make_llm()
AGENT_TOOLS = [
    generate_oedaxml_tool,
    runintegration_status_tool,
    runintegration_idle_envs_tool,
    runintegration_disabled_envs_tool,
    summarize_text_tool,
    rag_query_tool,
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


def _handle_tool_side_effects(thread_id: str, channel_id: str, thread_ts: str | None, slack_client) -> None:
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
                    slack_client.files_upload_v2(
                        channels=[channel_id],
                        thread_ts=thread_ts,
                        initial_comment="Attached is the generated `es.xml` file from `generate_oedaxml`.",
                        file=tmp_path,
                        filename="es.xml",
                        title="es.xml",
                    )
                except Exception as upload_err:
                    slack_client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=f":warning: Failed to upload es.xml: {upload_err}",
                    )
                finally:
                    if tmp_path and OS.path.exists(tmp_path):
                        try:
                            OS.unlink(tmp_path)
                        except Exception:
                            pass
            live_check = result.get("live_mig_check")
            if live_check == "fail":
                reason = result.get("live_mig_reason") or "Live migration validation failed."
                rack_desc = result.get("rack_desc") or "unknown"
                slack_client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f":no_entry: Live migration check failed: {reason}\nRack description: `{rack_desc}`",
                )


# ---------------------------------------------------------------------------
# Slack event handler
# ---------------------------------------------------------------------------
@app.action("fb_up")
def handle_fb_up(ack, body, client, say):
    ack()
    record_feedback_click(body, "up", client)

@app.action("fb_down")
def handle_fb_down(ack, body, client, say):
    ack()
    record_feedback_click(body, "down", client)

@app.event("app_mention")
def handle_app_mention(event, say, client):
    user_question = event.get("text", "")
    cleaned = " ".join(tok for tok in user_question.split() if not tok.startswith("<@"))
    lower = cleaned.lower()
    channel_id = event["channel"]
    base_ts = event.get("ts") or str(uuid.uuid4())
    thread_ts = event.get("thread_ts") or base_ts
    thread_key = thread_ts or base_ts

    if "summarize" in lower:
        try:
            files = event.get("files") or []
            had_pdf = False
            for f in files:
                name = f.get("name", "document.pdf")
                mt = f.get("mimetype", "")
                if name.lower().endswith(".pdf") or mt in ("application/pdf", "application/octet-stream"):
                    had_pdf = True
                    say(f":page_facing_up: Got `{name}` — summarizing…", thread_ts=thread_ts)
                    headers = {"Authorization": f"Bearer {OS.getenv('SLACK_BOT_TOKEN', '')}"}
                    r = requests.get(f["url_private_download"], headers=headers, timeout=60)
                    r.raise_for_status()

                    with TF.NamedTemporaryFile(delete=False, suffix=f"_{name}") as tmp:
                        tmp.write(r.content)
                        tmp_path = tmp.name

                    res = SUM_CLIENT.call_tool("lc_summarize_pdf_file", {"path": tmp_path})

                    try:
                        OS.unlink(tmp_path)
                    except Exception:
                        pass

                    if res.get("error"):
                        say(f":x: Summarizer error: {res['error']}", thread_ts=thread_ts)
                    else:
                        pages = res.get("pages") or res.get("num_pages")
                        notes = res.get("notes", "") or res.get("chain_type", "")
                        summary = res.get("summary")
                        if not summary:
                            say(":warning: Summarizer returned no summary.", thread_ts=thread_ts)
                        else:
                            say(f"*Summary for* `{name}` ({pages if pages is not None else '?'} pages):\n{summary}", thread_ts=thread_ts)
                            if notes:
                                say(f"_Note_: {notes}", thread_ts=thread_ts)
            if had_pdf:
                return
        except Exception as e:
            say(f":x: MCP error (summarizer): {e}", thread_ts=thread_ts)
            return

    if ("jenkins" in lower and ("upgrade loop run" in lower or "upgrade_loop_run" in lower) and
        any(x in lower for x in ["submit", "build", "kick", "start"])):
        try:
            params = _parse_env_params_from_text(lower)
            say(text="Got it ✅ submitting Jenkins build: UPGRADE_LOOP_RUN / 01_PRE_SETUP_FOR_SM", thread_ts=thread_ts)
            result = trigger_upgrade_loop_run(params=params)
            msg = "\n".join(filter(None, [
                f"Params: {params}" if params else "",
                f"Queue: {result.get('queue_url')}",
                f"Job: {result.get('job_url')}",
            ]))
            say(text=msg, thread_ts=thread_ts)
            threading.Thread(
                target=_monitor_and_notify,
                args=(result.get("queue_url"), result.get("job_url"), channel_id, thread_ts),
                daemon=True
            ).start()
        except Exception as e:
            say(text=f"Trigger failed: `{e}`", thread_ts=thread_ts)
        return

    if any(w in lower for w in ["send", "transfer", "upload"]) and any(w in lower for w in ["file", "attachment"]):
        match = re.search(r'\b[\w.-]+@[\d.]+:[\w/\-_.]+\b', user_question)
        if not event.get("files"):
            say("⚠️ You asked me to send a file, but no attachment was found.", thread_ts=thread_ts)
            return
        if not match:
            say("❌ Please include a destination like `user@host:/path`.", thread_ts=thread_ts)
            return
        dest = match.group()
        try:
            for f in event["files"]:
                name = f["name"]
                url = f["url_private_download"]
                headers = {"Authorization": f"Bearer {OS.getenv('SLACK_BOT_TOKEN', '')}"}
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                with TF.NamedTemporaryFile(delete=False, suffix=f"_{name}") as tmp:
                    tmp.write(r.content)
                    tmp_path = tmp.name
                if scp_file_with_key(tmp_path, dest, ssh_key_path="/net/10.32.19.91/export/exadata_images/ImageTests/.pxeqa_connect"):
                    say(f"✅ Sent `{name}` to `{dest}`", thread_ts=thread_ts)
                else:
                    say(f"❌ Failed to send `{name}` to `{dest}`", thread_ts=thread_ts)
                OS.unlink(tmp_path)
        except Exception as e:
            say(f"⚠️ Error sending file: {e}", thread_ts=thread_ts)
        return

    config = {"configurable": {"thread_id": thread_key}}
    prior_messages = list(THREAD_HISTORY[thread_key])
    agent_messages = prior_messages + [{"role": "user", "content": cleaned}]
    token = CURRENT_THREAD_ID.set(thread_key)
    try:
        result = AGENT.invoke({"messages": agent_messages}, config=config)
    except Exception as agent_err:
        print(f"[AGENT ERROR] {agent_err}")
        traceback.print_exc()
        say(":x: I hit an error while routing that request—trying a direct search fallback.", thread_ts=thread_ts)
        try:
            fallback = RAG_CLIENT.call_tool("rag_query", {"question": cleaned, "k": 3})
            _append_history(thread_key, "user", cleaned)
            if fallback.get("error"):
                err_msg = f":warning: RAG fallback errored: {fallback['error']}"
                say(err_msg, thread_ts=thread_ts)
                _append_history(thread_key, "assistant", err_msg)
            else:
                ans = fallback.get("answer", "[no answer]")
                srcs = fallback.get("sources", []) or []
                src_lines = "\n".join(
                    f"• {s.get('title', 'untitled')} ({s.get('source') or 'n/a'})" for s in srcs
                )
                fallback_text = f"{ans}\n\n*Sources:*\n{src_lines or '—'}"
                say(fallback_text, thread_ts=thread_ts)
                _append_history(thread_key, "assistant", fallback_text)
        except Exception as fallback_err:
            err_msg = f":x: MCP error (RAG fallback): {fallback_err}"
            say(err_msg, thread_ts=thread_ts)
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

    feedback_context = {
        "feature": "langgraph_agent",
        "tools": tool_names,
        "question": cleaned[:300],
    }

    if final_text.strip():
        try:
            post_with_feedback(app, channel_id, thread_ts, final_text.strip(), context=feedback_context)
        except Exception:
            say(final_text.strip(), thread_ts=thread_ts)
    else:
        say(":grey_question: I couldn't produce a response for that.", thread_ts=thread_ts)

    _handle_tool_side_effects(thread_key, channel_id, thread_ts, client)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[BOOT] Starting Slack bot...")
    print("  LLM_PROVIDER:", OS.getenv("LLM_PROVIDER"))
    print("  RAG_CMD:", OS.getenv("RAG_CMD"))
    try:
        handler = SocketModeHandler(app, SLACK_APP_TOKEN)
        handler.start()
    finally:
        # graceful MCP shutdown
        for cli in (RUNINTEG_CLIENT, OEDA_CLIENT, RAG_CLIENT, SUM_CLIENT):
            try: cli.close()
            except Exception: pass
