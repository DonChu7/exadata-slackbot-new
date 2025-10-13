"""
oeda_agent.py
====================

This module provides a simple proof‑of‑concept implementation of a large‑language‑model
agent for generating Oracle Exadata OEDA configuration files.  It illustrates how
an LLM can be called to translate a natural‑language description of an Exadata
deployment into a structured `minconfig.json` file and how that JSON can be fed
to the existing `genoedaxml` script to produce an `es.xml` file.

**Important notes:**

* The code is designed as an example and cannot be run end‑to‑end without an
  appropriate LLM API key and access to the Oracle network.  In the absence of
  an LLM or when running outside Oracle, the `mock_llm_response` function can
  be used to simulate the model output for demonstration purposes.
* `genoedaxml` only exists inside Oracle’s network.  To fully generate an
  `es.xml`, you must run this script on a host where `genoedaxml` is installed
  and reachable.  If it is not available, the script will return the JSON and
  skip XML generation.

Usage example:

sh
export OPENAI_API_KEY="sk-..."
python oeda_agent.py \
  --request "I want to deploy environment scaqat15adm0506 with four clusters on cell disks and with qinq config" \
  --genoedaxml /net/dbdevfssmnt-shared01.dev3fss1phx.databasede3phx.oraclevcn.com/exadata_dev_image_oeda/genoeda/genoedaxml


This will print the generated minconfig.json and the location of the produced
es.xml (if genoedaxml is available and the run succeeds).

"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

LIVE_MIG_PAT = re.compile(r"\blive[-\s]*migration\b", re.I)

def is_live_migration_req(user_request: str) -> bool:
    return bool(LIVE_MIG_PAT.search(user_request or ""))

def apply_live_migration_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce the minimum required knobs for a live-migration-capable env.
    - Exascale VC environment (virtualCluster True)
    - Guest storage on celldisk (uniform across clusters by default)
    Leave security extras (pkey, secure fabric) optional.
    """
    cfg = dict(cfg or {})
    # Must be virtual + exascale
    cfg["virtualCluster"] = True
    cfg["exascale"] = True
    # Ensure cluster count at least 1 for VC
    cfg.setdefault("clusterCount", 1)

    # Prefer uniform celldisk guest storage unless user specified per-cluster map
    if "clusterGuestStorage" not in cfg and "clusterStorage" not in cfg:
        # old key in your mock: guestStorage = "celldisk"; for OEDA we’ll use clusterGuestStorage when clusters >1
        if cfg.get("clusterCount", 1) > 1:
            cfg["clusterGuestStorage"] = ",".join(["celldisk"] * int(cfg["clusterCount"]))
        else:
            cfg["guestStorage"] = "celldisk"

    # Optional toggles (leave as-is if user asked for them elsewhere)
    # cfg.setdefault("qinq", True)    # enable if you want to default secure fabric
    # cfg.setdefault("pkey", True)

    return cfg

try:
    # Importing openai is optional.  If it is not installed or the API key is
    # absent, the agent will fall back to a mock response.
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore


@dataclass
class MinConfig:
    """Dataclass representing the minimal Exadata configuration."""

    rackPrefix: str
    computeCount: int
    cellCount: int
    computeStartId: int
    cellStartId: int
    clusterCount: Optional[int] = 1
    virtualCluster: Optional[bool] = True
    exascale: Optional[bool] = None
    qinq: Optional[bool] = None
    pkey: Optional[bool] = None
    # Additional optional parameters can be added here as needed.


def call_llm_to_generate_json(user_request: str) -> Dict[str, Any]:
    """
    Robustly call an LLM (OpenAI if configured, else local if configured),
    with safe guards:
      - tolerate empty/odd responses
      - strip markdown fences
      - extract first JSON object
      - fallback to mock on any failure
    """
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        s = text.strip()
        # strip common markdown fences
        if s.startswith("```"):
            # remove first fence
            s = s.split("```", 2)
            if len(s) >= 3:
                s = s[1] if not s[1].strip().startswith("{") else s[1]
                s = s if s.strip().startswith("{") else s[2]
            else:
                s = s[-1]
        # find first {...}
        import re, json as _json
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if not m:
            return None
        try:
            return _json.loads(m.group(0))
        except Exception:
            return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if openai is not None and api_key:
        try:
            openai.api_key = api_key
            system_prompt = (
                "You are an Exadata configuration assistant. "
                "Return ONLY a JSON object matching this schema: "
                "{rackPrefix: string, computeCount: int, cellCount: int, "
                "computeStartId: int, cellStartId: int, clusterCount?: int, "
                "virtualCluster?: bool, exascale?: bool, qinq?: bool, pkey?: bool}. "
                "Do not add commentary."
            )
            few_shot_examples = [
                {"role": "user", "content": "generate oedaxml with a exadata baremetal env on rack scaqat17adm03,scaqat17adm04,scaqat17celadm04,scaqat17celadm05,scaqat17celadm06 "},
                {"role": "assistant", "content": '{"rackPrefix":"scaqab08","computeCount":2,"cellCount":3,"computeStartId":5,"cellStartId":7,"virtualCluster":false}'},
                {"role": "user", "content": "Deploy a exascale env with 4 clusters on scaqat15adm0506,scaqat15celadm07-09 that first three clusters on celldisk and last cluster on localdisk."},
                {"role": "assistant", "content": '{"virtualCluster": true, "rackPrefix": "scaqat15", "computeCount": 2, "computeStartId": 5,"cellCount":3,cellStartId": 7,"exascale": true,"clusterGuestStorage":"celldisk,celldisk,celldisk,localdisk","clusterCount": 4}'},
            ]
            messages = [{"role": "system", "content": system_prompt}, *few_shot_examples, {"role": "user", "content": user_request}]
            resp = openai.ChatCompletion.create(model="gpt-4o", messages=messages, temperature=0)
            # guard: choices may be missing/empty
            choices = resp.get("choices") or []
            content = choices[0]["message"]["content"] if choices and "message" in choices[0] else ""
            parsed = _extract_json(content)
            if parsed is not None:
                return parsed
        except Exception as exc:
            print(f"[OEDA] OpenAI call failed: {exc}")

    # local model path
    model_name = os.environ.get("LOCAL_LLM_MODEL")
    if model_name:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            prompt = (
                "You are an Exadata configuration assistant. "
                "Return only a JSON object matching this schema: "
                "{rackPrefix: string, computeCount: int, cellCount: int, "
                "computeStartId: int, cellStartId: int, clusterCount?: int, "
                "virtualCluster?: bool, exascale?: bool, qinq?: bool, pkey?: bool}. "
                "Input:\n" + user_request + "\nJSON:"
            )
            outs = generator(prompt, max_length=256, num_return_sequences=1, do_sample=False)
            text = outs[0]["generated_text"] if outs else ""
            parsed = _extract_json(text)
            if parsed is not None:
                return parsed
        except Exception as exc:
            print(f"[OEDA] Local model failed: {exc}")

    # final fallback
    return mock_llm_response(user_request)



def call_local_llm_or_mock(user_request: str) -> Dict[str, Any]:
    """Attempt to generate JSON using a local transformers model, otherwise mock.

    This helper tries to import `transformers` and load a local language model
    specified via the `LOCAL_LLM_MODEL` environment variable.  The model should
    be instruction‑tuned to follow prompts.  If the library or model cannot be
    loaded, the function falls back to `mock_llm_response`.
    """
    model_name = os.environ.get("LOCAL_LLM_MODEL")
    if not model_name:
        # No local model specified; use mock
        return mock_llm_response(user_request)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

        print(f"Loading local model '{model_name}' … this may take a while")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        # Construct a concise prompt instructing the model to output JSON only
        prompt = (
            "You are an Exadata configuration assistant. "
            "Return only a JSON object matching this schema: "
            "{rackPrefix: string, computeCount: int, cellCount: int, "
            "computeStartId: int, cellStartId: int, clusterCount?: int, "
            "virtualCluster?: bool, exascale?: bool, qinq?: bool, pkey?: bool}. "
            "Based on the following request: \n" + user_request + "\nJSON:"
        )
        # Generate output (limit to a few hundred tokens to avoid excessive runtime)
        gen = generator(prompt, max_length=200, num_return_sequences=1, do_sample=False)
        text = gen[0]["generated_text"]
        # Extract the first JSON object from the generated text
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            print("Local model did not return JSON; falling back to mock response.")
            return mock_llm_response(user_request)
    except Exception as exc:
        print(f"Error using local model: {exc}. Falling back to mock.")
        return mock_llm_response(user_request)


def mock_llm_response(user_request: str) -> Dict[str, Any]:
    """A fallback implementation that returns a hard‑coded JSON for demonstration.

    When no LLM is available, this function attempts to parse the request string
    heuristically.  It handles a few common phrases.  For real deployments,
    replace this with a proper model call.
    """
    # A simple parser to extract hostnames, counts and cluster information.
    result: Dict[str, Any] = {
        "virtualCluster": True,
    }
    import re

    # Extract rack prefix from the first hostname containing 'adm' or 'celadm'
    prefix_match = re.search(r"(sc[a-z0-9]+)(?:adm|celadm)", user_request)
    if prefix_match:
        result["rackPrefix"] = prefix_match.group(1)

    # Count compute and cell hostnames provided explicitly
    # We look for patterns like scaqat15adm05, scaqat15celadm03
    # Extract compute and cell IDs, handling ranges and concatenated identifiers.
    # We first capture the numeric portion after 'adm' or 'celadm'.  For
    # computes, use a negative lookbehind to avoid matching the 'adm' in
    # 'celadm'.  The captured string can be a single number, a concatenated
    # sequence of two?digit numbers (e.g. "0506"), or a range (e.g. "03-05").
    compute_ids = []  # type: list[int]
    cell_ids = []  # type: list[int]
    # Find compute matches
    for m in re.finditer(r"(?<!cel)adm(\d+(?:-\d+)?)", user_request):
        part = m.group(1)
        if '-' in part:
            # Range like 03-05 -> 3,4,5
            try:
                start_str, end_str = part.split('-')
                start_num = int(start_str)
                end_num = int(end_str)
                # Ensure ascending order
                step = 1 if end_num >= start_num else -1
                compute_ids.extend(list(range(start_num, end_num + step, step)))
            except ValueError:
                # Fallback: treat as single number if split fails
                try:
                    compute_ids.append(int(part))
                except ValueError:
                    pass
        else:
            # If part is a concatenated string of numbers and its length
            # suggests multiple two-digit IDs, split into pairs.
            if len(part) > 2 and len(part) % 2 == 0:
                for i in range(0, len(part), 2):
                    segment = part[i : i + 2]
                    try:
                        compute_ids.append(int(segment))
                    except ValueError:
                        pass
            else:
                try:
                    compute_ids.append(int(part))
                except ValueError:
                    pass
    # Find cell matches
    for m in re.finditer(r"celadm(\d+(?:-\d+)?)", user_request):
        part = m.group(1)
        if '-' in part:
            try:
                start_str, end_str = part.split('-')
                start_num = int(start_str)
                end_num = int(end_str)
                step = 1 if end_num >= start_num else -1
                cell_ids.extend(list(range(start_num, end_num + step, step)))
            except ValueError:
                try:
                    cell_ids.append(int(part))
                except ValueError:
                    pass
        else:
            if len(part) > 2 and len(part) % 2 == 0:
                for i in range(0, len(part), 2):
                    segment = part[i : i + 2]
                    try:
                        cell_ids.append(int(segment))
                    except ValueError:
                        pass
            else:
                try:
                    cell_ids.append(int(part))
                except ValueError:
                    pass
    if compute_ids:
        result["computeCount"] = len(compute_ids)
        # Use the lowest id as the start id
        result["computeStartId"] = int(sorted(compute_ids)[0])
    if cell_ids:
        result["cellCount"] = len(cell_ids)
        result["cellStartId"] = int(sorted(cell_ids)[0])

    # Determine clusterCount by looking for a number or word before 'cluster'
    cluster_count = None
    # First look for a numeric count (e.g. "4 clusters")
    num_match = re.search(r"(\d+)\s+cluster", user_request, re.IGNORECASE)
    if num_match:
        cluster_count = int(num_match.group(1))
    else:
        # Look for word numbers
        word_map = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        # Use a raw string for the regex to avoid escape warnings
        pattern = r"\b(" + "|".join(word_map.keys()) + r")\s+cluster"
        word_match = re.search(pattern, user_request, re.IGNORECASE)
        if word_match:
            cluster_count = word_map[word_match.group(1).lower()]
    # Defer inclusion of clusterCount until we know whether the deployment
    # is bare metal or virtual.  Store the parsed value for later use.
    parsed_cluster_count = cluster_count

    # Detect qinq and exascale/celldisk
    user_lower = user_request.lower()
    # If the user mentions bare metal, set virtualCluster to false
    if "baremetal" in user_lower or "bare metal" in user_lower or "bare-metal" in user_lower:
        result["virtualCluster"] = False
    # Recognize qinq synonyms (explicit "qinq" or "secure fabric")
    if "qinq" in user_lower or "secure fabric" in user_lower:
        result["qinq"] = True
    # Detect exascale or cell disk usage (virtual cluster storage)
    if "exascale" in user_lower or "cell disks" in user_lower or "celldisk" in user_lower:
        result["guestStorage"] = "celldisk"
        # If the user explicitly mentions exascale, set the exascale flag
        if "exascale" in user_lower:
            result["exascale"] = True

    # Handle per-cluster guest storage directives such as
    # "first three clusters on celldisk and last cluster on localdisk".
    # We only apply this when a cluster count is known or implied.
    # Determine total number of clusters for assignment.  Use parsed_cluster_count
    # if available, otherwise default to 1 (for virtual deployments) or skip
    # for bare metal (since clusterCount is omitted for BM).
    total_clusters = None
    if result.get("virtualCluster", True):
        if parsed_cluster_count:
            total_clusters = parsed_cluster_count
        else:
            total_clusters = 1
    # Proceed only if total_clusters > 1 to assign per-cluster storage
    if total_clusters and total_clusters > 1:
        # Build a mapping of word numbers to integers for pattern matching
        word_map_full = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }
        # Look for patterns specifying the first N clusters
        import re as _re  # Avoid shadowing outer re
        first_pattern = _re.search(
            r"first\s+(\d+|" + "|".join(word_map_full.keys()) + r")\s+cluster\w*\s+on\s+(celldisk|localdisk)",
            user_lower,
        )
        last_pattern = _re.search(
            r"last\s+(\d+|" + "|".join(word_map_full.keys()) + r")?\s*cluster\w*\s+on\s+(celldisk|localdisk)",
            user_lower,
        )
        if first_pattern and last_pattern:
            # Determine number of clusters and storage types
            first_count_str, first_storage = first_pattern.groups()
            last_count_str, last_storage = last_pattern.groups()
            def parse_count(val: str) -> int:
                if not val or val.strip() == "":
                    return 1
                if val.isdigit():
                    return int(val)
                return word_map_full.get(val, 1)
            first_count = parse_count(first_count_str)
            last_count = parse_count(last_count_str)
            # Clamp counts to the total number of clusters
            if first_count + last_count > total_clusters:
                # Adjust last_count to fit into the total
                last_count = max(total_clusters - first_count, 0)
            # Initialize all clusters to first_storage
            cluster_storage_list = [first_storage] * total_clusters
            # Assign last clusters to last_storage
            for i in range(total_clusters - last_count, total_clusters):
                cluster_storage_list[i] = last_storage
            # Only include clusterGuestStorage if the assignments differ
            if any(s != cluster_storage_list[0] for s in cluster_storage_list[1:]):
                result["clusterGuestStorage"] = ",".join(cluster_storage_list)
                result.pop("guestStorage", None)

        # Parse per-cluster storage type assignments (e.g. "first cluster storage on exc, second cluster storage on asm").
        # Recognize ordinal words mapping to cluster indices.
        ordinal_map = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5,
            "sixth": 6,
            "seventh": 7,
            "eighth": 8,
            "ninth": 9,
            "tenth": 10,
        }
        # Regex pattern to capture assignments like "first cluster storage on exc"
        # Match storage types, prioritising longer names (asmonedv) before shorter ones (asm)
        storage_pattern = _re.findall(
            r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+cluster[^\w]*storage\s+(?:on\s+)?(asmonedv|basedb|exc|asm)",
            user_lower,
        )
        if storage_pattern:
            assignments: Dict[int, str] = {}
            for ordinal_word, storage_type in storage_pattern:
                idx = ordinal_map.get(ordinal_word)
                if idx is not None and idx >= 1 and idx <= total_clusters:
                    assignments[idx] = storage_type
            # Only proceed if at least one assignment found
            if assignments:
                # Default storage: exc if exascale is true, else asm
                default_storage = "exc" if result.get("exascale") else "asm"
                cluster_storage_list2 = [default_storage] * total_clusters
                for idx, stype in assignments.items():
                    cluster_storage_list2[idx - 1] = stype
                # Include clusterStorage when assignments differ from default or specify all clusters
                if any(s != default_storage for s in cluster_storage_list2):
                    result["clusterStorage"] = ",".join(cluster_storage_list2)
    # Decide whether to include clusterCount.  Bare metal deployments do not
    # require a clusterCount (OEDA treats BM as having a single implicit
    # cluster).  For virtual clusters, include the explicit count if provided,
    # otherwise default to one cluster.
    if result.get("virtualCluster", True) is False:
        # Bare metal: omit clusterCount
        pass
    else:
        # Virtual cluster: include parsed count or default to 1
        if parsed_cluster_count:
            result["clusterCount"] = parsed_cluster_count
        else:
            result["clusterCount"] = 1

    # Live-migration hint → enforce VC + exascale + celldisk
    if is_live_migration_req(user_request):
        result = apply_live_migration_defaults(result)

    return result


def run_genoedaxml_with_log(minconfig: Dict[str, Any], genoedaxml_path: str) -> tuple[Optional[str], str]:
    """
    Run genoedaxml and return (xml_path, combined_stdout_stderr).
    """
    import time
    log_buf = []

    if not os.path.isfile(genoedaxml_path):
        msg = f"genoedaxml not found at '{genoedaxml_path}'. Skipping XML generation."
        print(msg); return None, msg

    genoeda_dir = os.path.dirname(os.path.abspath(genoedaxml_path))
    genconfig_dir = os.path.join(genoeda_dir, "WorkDir", "genconfig")
    os.makedirs(genconfig_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as json_file:
        json.dump(minconfig, json_file, indent=2)
        json_file_path = json_file.name

    start_monotonic = time.monotonic()
    cmd = [genoedaxml_path, json_file_path]
    try:
        completed = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
        )
        out = (completed.stdout or "")
        err = (completed.stderr or "")
        log_text = f"{out}\n{err}".strip()
        log_buf.append(log_text)

        if completed.returncode != 0:
            print("[genoedaxml] return code", completed.returncode)
            print("[genoedaxml][stdout]\n" + out)
            print("[genoedaxml][stderr]\n" + err)
            return None, log_text
    finally:
        try: os.unlink(json_file_path)
        except Exception: pass

    newest_path, newest_mtime = None, -1.0
    for root, _, files in os.walk(genconfig_dir):
        for fn in files:
            if fn.endswith("-generated.xml"):
                p = os.path.join(root, fn)
                try:
                    m = os.path.getmtime(p)
                except Exception:
                    continue
                if m > newest_mtime:
                    newest_mtime, newest_path = m, p

    if not newest_path:
        return None, "\n".join(log_buf)

    if time.monotonic() - start_monotonic > 300:
        print("[genoedaxml] Found XML but it may be older than this run")

    return os.path.abspath(newest_path), "\n".join(log_buf)

def run_genoedaxml(minconfig: Dict[str, Any], genoedaxml_path: str) -> Optional[str]:
    """
    Back-compat wrapper: returns only the XML path.
    """
    xml_path, _log = run_genoedaxml_with_log(minconfig, genoedaxml_path)
    return xml_path

_RACK_PAT = re.compile(r"rack\s*description:\s*(?P<desc>.+)", re.I)
_DEDUCED_PAT = re.compile(r"deduced\s+rackDescription\s+to:\s*(?P<desc>.+)", re.I)

def _parse_rack_description(log_text: str) -> Optional[str]:
    for line in (log_text or "").splitlines():
        m = _RACK_PAT.search(line) or _DEDUCED_PAT.search(line)
        if m:
            return m.group("desc").strip()
    return None

def _rack_version_ok(rack_desc: str) -> bool:
    m = re.match(r"\s*X\s*(\d+)", (rack_desc or "").upper())
    return bool(m and int(m.group(1)) >= 10)

def validate_hw_support_from_log(log_text: str) -> tuple[bool, Optional[str], str]:
    desc = _parse_rack_description(log_text)
    if not desc:
        return False, None, "Could not determine rackDescription from genoedaxml output"
    if not _rack_version_ok(desc):
        return False, desc, ("The hardware version doesn't support live migration. "
                             "Pick hardware that is X10 or above.")
    return True, desc, "OK"

def build_agent_request(user_request: str, genoedaxml_path: Optional[str] = None) -> None:
    """High-level function to process a user request and generate JSON/XML.

    Parameters
    ----------
    user_request : str
        Natural language description of the desired configuration.
    genoedaxml_path : Optional[str]
        Path to the genoedaxml script.  If None, XML generation is skipped.
    """
    # Call the LLM (or mock) to get the minconfig dictionary
    config_dict = call_llm_to_generate_json(user_request)

    if is_live_migration_req(user_request):
        config_dict = apply_live_migration_defaults(config_dict)

    print("Generated minconfig.json:")
    print(json.dumps(config_dict, indent=2))

    if not genoedaxml_path:
        print("genoedaxml path not provided; only the JSON configuration was generated.")
        return

    es_xml, log_text = run_genoedaxml_with_log(config_dict, genoedaxml_path)
    # HW gate: X10 or higher only
    ok_hw, rack_desc, reason = validate_hw_support_from_log(log_text)
    if not ok_hw:
        print(f"[LIVE-MIGRATION] {reason} (rackDescription: {rack_desc or 'N/A'})")
        return

    if es_xml:
        print(f"Generated OEDA XML: {es_xml}")
    else:
        print("Failed to generate es.xml.")


def _cli() -> None:
    """Command-line interface for interactive use."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Exadata minconfig and es.xml via an LLM agent")
    parser.add_argument(
        "--request",
        type=str,
        required=True,
        help="Natural language description of the deployment",
    )
    parser.add_argument(
        "--genoedaxml",
        type=str,
        default=None,
        help="Absolute path to genoedaxml script (optional)",
    )
    args = parser.parse_args()
    build_agent_request(args.request, args.genoedaxml)


if __name__ == "__main__":
    _cli()
