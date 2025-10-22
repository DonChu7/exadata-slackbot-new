#!/usr/bin/env python3
from __future__ import annotations
import os, json, re, traceback
import requests
from typing import Any, Dict, Optional, Type
from mcp.server.fastmcp import FastMCP
from urllib.parse import quote
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from urllib.parse import quote, urlparse, urljoin
import base64
import subprocess
import logging
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastMCP("label_health-mcp")


def make_api_request(url, params=None, timeout=15):
    """Make API request and return response data"""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching data: {e}"
    except ValueError:
        return None, "Invalid JSON returned from API"


def get_field_value(item, field_name, default="", show_if_empty=False):
    """Get field value from item, returning empty string if not present or empty"""
    value = item.get(field_name)
    if value is None or value == "":
        return default if show_if_empty else ""
    return str(value).strip()


def format_structured_output(data_type, items, summary=None):
    """Format output in a clean, structured way for agent consumption"""
    if not items:
        return f"No {data_type} found."

    lines = []
    if summary:
        lines.append(f"{summary}")

    for i, item in enumerate(items, 1):
        # Create a clean dictionary with only non-empty values
        clean_item = {
            k: v for k, v in item.items() if v is not None and str(v).strip()
        }

        if clean_item:
            # More compact format: single line per item
            item_str = f"Item {i}: " + ", ".join(
                f"{k}={v[:100]}{'...' if len(str(v)) > 100 else ''}"
                for k, v in clean_item.items()
            )
            lines.append(item_str)

    return "\n".join(lines)


@app.tool()
def get_labels_from_series(series: str, n: int = 10) -> dict:
    """
    Get recent labels from the given series.

    Parameters:
    - series (str): The series name (e.g., 'OSS_MAIN', 'OSS_25.1')
    - n (int, optional): Number of labels to return (default: 10)

    Returns:
    dict: {
        "labels": [{"label": "OSS_MAIN_LINUX.X64_250929"}, ...],
        "series": "OSS_MAIN",
        "count": 10
    } or {"error": "error message"}
    """
    # Make API request
    url = f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/label_from_series"
    params = {"series": series, "n": n}
    data, error = make_api_request(url, params)
    if error:
        return {"error": f"Error fetching labels for `{series}`: {error}"}

    items = data.get("items") or []
    if not items:
        return {
            "labels": [],
            "series": series,
            "message": f"No labels found for `{series}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        # Create clean item with only meaningful data
        clean_item = {}
        if get_field_value(item, "label"):
            clean_item["label"] = get_field_value(item, "label")

        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "labels": cleaned_items,
        "series": series,
        "count": len(cleaned_items),
    }


@app.tool()
def get_lrg_info(lrg: str) -> dict:
    """
    Get information about an LRG using FTS (Full Text Search) query.

    Parameters:
    - lrg (str): The LRG identifier to search for

    Returns:
    dict: {
        "lrg": "lrgsample",
        "info": "FTS search results..."
    } or {"error": "error message"}
    """
    lrg_value = lrg

    # Run subprocess
    command = [
        "python3",
        "query_fts_refined.py",
        "--db",
        "/home/kbaboota/scripts/label_health_copilot/shray/profiles_fts.db",
        "--q",
        lrg_value,
        "--l2t",
        "/home/kbaboota/scripts/label_health_copilot/shray/jsons/lrg_to_tests.json",
        "--lrgs-json",
        "/home/kbaboota/scripts/label_health_copilot/shray/jsons/lrg_map_with_runtimes.json",
        "--doc-type",
        "LRG",
        "--k",
        "5",
        "--debug",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd="/home/kbaboota/scripts/label_health_copilot/shray",
        )
        if result.returncode == 0:
            return {"lrg": lrg_value, "info": result.stdout}
        else:
            return {"error": f"Error: {result.stderr}"}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


@app.tool()
def get_lrgs_from_regress(regress: str) -> dict:
    """
    Get all LRGs associated with a specific regress.

    Parameters:
    - regress (str): The regress name (e.g., 'SAGE_FC', 'EXAC_REGRESS')

    Returns:
    dict: {
        "lrgs": [{"lrg": "lrgsample"}, ...],
        "regress": "SAGE_FC",
        "count": 5
    } or {"error": "error message"}
    """

    # Validate that the regress exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/suite/{regress}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {
            "error": f"Error validating regress `{regress}`: {validate_error}"
        }

    # Check if regress exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {"error": f"Regress `{regress}` does not exist or has no LRGs."}

    # Make API request
    url = f"https://apex.oraclecorp.com/pls/apex/lrg_times/suite/{regress}"
    data, error = make_api_request(url)
    if error:
        return {"error": f"Error fetching LRGs for `{regress}`: {error}"}

    items = data.get("items") or []
    if not items:
        return {
            "lrgs": [],
            "regress": regress,
            "message": f"No LRGs found for `{regress}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        lrg = get_field_value(item, "lrg")
        if lrg:
            cleaned_items.append({"lrg": lrg})

    return {
        "lrgs": cleaned_items,
        "regress": regress,
        "count": len(cleaned_items),
    }


@app.tool()
def find_crashes(
    label: str,
    lrgs: Optional[str] = None,
    lrg: Optional[str] = None,
    regress: Optional[str] = None,
) -> dict:
    """
    Get crash information for a specific label, with optional LRG and regress filtering.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')
    - lrgs (str, optional): Comma-separated LRGs to filter by (e.g., 'lrg1,lrg2')
    - lrg (str, optional): Single LRG to filter by (alternative to lrgs)
    - regress (str, optional): Regress name to filter by

    Returns:
    dict: {
        "crashes": [
            {
                "lrg": "lrgsample",
                "name": "crash_name",
                "status": "status",
                "rti_number": "RTI-123",
                "rti_assigned_to": "user",
                "comments": "cleaned comments"
            }, ...
        ],
        "label": "OSS_MAIN_LINUX.X64_250929",
        "count": 5
    } or {"error": "error message"}
    """

    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request
    url = "https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/crash_info"
    params = {"label": label}
    if lrgs:
        params["lrg_filter"] = lrgs
    elif lrg:
        params["lrg_filter"] = lrg
    if regress:
        params["regress"] = regress
    data, error = make_api_request(url, params)
    if error:
        return {"error": f"Error fetching crashes for `{label}`: {error}"}

    items = data.get("items") or []
    if not items:
        return {
            "crashes": [],
            "label": label,
            "message": f"No crashes found for `{label}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        # Create clean item with only meaningful data
        clean_item = {}
        if get_field_value(item, "lrg"):
            clean_item["lrg"] = get_field_value(item, "lrg")
        if get_field_value(item, "name"):
            clean_item["name"] = get_field_value(item, "name")
        if get_field_value(item, "status"):
            clean_item["status"] = get_field_value(item, "status")
        if get_field_value(item, "rti_number"):
            clean_item["rti_number"] = get_field_value(item, "rti_number")
        if get_field_value(item, "rti_assigned_to"):
            clean_item["rti_assigned_to"] = get_field_value(
                item, "rti_assigned_to"
            )
        if get_field_value(item, "comments"):
            clean_item["comments"] = clean_html_comments(
                get_field_value(item, "comments")
            )

        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "crashes": cleaned_items,
        "label": label,
        "count": len(cleaned_items),
    }


@app.tool()
def find_lrg_with_difs(
    label: str, lrgs: Optional[str] = None, regress: Optional[str] = None
) -> dict:
    """
    Get LRGs that have difs/failures for a specific label, with optional LRG filtering.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')
    - lrgs (str, optional): Comma-separated list of LRGs to filter by (e.g., 'lrg1,lrg2,lrg3')

    Returns:
    dict: {
        "lrgs_with_difs": [
            {
                "lrg": "lrgsample",
                "sucs": 100,
                "difs": 5,
                "nwdif": 2,
                "intdif": 1,
                "szdif": 2,
                "comments": "cleaned comments"
            }, ...
        ],
        "label": "OSS_MAIN_LINUX.X64_250929",
        "count": 10
    } or {"error": "error message"}
    """
    # Parse lrgs if provided
    lrg_list = None
    if lrgs:
        try:
            lrg_list = [l.strip() for l in lrgs.split(",") if l.strip()]
        except:
            return {"error": "Invalid lrgs format. Use comma-separated list."}

    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request
    url = f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/lrg_diff"
    params = {"label": label}
    if regress:
        params["regress"] = regress
    data, error = make_api_request(url, params)
    if error:
        return {
            "error": f"Error fetching LRGs with difs for `{label}`: {error}"
        }

    items = data.get("items") or []
    if not items:
        return {
            "lrgs_with_difs": [],
            "label": label,
            "message": f"No LRGs with difs found for `{label}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        lrg = get_field_value(item, "lrg")
        if lrg_list and lrg not in lrg_list:
            continue

        # Create clean item with only meaningful data
        clean_item = {}
        if lrg:
            clean_item["lrg"] = lrg
        if get_field_value(item, "sucs"):
            clean_item["sucs"] = get_field_value(item, "sucs")
        if get_field_value(item, "difs"):
            clean_item["difs"] = get_field_value(item, "difs")
        if get_field_value(item, "nwdif"):
            clean_item["nwdif"] = get_field_value(item, "nwdif")
        if get_field_value(item, "intdif"):
            clean_item["intdif"] = get_field_value(item, "intdif")
        if get_field_value(item, "szdif"):
            clean_item["szdif"] = get_field_value(item, "szdif")
        if get_field_value(item, "comments"):
            clean_item["comments"] = clean_html_comments(
                get_field_value(item, "comments")
            )

        if clean_item:
            cleaned_items.append(clean_item)

    if not cleaned_items and lrg_list:
        return {
            "lrgs_with_difs": [],
            "label": label,
            "filtered_lrgs": lrg_list,
            "message": f"No matching LRGs ({', '.join(lrg_list)}) found with difs in `{label}`",
        }

    return {
        "lrgs_with_difs": cleaned_items,
        "label": label,
        "count": len(cleaned_items),
    }


@app.tool()
def find_dif_details(
    label: str,
    lrgs: Optional[str] = None,
    name: Optional[str] = None,
    status: Optional[str] = None,
    text: Optional[str] = None,
    rti_number: Optional[str] = None,
    rti_assigned_to: Optional[str] = None,
    rti_status: Optional[str] = None,
    comments: Optional[str] = None,
    regress: Optional[str] = None,
) -> dict:
    """
    Get detailed dif/failure information for a label with extensive filtering options.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')
    - lrgs (str, optional): Comma-separated LRGs to filter by (e.g., 'lrg1,lrg2')
    - name (str, optional): Filter by dif name
    - status (str, optional): Filter by status
    - text (str, optional): Filter by dif text/description
    - rti_number (str, optional): Filter by RTI number
    - rti_assigned_to (str, optional): Filter by RTI assignee
    - rti_status (str, optional): Filter by RTI status
    - comments (str, optional): Filter by comments content
    - regress (str, optional): Filter by regress name

    Returns:
    dict: {
        "dif_details": [
            {
                "lrg": "lrgsample",
                "name": "dif_name",
                "rti_number": "RTI-123",
                "rti_assigned_to": "user",
                "rti_status": "OPEN",
                "text": "dif description",
                "comments": "cleaned comments"
            }, ...
        ],
        "label": "OSS_MAIN_LINUX.X64_250929",
        "filters_applied": "regress: SAGE_FC; status: OPEN",
        "count": 25
    } or {"error": "error message"}
    """
    # Parse lrgs if provided
    lrg_list = None
    if lrgs:
        try:
            lrg_list = [l.strip() for l in lrgs.split(",") if l.strip()]
        except:
            return {"error": "Invalid lrgs format. Use comma-separated list."}

    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Build query parameters
    query_params = {}
    for param_name, param_value in [
        ("name", name),
        ("status", status),
        ("text", text),
        ("rti_number", rti_number),
        ("rti_assigned_to", rti_assigned_to),
        ("rti_status", rti_status),
        ("comments", comments),
        ("regress", regress),
    ]:
        if param_value:
            query_params[param_name] = param_value

    all_items = []
    base_url = f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/diff_details/{label}"

    if lrg_list:
        query_params["lrg_filter"] = ",".join(lrg_list)

    items, error = make_api_request(base_url, query_params)
    if error:
        return {"error": f"Error fetching dif details: {error}"}
    all_items.extend(items.get("items") or [])

    if not all_items:
        filters_desc = []
        if lrg_list:
            filters_desc.append(f"LRGs: {', '.join(lrg_list)}")
        for k, v in query_params.items():
            if v:
                filters_desc.append(f"{k}: {v}")
        filter_str = "; ".join(filters_desc) if filters_desc else "no filters"
        return {
            "dif_details": [],
            "label": label,
            "filters_applied": filter_str,
            "message": f"No dif details found for `{label}` with given filters",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in all_items:
        # Create clean item with only meaningful data
        clean_item = {}
        if get_field_value(item, "lrg"):
            clean_item["lrg"] = get_field_value(item, "lrg")
        if get_field_value(item, "name"):
            clean_item["name"] = get_field_value(item, "name")
        if get_field_value(item, "rti_number"):
            clean_item["rti_number"] = get_field_value(item, "rti_number")
        if get_field_value(item, "rti_assigned_to"):
            clean_item["rti_assigned_to"] = get_field_value(
                item, "rti_assigned_to"
            )
        if get_field_value(item, "rti_status"):
            clean_item["rti_status"] = get_field_value(item, "rti_status")
        if get_field_value(item, "text"):
            clean_item["text"] = get_field_value(item, "text")
        if get_field_value(item, "comments"):
            clean_item["comments"] = clean_html_comments(
                get_field_value(item, "comments")
            )

        if clean_item:
            cleaned_items.append(clean_item)

    filters_desc = []
    if lrg_list:
        filters_desc.append(f"LRGs: {', '.join(lrg_list)}")
    for k, v in query_params.items():
        if v:
            filters_desc.append(f"{k}: {v}")
    filter_str = "; ".join(filters_desc) if filters_desc else "none"

    return {
        "dif_details": cleaned_items,
        "label": label,
        "filters_applied": filter_str,
        "count": len(cleaned_items),
    }


def clean_html_comments(comments_raw):
    """Clean HTML tags from comments and format properly"""
    if not comments_raw:
        return "No comments available."

    comments = str(comments_raw)
    # Replace <br> tags with newlines
    comments = re.sub(r"<br\s*/?>", "\n", comments)
    # Remove all HTML tags
    comments = re.sub(r"<[^>]+>", "", comments).strip()
    return comments or "No comments available."


@app.tool()
def find_dif_occurrence(dif: str, series: str) -> dict:
    """
    Find occurrences of a dif in the given series.

    Parameters:
    - dif (str): The dif name to search for
    - series (str): The series name (e.g., 'OSS_MAIN', 'OSS_25.1')

    Returns:
    dict: {
        "dif_occurrences": [
            {
                "label": "OSS_MAIN_LINUX.X64_250929",
                "lrg": "lrgsample",
                "name": "dif_name",
                "rti_number": "RTI-123",
                "rti_assigned_to": "user",
                "comments": "cleaned comments"
            }, ...
        ],
        "dif": "dif_name",
        "series": "OSS_MAIN",
        "count": 10
    } or {"error": "error message"}
    """
    # Make API request
    url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/diff_tracker"
    )
    params = {"dif": dif, "series": series}
    data, error = make_api_request(url, params)
    if error:
        return {
            "error": f"Error fetching occurrences for `{dif}` in `{series}`: {error}"
        }

    items = data.get("items") or []
    if not items:
        return {
            "dif_occurrences": [],
            "dif": dif,
            "series": series,
            "message": f"No occurrences found for `{dif}` in `{series}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        # Clean comments
        comments = clean_html_comments(item.get("comments", ""))

        # Create clean item with only meaningful data
        clean_item = {}
        if get_field_value(item, "label"):
            clean_item["label"] = get_field_value(item, "label")
        if get_field_value(item, "lrg"):
            clean_item["lrg"] = get_field_value(item, "lrg")
        if get_field_value(item, "name"):
            clean_item["name"] = get_field_value(item, "name")
        if get_field_value(item, "text"):
            clean_item["text"] = get_field_value(item, "text")
        if get_field_value(item, "rti_number"):
            clean_item["rti_number"] = get_field_value(item, "rti_number")
        if get_field_value(item, "rti_assigned_to"):
            clean_item["rti_assigned_to"] = get_field_value(
                item, "rti_assigned_to"
            )
        if comments and comments != "No comments available.":
            clean_item["comments"] = comments

        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "dif_occurrences": cleaned_items,
        "dif": dif,
        "series": series,
        "count": len(cleaned_items),
    }


@app.tool()
def find_widespread_issues(label: str, n: int = 3) -> dict:
    """
    Get widespread issues for a specific label.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')
    - n (int, optional): Minimum number of occurrences to be considered widespread (default: 3)

    Returns:
    dict: {
        "widespread_issues": [
            {
                "name": "dif_name",
                "lrgs": "lrg1,lrg2,lrg3"
            }, ...
        ],
        "label": "OSS_MAIN_LINUX.X64_250929",
        "count": 5
    } or {"error": "error message"}
    """
    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request
    url = f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/widespread_issue"
    params = {"label": label, "n": n}
    data, error = make_api_request(url, params)
    if error:
        return {
            "error": f"Error fetching widespread issues for `{label}`: {error}"
        }

    items = data.get("items") or []
    if not items:
        return {
            "widespread_issues": [],
            "label": label,
            "message": f"No widespread issues found for `{label}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        # Create clean item with only meaningful data
        clean_item = {}
        if get_field_value(item, "name"):
            clean_item["name"] = get_field_value(item, "name")
        if get_field_value(item, "lrgs"):
            clean_item["lrgs"] = get_field_value(item, "lrgs")

        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "widespread_issues": cleaned_items,
        "label": label,
        "count": len(cleaned_items),
    }


@app.tool()
def query_ai_crash_summary(label: str, lrg: str, dif_name: str) -> dict:
    """
    Get AI-generated crash summary for a specific crash.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')
    - lrg (str): The LRG identifier
    - dif_name (str): The dif name

    Returns:
    dict: {
        "ai_crash_summary": [
            {
                "ai_generated_info": "AI summary text"
            }
        ],
        "label": "OSS_MAIN_LINUX.X64_250929",
        "lrg": "lrgsample",
        "dif_name": "dif_name"
    } or {"error": "error message"}
    """
    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request
    url = f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/ai_generated_crash_summary"
    params = {"label": label, "lrg": lrg, "dif": dif_name}
    data, error = make_api_request(url, params)
    if error:
        return {
            "error": f"Error fetching AI crash summary for `{label}`: {error}"
        }

    items = data.get("items") or []
    if not items:
        return {
            "ai_crash_summary": [],
            "label": label,
            "lrg": lrg,
            "dif_name": dif_name,
            "message": f"No AI crash summary found for `{label}`, LRG `{lrg}`, dif `{dif_name}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        if get_field_value(item, "ai_generated_info"):
            cleaned_items.append(
                {
                    "ai_generated_info": get_field_value(
                        item, "ai_generated_info"
                    )
                }
            )

    return {
        "ai_crash_summary": cleaned_items,
        "label": label,
        "lrg": lrg,
        "dif_name": dif_name,
    }


@app.tool()
def get_se_rerun_details(label: str, se_job_id: Optional[str] = None) -> dict:
    """
    Get SE rerun details for a specific label or SE job ID.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')
    - se_job_id (str, optional): The SE job ID (7-9 digit string like '39586032')

    Returns:
    dict: {
        "se_rerun_details": [
            {
                "se_job_id": "39586032",
                "lrg": "lrgsample",
                "dif": "dif_name",
                "dif_type": "type",
                "no_of_reruns": 5,
                "fj_id_1": "123456",
                "fj_status_1": "PASSED",
                "fj_id_2": "123457",
                "fj_status_2": "FAILED",
                "se_notes": "notes"
            }, ...
        ],
        "label": "OSS_MAIN_LINUX.X64_250929",
        "count": 5
    } or {"error": "error message"}
    """
    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request
    url = f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/se_auto_analysis_data"
    params = {"label": label}
    if se_job_id:
        params["se_job_id"] = se_job_id

    data, error = make_api_request(url, params)
    if error:
        return {
            "error": f"Error fetching SE rerun details for `{label}`: {error}"
        }

    items = data.get("items") or []
    if not items:
        return {
            "se_rerun_details": [],
            "label": label,
            "message": f"No SE rerun details found for `{label}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        # Create clean item with only meaningful data
        clean_item = {}
        if get_field_value(item, "se_job_id"):
            clean_item["se_job_id"] = get_field_value(item, "se_job_id")
        if get_field_value(item, "lrg"):
            clean_item["lrg"] = get_field_value(item, "lrg")
        if get_field_value(item, "dif"):
            clean_item["dif"] = get_field_value(item, "dif")
        if get_field_value(item, "dif_type"):
            clean_item["dif_type"] = get_field_value(item, "dif_type")
        if get_field_value(item, "no_of_reruns"):
            clean_item["no_of_reruns"] = get_field_value(item, "no_of_reruns")
        if get_field_value(item, "fj_id_1"):
            clean_item["fj_id_1"] = get_field_value(item, "fj_id_1")
        if get_field_value(item, "fj_status_1"):
            clean_item["fj_status_1"] = get_field_value(item, "fj_status_1")
        if get_field_value(item, "fj_id_2"):
            clean_item["fj_id_2"] = get_field_value(item, "fj_id_2")
        if get_field_value(item, "fj_status_2"):
            clean_item["fj_status_2"] = get_field_value(item, "fj_status_2")
        if get_field_value(item, "se_notes"):
            clean_item["se_notes"] = get_field_value(item, "se_notes")

        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "se_rerun_details": cleaned_items,
        "label": label,
        "count": len(cleaned_items),
    }


@app.tool()
def get_regress_summary(regress: str, series: str) -> dict:
    """
    Get regress summary for a specific regress and series.

    Parameters:
    - regress (str): The regress name (e.g., 'SAGE_FC', 'EXAC_REGRESS')
    - series (str): The series name (e.g., 'OSS_MAIN', 'OSS_25.1')

    Returns:
    dict: {
        "regress_summary": [
            {
                "lrg": "lrgsample",
                "name": "dif_name",
                "series": "OSS_MAIN",
                "rti_numbers": "RTI-123,RTI-456",
                "latest_comment": "comment text"
            }, ...
        ],
        "regress": "SAGE_FC",
        "series": "OSS_MAIN",
        "count": 10
    } or {"error": "error message"}
    """
    # Validate that the regress exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/suite/{regress}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {
            "error": f"Error validating regress `{regress}`: {validate_error}"
        }

    # Check if regress exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {"error": f"Regress `{regress}` does not exist or has no LRGs."}

    # Make API request
    url = "https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/weekly_regress_summary"
    params = {"regress": regress, "series": series}
    data, error = make_api_request(url, params)
    if error:
        return {
            "error": f"Error fetching regress summary for `{regress}` in `{series}`: {error}"
        }

    items = data.get("items") or []
    if not items:
        return {
            "regress_summary": [],
            "regress": regress,
            "series": series,
            "message": f"No regress summary found for `{regress}` in `{series}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        # Create clean item with only meaningful data
        clean_item = {}
        if get_field_value(item, "lrg"):
            clean_item["lrg"] = get_field_value(item, "lrg")
        if get_field_value(item, "name"):
            clean_item["name"] = get_field_value(item, "name")
        if get_field_value(item, "series"):
            clean_item["series"] = get_field_value(item, "series")
        if get_field_value(item, "rti_numbers"):
            clean_item["rti_numbers"] = get_field_value(item, "rti_numbers")
        if get_field_value(item, "latest_comment"):
            clean_item["latest_comment"] = get_field_value(
                item, "latest_comment"
            )

        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "regress_summary": cleaned_items,
        "regress": regress,
        "series": series,
        "count": len(cleaned_items),
    }


@app.tool()
def get_label_info(label: str) -> dict:
    """
    Get detailed information about a specific label.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')

    Returns:
    dict: {
        "label_info": [
            {
                "field_name": "field_value",
                ...
            }
        ],
        "label": "OSS_MAIN_LINUX.X64_250929"
    } or {"error": "error message"}
    """
    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request
    url = f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    data, error = make_api_request(url)
    if error:
        return {"error": f"Error fetching label info for `{label}`: {error}"}

    items = data.get("items") or []
    if not items:
        return {
            "label_info": [],
            "label": label,
            "message": f"No information found for label `{label}`",
        }

    # Clean up items and format response - show all fields that exist
    cleaned_items = []
    for item in items:
        clean_item = {}
        for key, value in item.items():
            if value is not None and str(value).strip():
                clean_item[key] = str(value).strip()
        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "label_info": cleaned_items,
        "label": label,
    }


@app.tool()
def get_ai_label_summary(label: str) -> dict:
    """
    Get AI-generated summary for a specific label.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')

    Returns:
    dict: {
        "ai_label_summary": [
            {
                "label": "OSS_MAIN_LINUX.X64_250929",
                "ai_summary": "AI summary text"
            }
        ],
        "label": "OSS_MAIN_LINUX.X64_250929"
    } or {"error": "error message"}
    """
    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request
    url = "https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/ai_label_summary"
    params = {"label": label}
    data, error = make_api_request(url, params)
    if error:
        return {
            "error": f"Error fetching AI label summary for `{label}`: {error}"
        }

    items = data.get("items") or []
    if not items:
        return {
            "ai_label_summary": [],
            "label": label,
            "message": f"No AI summary found for label `{label}`. Consider generating one.",
        }

    # Clean up items and format response - extract label and ai_summary
    cleaned_items = []
    for item in items:
        clean_item = {}
        if get_field_value(item, "label"):
            clean_item["label"] = get_field_value(item, "label")
        if get_field_value(item, "ai_summary"):
            clean_item["ai_summary"] = get_field_value(item, "ai_summary")

        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "ai_label_summary": cleaned_items,
        "label": label,
    }


@app.tool()
def generate_ai_label_summary(label: str) -> dict:
    """
    Generate AI-generated summary for a specific label.

    Parameters:
    - label (str): The label name (e.g., 'OSS_MAIN_LINUX.X64_250929')

    Returns:
    dict: {
        "generation_result": "Generation result message",
        "label": "OSS_MAIN_LINUX.X64_250929"
    } or {"error": "error message"}
    """
    # Validate that the label exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {"error": f"Error validating label `{label}`: {validate_error}"}

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request to LH API Server to generate Label Summary
    url = f"https://phoenix518455.dev3sub2phx.databasede3phx.oraclevcn.com:8000/label_health/ai_label_summary/{label}"
    try:
        response = requests.get(url, timeout=300)  # 5 minute timeout
        if response.ok:
            return {
                "generation_result": response.text,
                "label": label,
            }
        else:
            return {
                "error": f"Failed to generate AI summary. API returned: {response.text}",
                "label": label,
            }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Error generating AI summary: {e}",
            "label": label,
        }


@app.tool()
def get_test_info(test: str) -> dict:
    """
    Get information about a test using FTS query.

    Parameters:
    - test (str): The test name

    Returns:
    dict: {
        "test": "test_name",
        "info": "FTS search results..."
    } or {"error": "error message"}
    """
    test_value = test

    # Run subprocess
    command = [
        "python3",
        "query_fts_refined.py",
        "--db",
        "/home/kbaboota/scripts/label_health_copilot/shray/profiles_fts.db",
        "--q",
        test_value,
        "--l2t",
        "/home/kbaboota/scripts/label_health_copilot/shray/jsons/lrg_to_tests.json",
        "--doc-type",
        "TEST",
        "--k",
        "5",
        "--debug",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd="/home/kbaboota/scripts/label_health_copilot/shray",
        )
        if result.returncode == 0:
            return {"test": test_value, "info": result.stdout}
        else:
            return {"error": f"Error: {result.stderr}"}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


@app.tool()
def find_tests_with_setup_and_flag(setup: str, flag: str) -> dict:
    """
    Find tests/LRGs with specific setups and flags using FTS query.

    Parameters:
    - setup (str): The setup name
    - flag (str): The flag value

    Returns:
    dict: {
        "setup": "setup_name",
        "flag": "flag_value",
        "info": "FTS search results..."
    } or {"error": "error message"}
    """
    setup_value = setup
    flag_value = flag

    # Run subprocess - construct the query from setup and flag
    query = f"{setup_value} {flag_value}"
    command = [
        "python3",
        "query_fts_refined.py",
        "--db",
        "/home/kbaboota/scripts/label_health_copilot/shray/profiles_fts.db",
        "--q",
        query,
        "--t2l",
        "/home/kbaboota/scripts/label_health_copilot/shray/jsons/test_to_lrgs.json",
        "--l2t",
        "/home/kbaboota/scripts/label_health_copilot/shray/jsons/lrg_to_tests.json",
        "--lrgs-json",
        "/home/kbaboota/scripts/label_health_copilot/shray/jsons/lrg_map_with_runtimes.json",
        "--require-setup",
        setup_value,
        "--require-flag",
        flag_value,
        "--k",
        "20",
        "--debug",
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd="/home/kbaboota/scripts/label_health_copilot/shray",
        )
        if result.returncode == 0:
            return {
                "setup": setup_value,
                "flag": flag_value,
                "info": result.stdout,
            }
        else:
            return {"error": f"Error: {result.stderr}"}
    except Exception as e:
        return {"error": f"Exception: {str(e)}"}


@app.tool()
def get_lrg_history(
    lrg: str, series: Optional[str] = None, n: int = 20
) -> dict:
    """
    Get LRG history for a given LRG, optionally filtered by series and number of labels.

    Parameters:
    - lrg (str): The LRG identifier
    - series (str, optional): The series name to filter by
    - n (int, optional): Number of labels for history (default: 10)

    Returns:
    dict: {
        "lrg_history": [
            {
                "field_name": "field_value",
                ...
            }, ...
        ],
        "lrg": "lrgsample",
        "series": "OSS_MAIN",
        "count": 20
    } or {"error": "error message"}
    """
    # Make API request
    url = "https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/lrg_health"
    params = {"lrg": lrg}
    if series:
        params["series"] = series
    if n:
        params["n"] = n

    data, error = make_api_request(url, params)
    if error:
        return {"error": f"Error fetching LRG history for `{lrg}`: {error}"}

    items = data.get("items") or []
    if not items:
        return {
            "lrg_history": [],
            "lrg": lrg,
            "message": f"No LRG history found for `{lrg}`",
        }

    # Clean up items and format response
    cleaned_items = []
    for item in items:
        clean_item = {}
        for key, value in item.items():
            if value is not None and str(value).strip():
                clean_item[key] = str(value).strip()
        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "lrg_history": cleaned_items,
        "lrg": lrg,
        "series": series,
        "count": len(cleaned_items),
    }


@app.tool()
def get_delta_diffs_between_labels(
    label_1: str, compare_labels: str, show_common: str
) -> dict:
    """
    Get delta diffs between labels.

    Parameters:
    - label_1 (str): The source label name (e.g., 'OSS_MAIN_LINUX.X64_250929')
    - compare_labels (str): Comma-separated list of labels to compare against
    - show_common (str): Whether to show common diffs ('Y' or 'N')

    Returns:
    dict: {
        "delta_diffs": [
            {
                "lrg": "lrgsample",
                "name": "dif_name",
                "text": "dif description",
                "rti_number": "RTI-123",
                "rti_assigned_to": "user"
            }, ...
        ],
        "label_1": "OSS_MAIN_LINUX.X64_250929",
        "compare_labels": "OSS_MAIN_LINUX.X64_250928,OSS_MAIN_LINUX.X64_250927",
        "show_common": "Y",
        "count": 10
    } or {"error": "error message"}
    """
    # Validate that the label_1 exists
    validate_url = (
        f"https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/{label_1}"
    )
    validate_data, validate_error = make_api_request(validate_url)
    if validate_error:
        return {
            "error": f"Error validating label `{label_1}`: {validate_error}"
        }

    # Check if label exists (count should be > 0)
    if validate_data.get("count", 0) == 0:
        return {
            "error": f"Label `{label_1}` is invalid, not loaded, or deleted from the dataset."
        }

    # Make API request
    url = "https://apex.oraclecorp.com/pls/apex/lrg_times/sucs_difs/delta_label_difs"
    params = {
        "label_1": label_1,
        "compare_labels": compare_labels,
        "show_common": show_common,
    }
    data, error = make_api_request(url, params)
    if error:
        return {"error": f"Error fetching delta diffs for `{label_1}`: {error}"}

    items = data.get("items") or []
    if not items:
        return {
            "delta_diffs": [],
            "label_1": label_1,
            "compare_labels": compare_labels,
            "show_common": show_common,
            "message": f"No delta diffs found for `{label_1}` with given compare_labels",
        }

    # Clean up items and format response - return only specified fields
    cleaned_items = []
    for item in items:
        # Create clean item with only the specified fields
        clean_item = {}
        if get_field_value(item, "lrg"):
            clean_item["lrg"] = get_field_value(item, "lrg")
        if get_field_value(item, "name"):
            clean_item["name"] = get_field_value(item, "name")
        if get_field_value(item, "text"):
            clean_item["text"] = get_field_value(item, "text")
        if get_field_value(item, "rti_number"):
            clean_item["rti_number"] = get_field_value(item, "rti_number")
        if get_field_value(item, "rti_assigned_to"):
            clean_item["rti_assigned_to"] = get_field_value(
                item, "rti_assigned_to"
            )

        if clean_item:
            cleaned_items.append(clean_item)

    return {
        "delta_diffs": cleaned_items,
        "label_1": label_1,
        "compare_labels": compare_labels,
        "show_common": show_common,
        "count": len(cleaned_items),
    }


if __name__ == "__main__":
    app.run()
    # print(get_labels_from_series("OSS_MAIN", 5))
