#!/usr/bin/env python3
"""Weekly feedback & MCP usage report."""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse a narrow subset of ISO timestamps into an aware UTC datetime."""
    if not ts:
        return None
    ts = ts.strip()
    if not ts:
        return None

    tz_segment = None
    body = ts
    if body.endswith("Z"):
        tz_segment = "+00:00"
        body = body[:-1]
    elif len(body) >= 6 and body[-6] in "+-" and body[-3] == ":":
        tz_segment = body[-6:]
        body = body[:-6]
    elif len(body) >= 5 and body[-5] in "+-":
        tz_segment = body[-5:]
        body = body[:-5]

    formats = (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    )

    parsed = None
    for fmt in formats:
        try:
            parsed = datetime.strptime(body, fmt)
            break
        except ValueError:
            continue
    if not parsed:
        return None

    if tz_segment:
        tz_clean = tz_segment
        if len(tz_clean) == 6 and tz_clean[-3] == ":":
            tz_clean = tz_clean[:3] + tz_clean[4:]
        sign = 1 if tz_clean[0] == "+" else -1
        hours = int(tz_clean[1:3] or 0)
        minutes = int(tz_clean[3:5] or 0) if len(tz_clean) >= 5 else 0
        offset = timezone(sign * timedelta(hours=hours, minutes=minutes))
    else:
        offset = timezone.utc

    parsed = parsed.replace(tzinfo=offset)
    return parsed.astimezone(timezone.utc)


def _iter_jsonl(path: str) -> Iterable[dict]:
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _default_week(anchor: Optional[date] = None) -> Tuple[date, date]:
    """Return (start, end) covering last Saturday through this Friday."""
    anchor = anchor or datetime.utcnow().date()
    weekday = anchor.weekday()  # Monday=0
    days_since_sat = (weekday - 5) % 7
    if days_since_sat == 0:
        days_since_sat = 7
    start = anchor - timedelta(days=days_since_sat)
    end = start + timedelta(days=6)
    return start, end


def _to_utc_bounds(start: date, end: date) -> Tuple[datetime, datetime]:
    start_dt = datetime.combine(start, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return start_dt, end_dt


def _normalize_question(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = " ".join(text.strip().split())
    return cleaned.lower().rstrip("?!. ")


class FeedbackRecord(object):
    __slots__ = (
        "uuid",
        "ts",
        "user",
        "feature",
        "tools",
        "question",
        "original_text",
        "thumb",
        "comments",
    )

    def __init__(self, uuid: str):
        self.uuid = uuid
        self.ts = None  # type: Optional[datetime]
        self.user = None  # type: Optional[str]
        self.feature = None  # type: Optional[str]
        self.tools = set()
        self.question = None  # type: Optional[str]
        self.original_text = None  # type: Optional[str]
        self.thumb = None  # type: Optional[str]
        self.comments = []  # type: List[dict]

    def as_dict(self) -> Dict[str, object]:
        return {
            "uuid": self.uuid,
            "ts": self.ts.isoformat() if self.ts else None,
            "user": self.user,
            "feature": self.feature,
            "tools": sorted(self.tools),
            "question": self.question,
            "original_text": self.original_text,
            "thumb": self.thumb,
            "comments": self.comments,
        }


class FeedbackAggregator:
    def __init__(self):
        self.records: Dict[str, FeedbackRecord] = {}

    def ingest(self, payload: dict):
        uuid = payload.get("uuid")
        if not uuid:
            return
        record = self.records.setdefault(uuid, FeedbackRecord(uuid=uuid))

        ts = _parse_iso(payload.get("ts"))
        if ts and (record.ts is None or ts < record.ts):
            record.ts = ts

        user = payload.get("user") or (payload.get("context") or {}).get("user")
        if user:
            record.user = user

        context = payload.get("context") or {}
        feature = context.get("feature")
        if feature:
            record.feature = feature

        tools = context.get("tools") or context.get("tool")
        if isinstance(tools, str):
            record.tools.add(tools)
        elif isinstance(tools, Iterable):
            for tool in tools:
                if tool:
                    record.tools.add(str(tool))

        question = context.get("question")
        if isinstance(question, str) and question.strip():
            record.question = question.strip()

        original = payload.get("original_text")
        if isinstance(original, str) and original.strip():
            record.original_text = original.strip()

        vote = payload.get("vote")
        event = payload.get("event")
        thumb: Optional[str] = None
        if vote in {"up", "down"}:
            thumb = vote
        elif event in {"thumb_up", "thumb_down"}:
            thumb = "up" if event == "thumb_up" else "down"
        if thumb:
            record.thumb = thumb

        comment = payload.get("comment")
        if isinstance(comment, str) and comment.strip():
            record.comments.append({
                "ts": ts.isoformat() if ts else None,
                "user": user,
                "comment": comment.strip(),
                "channel": payload.get("channel") or payload.get("slack_channel"),
                "thread_ts": payload.get("thread_ts") or payload.get("slack_msg_ts"),
            })


def _summarize_feedback(records: List[FeedbackRecord], window_start: datetime, window_end: datetime) -> Dict[str, object]:
    calls = [r for r in records if r.ts and window_start <= r.ts < window_end]
    calls_per_user = Counter((r.user or "unknown") for r in calls)
    calls_per_tool = Counter()
    for record in calls:
        if record.tools:
            calls_per_tool.update(record.tools)
        elif record.feature:
            calls_per_tool.update([record.feature])

    thumbs_up = sum(1 for r in calls if r.thumb == "up")
    thumbs_down = sum(1 for r in calls if r.thumb == "down")

    comment_rows: List[dict] = []
    for record in calls:
        for comment in record.comments:
            ts = _parse_iso(comment.get("ts"))
            if ts and not (window_start <= ts < window_end):
                continue
            comment_rows.append({
                "ts": ts.isoformat() if ts else None,
                "user": comment.get("user") or record.user,
                "comment": comment.get("comment"),
                "uuid": record.uuid,
            })

    question_counter = Counter()
    exemplars: Dict[str, str] = {}
    for record in calls:
        norm = _normalize_question(record.question)
        if not norm:
            continue
        question_counter[norm] += 1
        exemplars.setdefault(norm, record.question)

    top_questions = [
        {"question": exemplars[key], "count": count}
        for key, count in question_counter.most_common()
    ]

    return {
        "calls": calls,
        "calls_per_user": calls_per_user,
        "calls_per_tool": calls_per_tool,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "comments": sorted(comment_rows, key=lambda row: row.get("ts") or ""),
        "top_questions": top_questions,
    }


def _count_mcp_calls(path: str, window_start: datetime, window_end: datetime) -> Tuple[int, Counter]:
    total = 0
    per_tool = Counter()
    for event in _iter_jsonl(path):
        ts = _parse_iso(event.get("ts"))
        if not ts or not (window_start <= ts < window_end):
            continue
        total += 1
        tool = event.get("tool") or "unknown"
        per_tool[tool] += 1
    return total, per_tool


def _format_section(title: str, rows: List[str]) -> str:
    if not rows:
        return f"{title}\n  (none)"
    body = "\n".join(f"  - {row}" for row in rows)
    return f"{title}\n{body}"


def render_text(summary: Dict[str, object], start: date, end: date, mcp_total: int) -> str:
    lines = [
        f"Feedback window: {start.isoformat()} → {end.isoformat()} (UTC)",
        "",
        "Totals",
        f"  - MCP calls: {mcp_total}",
        f"  - Unique feedback threads: {len(summary['calls'])}",
        f"  - Thumbs up: {summary['thumbs_up']}",
        f"  - Thumbs down: {summary['thumbs_down']}",
    ]

    user_rows = [
        f"{user}: {count}"
        for user, count in summary["calls_per_user"].most_common()
    ]
    lines.append("")
    lines.append(_format_section("Calls per user", user_rows))

    tool_rows = [
        f"{tool}: {count}"
        for tool, count in summary["calls_per_tool"].most_common()
    ]
    lines.append("")
    lines.append(_format_section("Calls per tool", tool_rows))

    comment_rows = []
    for comment in summary["comments"]:
        stamp = comment.get("ts") or "unknown ts"
        comment_rows.append(f"{stamp} – {comment.get('user') or 'unknown'}: {comment.get('comment')}")
    lines.append("")
    lines.append(_format_section("Feedback comments", comment_rows))

    repeated = [item for item in summary["top_questions"] if item["count"] > 1]
    question_rows = [
        f"{item['question']} (x{item['count']})"
        for item in repeated[:10]
    ]
    if question_rows:
        lines.append("")
        lines.append(_format_section("Top repeated questions", question_rows))
    else:
        lines.append("")
        lines.append("Top repeated questions\n  No repeated question patterns this week.")

    return "\n".join(lines)


def _report_path(start_date: date, end_date: date, as_json: bool) -> str:
    directory = os.getenv("FEEDBACK_REPORT_DIR", os.path.join("metrics", "weekly_reports"))
    ext = "json" if as_json else "txt"
    filename = "feedback_report_%s_%s.%s" % (start_date.isoformat(), end_date.isoformat(), ext)
    return os.path.join(directory, filename)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Weekly feedback usage report")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--feedback-path", default="metrics/feedback.jsonl")
    parser.add_argument("--mcp-path", default="metrics/mcp_calls.jsonl")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text")
    parser.add_argument("--force", action="store_true", help="Regenerate even if a cached report exists")
    args = parser.parse_args(argv)

    if args.start and not args.end or args.end and not args.start:
        parser.error("--start and --end must be provided together")

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        start_date, end_date = _default_week()

    report_path = _report_path(start_date, end_date, args.json)

    if not args.force and os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as cached:
            sys.stdout.write(cached.read())
        return 0

    window_start, window_end = _to_utc_bounds(start_date, end_date)

    aggregator = FeedbackAggregator()
    for event in _iter_jsonl(args.feedback_path):
        aggregator.ingest(event)

    feedback_records = list(aggregator.records.values())
    summary = _summarize_feedback(feedback_records, window_start, window_end)
    mcp_total, _ = _count_mcp_calls(args.mcp_path, window_start, window_end)

    if args.json:
        payload = {
            "window": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "mcp_calls": mcp_total,
            "calls_per_user": dict(summary["calls_per_user"]),
            "calls_per_tool": dict(summary["calls_per_tool"]),
            "thumbs_up": summary["thumbs_up"],
            "thumbs_down": summary["thumbs_down"],
            "comments": summary["comments"],
            "top_questions": summary["top_questions"],
        }
        output = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        output = render_text(summary, start_date, end_date, mcp_total)

    directory = os.path.dirname(report_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(output)

    print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
