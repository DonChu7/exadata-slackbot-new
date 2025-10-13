# router.py
from typing import Dict, Any, List, Tuple, Optional
import re

class Router:
    def __init__(self):
        self.tools: List[Dict[str, Any]] = []        # [{service,name,intents,patterns,desc}]
        self.compiled: List[Tuple[re.Pattern, Dict[str,Any]]] = []

    def load_from_manifests(self, manifests: List[Dict[str, Any]]):
        self.tools.clear(); self.compiled.clear()
        for m in manifests or []:
            svc = m.get("service")
            for t in (m.get("tools") or []):
                entry = {
                    "service": svc,
                    "name": t.get("name"),
                    "intents": t.get("intents", []),
                    "patterns": t.get("patterns", []),
                    "desc": t.get("description", ""),
                }
                self.tools.append(entry)
                for pat in entry["patterns"]:
                    try:
                        self.compiled.append((re.compile(pat, re.I), entry))
                    except re.error:
                        pass

    def rule_route(self, text: str) -> Optional[Dict[str, Any]]:
        low = (text or "").lower()
        # intent keywords first (fast path)
        for entry in self.tools:
            if any(k in low for k in entry["intents"]):
                return {"tool": entry["name"], "service": entry["service"], "confidence": 0.8, "reason": "intent"}
        # regex patterns next
        for rx, entry in self.compiled:
            if rx.search(text or ""):
                return {"tool": entry["name"], "service": entry["service"], "confidence": 0.75, "reason": f"regex:{rx.pattern}"}
        return None
