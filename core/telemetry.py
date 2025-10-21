from __future__ import annotations
import os, json, time
from typing import Dict, Any, List

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "history.jsonl")

def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)

def append_log(entry: Dict[str, Any]):
    ensure_log_dir()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def read_logs() -> List[Dict[str, Any]]:
    ensure_log_dir()
    if not os.path.exists(LOG_FILE):
        return []
    out: List[Dict[str, Any]] = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out
