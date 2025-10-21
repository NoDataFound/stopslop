from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List

@dataclass
class Rule:
    id: str
    description: str
    weight: float
    type: str
    pattern: str | None = None
    negate: bool = False
    threshold: int | None = None

def load_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    rules = [Rule(**r) for r in cfg["rules"]]
    score_cfg = cfg.get("score", {"llm_weight": 0.45, "rules_weight": 0.55, "slop_threshold": 0.6})
    return {"rules": rules, "score": score_cfg, "version": cfg.get("version", "1.0")}

def eval_rule(text: str, rule: Rule) -> dict:
    hit = False
    details = ""
    if rule.type == "regex_any" and rule.pattern:
        hit = bool(re.search(rule.pattern, text, flags=re.I))
    elif rule.type == "regex_absent" and rule.pattern:
        hit = not bool(re.search(rule.pattern, text, flags=re.I))
    elif rule.type == "regex_count" and rule.pattern and rule.threshold is not None:
        cnt = len(re.findall(rule.pattern, text, flags=re.I))
        hit = cnt >= rule.threshold
        details = f"count={cnt}"
    elif rule.type == "llm_only":
        hit = False
    if rule.negate:
        hit = not hit
    return {
        "rule_id": rule.id,
        "hit": hit,
        "weight": rule.weight,
        "description": rule.description,
        "details": details
    }

def run_rules(text: str, rules: List[Rule]) -> List[dict]:
    return [eval_rule(text, r) for r in rules]

def rules_score(findings: List[dict]) -> float:
    total = sum(f["weight"] for f in findings if f["hit"])
    cap = 10.0
    return min(1.0, total / cap)
