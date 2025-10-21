from __future__ import annotations
import json
import re
import math
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
    """Load rules and scoring configuration from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    rules = [Rule(**r) for r in cfg["rules"]]
    score_cfg = cfg.get("score", {"llm_weight": 0.45, "rules_weight": 0.55, "slop_threshold": 0.6})
    return {"rules": rules, "score": score_cfg, "version": cfg.get("version", "1.0")}

def eval_rule(text: str, rule: Rule) -> dict:
    """Evaluate a single rule against the text and return finding metadata."""
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
        # LLM-only rules are not evaluated locally
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
    """Run all rules on a text and return a list of findings."""
    return [eval_rule(text, r) for r in rules]

def _saturate(x: float, alpha: float = 0.7) -> float:
    """
    Map cumulative hit weights to a 0–1 score with diminishing returns.
    Prevents a few strong hits from being drowned out by total weight.
    """
    return 1.0 - math.exp(-alpha * max(0.0, x))

def rules_score(findings: List[dict]) -> float:
    """
    Compute the local rule score.
    Uses a saturation curve instead of naive normalization
    to give proportional influence to smaller hit sets.
    """
    hit_weight_sum = sum(f["weight"] for f in findings if f["hit"])
    return min(1.0, _saturate(hit_weight_sum, alpha=0.7))
