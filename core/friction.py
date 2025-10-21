from __future__ import annotations
import json
import os
import re
from typing import List, Dict, Any, Set
from core.config import RuntimeConfig

def _load_policies(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_gold_apps(cfg: RuntimeConfig) -> Set[str]:
    apps: Set[str] = set()
    if cfg.gold_apps_env:
        for a in cfg.gold_apps_env.split(","):
            a = a.strip()
            if a:
                apps.add(a.lower())
    if cfg.gold_apps_file and os.path.exists(cfg.gold_apps_file):
        try:
            with open(cfg.gold_apps_file, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        apps.add(t.lower())
        except Exception:
            pass
    if not apps:
        defaults = ["salesforce","workday","active directory","okta","aws","azure","gcp","servicenow","sap","oracle ebs","github","gitlab","snowflake","databricks"]
        apps.update(defaults)
    return apps

def detect_risk_tags(text: str, cfg: RuntimeConfig) -> Set[str]:
    tags: Set[str] = set()
    t = text.lower()
    if re.search(r'\b(wire transfer|invoice|payment|bank|swift|routing number|account number|ach|wallet|btc|usdt|crypto|bill|vendor payment)\b', t):
        tags.add("financial")
    if re.search(r'\b(legal|subpoena|settlement|nda|contract|regulatory|compliance|gdpr|ccpa|sec|consent order)\b', t):
        tags.add("legal")
    if re.search(r'\b(production|rollback|outage|disable|edr|siem|firewall|dns|api key|token|sso|root|domain admin|iam|cluster|kubernetes|kubectl|helm|mfa|privilege)\b', t):
        tags.add("operations")
    gold = _load_gold_apps(cfg)
    for app in gold:
        if app in t:
            tags.add("gold_app")
            break
    return tags

def _rule_hits(findings: List[Dict[str, Any]]) -> Set[str]:
    return {f["rule_id"] for f in findings if f.get("hit")}

def generate_friction(text: str,
                      findings: List[Dict[str, Any]],
                      llm_results: List[Dict[str, Any]],
                      combined_score: float,
                      threshold: float,
                      policies_path: str,
                      cfg: RuntimeConfig) -> List[Dict[str, Any]]:
    policies = _load_policies(policies_path)
    rules_hit = _rule_hits(findings)
    tags = detect_risk_tags(text, cfg)
    slop = combined_score >= threshold
    borderline = abs(combined_score - threshold) <= 0.05
    out: List[Dict[str, Any]] = []
    for p in policies.get("frictions", []):
        when = p.get("when", "always")
        if when == "slop_only" and not slop:
            continue
        if when == "slop_or_borderline" and not (slop or borderline):
            continue
        trig = p.get("triggers", {})
        rid_any = set(trig.get("rule_ids_any", []))
        tag_any = set(trig.get("risk_tags_any", []))
        rid_match = True if not rid_any else bool(rid_any & rules_hit)
        tag_match = True if not tag_any else bool(tag_any & tags)
        if rid_match and tag_match:
            suggestion = {
                "id": p["id"],
                "title": p["title"],
                "severity": float(p.get("severity", 0.5)),
                "when": when,
                "triggers": {
                    "rules_hit": sorted(list(rules_hit & rid_any)) if rid_any else [],
                    "risk_tags": sorted(list(tags & tag_any)) if tag_any else []
                },
                "steps": p.get("steps", []),
                "rationale": p.get("rationale", "")
            }
            out.append(suggestion)
    out.sort(key=lambda x: x["severity"], reverse=True)
    return out
