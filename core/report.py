from __future__ import annotations
import hashlib
import time
from typing import Dict, Any, List

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def build_report(content: str,
                 meta: dict,
                 rules_findings: List[dict],
                 rules_score: float,
                 llm_results: List[dict],
                 agg: dict,
                 threshold: float,
                 providers_used: List[str],
                 models_used: dict,
                 rules_version: str,
                 friction: List[dict]) -> dict:
    return {
        "schema": "slopwatch.report.v1",
        "created_at": int(time.time()),
        "content_sha256": sha256(content),
        "source_meta": meta,
        "providers_used": providers_used,
        "models_used": models_used,
        "rules_version": rules_version,
        "rules_findings": rules_findings,
        "rules_score": rules_score,
        "llm_results": llm_results,
        "combined_score": agg["combined_score"],
        "decision_slop": agg["combined_score"] >= threshold,
        "confidence": agg["confidence"],
        "threshold": threshold,
        "overall_reasons": agg["reasons"],
        "friction": friction
    }

def to_markdown(report: dict) -> str:
    lines: List[str] = []
    lines.append(f"# SLOPwatch report")
    lines.append(f"- Created: {report['created_at']}")
    lines.append(f"- Combined score: {report['combined_score']:.3f}")
    lines.append(f"- Decision slop: {report['decision_slop']}")
    lines.append(f"- Confidence: {report['confidence']:.2f}")
    lines.append("")
    lines.append("## Rule findings")
    for f in report["rules_findings"]:
        lines.append(f"- {f['rule_id']}: hit={f['hit']} weight={f['weight']} {f.get('details','')}")
    lines.append("")
    lines.append("## LLM results")
    for r in report["llm_results"]:
        lines.append(f"- is_slop={r.get('is_slop')} confidence={r.get('confidence')} reason={r.get('overall_reason')}")
    lines.append("")
    lines.append("## Reasons")
    for reason in report["overall_reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## Friction plan")
    if not report.get("friction"):
        lines.append("- No frictions suggested")
    else:
        for f in report["friction"]:
            lines.append(f"- [{f['severity']:.2f}] {f['title']}")
            if f.get("triggers"):
                trig = []
                rh = f["triggers"].get("rules_hit", [])
                if rh:
                    trig.append("rules: " + ",".join(rh))
                rt = f["triggers"].get("risk_tags", [])
                if rt:
                    trig.append("risk: " + ",".join(rt))
                if trig:
                    lines.append("  - triggers: " + " | ".join(trig))
            for step in f.get("steps", []):
                lines.append(f"  - {step}")
            if f.get("rationale"):
                lines.append(f"  - rationale: {f['rationale']}")
    return "\n".join(lines)
