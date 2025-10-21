from __future__ import annotations
from typing import Dict, Any, List, Tuple

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _summarize_hits_local(local_findings: List[Dict[str, Any]]) -> List[str]:
    reasons = []
    for f in local_findings:
        if f.get("hit"):
            rid = f.get("rule_id")
            desc = f.get("description") or ""
            reasons.append(f"[local:{rid}] {desc}")
    return reasons

def _llm_evidence_score_and_reasons(llm_results: List[Dict[str, Any]]) -> Tuple[float, float, List[str]]:
    if not llm_results:
        return 0.0, 0.5, []

    scores: List[float] = []
    confs: List[float] = []
    reasons: List[str] = []

    for r in llm_results:
        conf = _clamp(float(r.get("confidence", 0.5)))
        confs.append(conf)
        hits = [f for f in (r.get("rule_findings") or []) if f.get("hit") is True]
        if hits:
            sev_vals = [_clamp(float(f.get("severity", 0.0))) for f in hits]
            sev_mean = sum(sev_vals) / len(sev_vals)
            scores.append(conf * sev_mean)
            for f in hits:
                rid = f.get("rule_id")
                reason = f.get("reason") or ""
                reasons.append(f"[llm:{rid}] {reason}")
        else:
            scores.append(0.0)

    score = (sum(scores) / len(scores)) if scores else 0.0
    avg_conf = (sum(confs) / len(confs)) if confs else 0.5
    return _clamp(score), _clamp(avg_conf), reasons

def aggregate_score(local_score: float,
                    llm_results: List[Dict[str, Any]],
                    weights: Dict[str, float],
                    local_findings: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """
    Final combined score = rules_weight * local_score + llm_weight * llm_evidence_score.
    Reasons = union of local hit reasons (if provided) and llm hit reasons only.
    Confidence = llm confidence lightly boosted to reflect mixed-method.
    """
    rw = float(weights.get("rules_weight", 0.55))
    lw = float(weights.get("llm_weight", 0.45))

    llm_score, llm_conf, llm_reasons = _llm_evidence_score_and_reasons(llm_results)
    combined = _clamp(rw * _clamp(local_score) + lw * _clamp(llm_score))

    reasons: List[str] = []
    if local_findings:
        reasons.extend(_summarize_hits_local(local_findings))
    reasons.extend(llm_reasons)

    confidence = _clamp(llm_conf + 0.2)
    return {"combined_score": combined, "reasons": reasons, "confidence": confidence}

def final_decision(combined_score: float, threshold: float) -> bool:
    return float(combined_score) >= float(threshold)
