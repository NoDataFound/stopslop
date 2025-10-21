from __future__ import annotations
from typing import Dict, Any, List

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))

def _llm_evidence_score(llm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert LLM outputs to a continuous evidence signal.
    Ignore any 'is_slop' fields from providers.
    Score = mean_over_models( confidence * mean_severity_of_true_hits ).
    If a model gives reasons but no hits, give it weak advisory weight (conf * 0.5).
    """
    if not llm_results:
        return {"score": 0.0, "confidence": 0.5, "reasons": ["LLM results unavailable. Using rule based score only."]}

    scores: List[float] = []
    confs: List[float] = []
    reasons: List[str] = []

    for r in llm_results:
        conf = _clamp(float(r.get("confidence", 0.5)))
        confs.append(conf)

        findings = r.get("rule_findings") or []
        hits = [f for f in findings if f.get("hit") is True]
        if hits:
            sev_vals = [_clamp(float(f.get("severity", 0.0))) for f in hits]
            sev_mean = sum(sev_vals) / len(sev_vals)
            scores.append(conf * sev_mean)
        else:
            if r.get("overall_reason"):
                scores.append(conf * 0.5)
            else:
                scores.append(0.0)

        if r.get("overall_reason"):
            reasons.append(r["overall_reason"])

    score = sum(scores) / len(scores) if scores else 0.0
    avg_conf = sum(confs) / len(confs) if confs else 0.5
    return {"score": _clamp(score), "confidence": _clamp(avg_conf), "reasons": reasons}

def aggregate_score(local_score: float, llm_results: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
    rw = float(weights.get("rules_weight", 0.55))
    lw = float(weights.get("llm_weight", 0.45))

    llm = _llm_evidence_score(llm_results)
    combined = _clamp(rw * _clamp(local_score) + lw * _clamp(llm["score"]))
    confidence = _clamp(llm["confidence"] + 0.2)
    reasons = llm["reasons"]
    return {"combined_score": combined, "reasons": reasons, "confidence": confidence}

def final_decision(combined_score: float, threshold: float) -> bool:
    return float(combined_score) >= float(threshold)
