from __future__ import annotations
from typing import Dict, Any, List

def aggregate_score(local_score: float, llm_results: List[Dict[str, Any]], weights: Dict[str, float]) -> Dict[str, Any]:
    if not llm_results:
        combined = local_score
        reasons = ["LLM results unavailable. Using rule based score only."]
        confidence = 0.5
    else:
        llm_scores = []
        reasons = []
        confs = []
        for r in llm_results:
            is_slop = 1.0 if r.get("is_slop") else 0.0
            conf = float(r.get("confidence", 0.5))
            llm_scores.append(is_slop * conf)
            if r.get("overall_reason"):
                reasons.append(r["overall_reason"])
            confs.append(conf)
        llm_score = sum(llm_scores) / len(llm_scores)
        combined = weights["rules_weight"] * local_score + weights["llm_weight"] * llm_score
        confidence = min(1.0, (sum(confs) / len(confs) + 0.2))
    return {"combined_score": combined, "reasons": reasons, "confidence": confidence}

def final_decision(combined_score: float, threshold: float) -> bool:
    return combined_score >= threshold
