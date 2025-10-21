from __future__ import annotations
from typing import List, Dict, Any
from core.utils.json_sanitize import safe_json_loads

class ProviderBase:
    name = "base"

    def audit(self, content: str, rules: List[dict], system_prompt: str, model: str, timeout: int = 20) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def build_prompt_payload(content: str, rules: List[dict]) -> str:
        rule_slim = [{"rule_id": r["rule_id"], "hit": r["hit"], "weight": r["weight"], "desc": r["description"]} for r in rules]
        return f"CONTENT:\n{content}\n\nLOCAL_RULES:\n{rule_slim}\n"

    @staticmethod
    def parse_json_only(output: str) -> Dict[str, Any]:
        return safe_json_loads(output)
