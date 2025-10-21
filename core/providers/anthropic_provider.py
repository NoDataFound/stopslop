from __future__ import annotations
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_fixed
import anthropic
from core.providers.base import ProviderBase

class AnthropicProvider(ProviderBase):
    name = "anthropic"

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def audit(self, content: str, rules: List[dict], system_prompt: str, model: str, timeout: int = 20) -> Dict[str, Any]:
        payload = self.build_prompt_payload(content, rules)
        msg = self.client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0,
            system=system_prompt,
            messages=[{"role": "user", "content": payload}],
            timeout=timeout
        )
        out = "".join([blk.text for blk in msg.content if getattr(blk, "type", "") == "text"])
        return self.parse_json_only(out)
