from __future__ import annotations
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from core.providers.base import ProviderBase

class OpenAIProvider(ProviderBase):
    name = "openai"

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def audit(self, content: str, rules: List[dict], system_prompt: str, model: str, timeout: int = 20) -> Dict[str, Any]:
        payload = self.build_prompt_payload(content, rules)
        resp = self.client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload}
            ],
            timeout=timeout
        )
        out = resp.choices[0].message.content
        return self.parse_json_only(out)
