from __future__ import annotations
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_fixed
import google.generativeai as genai
from core.providers.base import ProviderBase

class GeminiProvider(ProviderBase):
    name = "gemini"

    def __init__(self, api_key: str, model: str):
        genai.configure(api_key=api_key)
        self.model_name = model

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def audit(self, content: str, rules: List[dict], system_prompt: str, model: str | None = None, timeout: int = 20) -> Dict[str, Any]:
        payload = self.build_prompt_payload(content, rules)
        mdl = genai.GenerativeModel((model or self.model_name), system_instruction=system_prompt)
        resp = mdl.generate_content(payload, generation_config={"temperature": 0, "response_mime_type": "application/json"})
        out = resp.text
        return self.parse_json_only(out)
