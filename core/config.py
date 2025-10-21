from __future__ import annotations
import os
from pydantic import BaseModel, Field

def _get_secret(name: str, default: str | None = None) -> str | None:
    try:
        import streamlit as st
        try:
            v = st.secrets.get(name)  # returns None if missing
        except Exception:
            v = None
        if v is not None:
            return str(v)
    except Exception:
        pass
    return os.getenv(name, default)

def _get_bool(name: str, default: bool) -> bool:
    v = _get_secret(name, None)
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}

def _get_int(name: str, default: int) -> int:
    v = _get_secret(name, None)
    if v is None:
        return default
    try:
        return int(str(v))
    except Exception:
        return default

class RuntimeConfig(BaseModel):
    max_chars: int = Field(default=_get_int("SLOPWATCH_MAX_CHARS", 200000))
    timeout_sec: int = Field(default=_get_int("SLOPWATCH_TIMEOUT_SEC", 20))
    enable_selenium: bool = Field(default=_get_bool("SLOPWATCH_ENABLE_SELENIUM", False))
    block_private_ips: bool = Field(default=_get_bool("SLOPWATCH_BLOCK_PRIVATE_IPS", True))

    openai_key: str | None = Field(default=_get_secret("OPENAI_API_KEY"))
    anthropic_key: str | None = Field(default=_get_secret("ANTHROPIC_API_KEY"))
    google_key: str | None = Field(default=_get_secret("GOOGLE_API_KEY"))

    default_models: dict = Field(default_factory=lambda: {
        "openai": _get_secret("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic": _get_secret("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
        "gemini": _get_secret("GEMINI_MODEL", "gemini-1.5-pro")
    })

    gold_apps_env: str | None = Field(default=_get_secret("SLOPWATCH_GOLD_APPS"))
    gold_apps_file: str | None = Field(default=_get_secret("SLOPWATCH_GOLD_APPS_FILE"))

    @property
    def providers_available(self) -> list[str]:
        out: list[str] = []
        if self.openai_key:
            out.append("openai")
        if self.anthropic_key:
            out.append("anthropic")
        if self.google_key:
            out.append("gemini")
        return out
