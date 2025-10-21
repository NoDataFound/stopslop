import re

def normalize_text(s: str, max_chars: int) -> str:
    s = s.replace('\r', '\n')
    s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    if len(s) > max_chars:
        s = s[:max_chars]
    return s
