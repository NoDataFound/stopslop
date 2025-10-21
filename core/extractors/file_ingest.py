from __future__ import annotations
from typing import Tuple
import filetype
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import re

def _read_txt(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        return str(best)
    except Exception:
        return raw.decode("utf-8", errors="ignore")

def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_file(path: str, max_pages: int = 50) -> Tuple[str, dict]:
    kind = filetype.guess(path)
    meta = {"source": path}
    if kind is None:
        if path.lower().endswith((".md", ".txt", ".log", ".ioc")):
            return _read_txt(path), meta
        if path.lower().endswith((".html", ".htm")):
            return _html_to_text(_read_txt(path)), meta
        raise ValueError("Unknown file type")
    mime = kind.mime or ""
    if mime == "application/pdf":
        reader = PdfReader(path)
        pages = min(len(reader.pages), max_pages)
        text = " ".join([reader.pages[i].extract_text() or "" for i in range(pages)])
        return text, meta
    if mime in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
        d = Document(path)
        text = " ".join(p.text for p in d.paragraphs)
        return text, meta
    if mime.startswith("text/"):
        return _read_txt(path), meta
    if mime in ("text/html", "application/xhtml+xml"):
        return _html_to_text(_read_txt(path)), meta
    raise ValueError(f"Unsupported mime {mime}")
