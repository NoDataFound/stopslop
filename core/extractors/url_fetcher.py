from __future__ import annotations
import ipaddress
import socket
import tldextract
from typing import Tuple
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re
import time

def _is_public_host(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, 80, proto=socket.IPPROTO_TCP)
    except Exception:
        return False
    for family, _, _, _, sockaddr in infos:
        ip = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_reserved or ip_obj.is_multicast:
                return False
        except Exception:
            return False
    return True

def _simple_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_url(url: str, timeout_sec: int = 20, block_private_ips: bool = True, use_selenium: bool = False) -> Tuple[str, dict]:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("Only http and https are allowed")
    ext = tldextract.extract(url)
    host = ext.registered_domain or ext.fqdn
    if block_private_ips and host and not _is_public_host(host):
        raise ValueError("Blocked non public host to avoid SSRF")
    headers = {"User-Agent": "SLOPwatch/1.0 (+https://local)"}
    r = requests.get(url, headers=headers, timeout=timeout_sec)
    r.raise_for_status()
    html = r.text
    art = Article(url)
    try:
        art.download(input_html=html)
        art.parse()
        text = art.text or ""
    except Exception:
        text = ""
    if len(text) < 500:
        text = _simple_html_to_text(html)
    meta = {"source": url, "fetched_at": int(time.time()), "bytes": len(html)}
    if use_selenium and len(text) < 1000:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from webdriver_manager.chrome import ChromeDriverManager
            opts = Options()
            opts.add_argument("--headless=new")
            opts.add_argument("--disable-gpu")
            opts.add_argument("--no-sandbox")
            opts.add_argument("--disable-dev-shm-usage")
            driver = webdriver.Chrome(ChromeDriverManager().install(), options=opts)
            driver.set_page_load_timeout(timeout_sec)
            driver.get(url)
            html2 = driver.page_source
            driver.quit()
            text2 = _simple_html_to_text(html2)
            if len(text2) > len(text):
                text = text2
                meta["bytes"] = len(html2)
                meta["rendered"] = True
        except Exception:
            pass
    return text, meta
