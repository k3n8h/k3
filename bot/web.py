"""HTTP fetching, text extraction, robots.txt, search and crawling (no API keys required)."""
import ipaddress
import os
import re
import socket
import time
from collections import deque
from urllib.parse import parse_qs, urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

UA = "K3Bot/0.1 (+research assistant)"
MAX_BYTES = 2_000_000
_robots: dict[str, RobotFileParser | None] = {}


def allow_private() -> bool:
    return os.environ.get("BOT_ALLOW_PRIVATE_NETS") == "1"


def resolve_public(url: str) -> tuple[list[str], str]:
    """SSRF guard. Returns ([vetted ips], hostname); every resolved address must be public.

    fetch() connects only to these exact IPs, so the check and the connection can't disagree
    (no DNS rebinding).
    """
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("only http(s) URLs are allowed")
    infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    ips = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not allow_private() and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                                    or ip.is_multicast or ip.is_unspecified):
            raise ValueError(f"blocked: {p.hostname} resolves to a non-public address")
        if info[4][0] not in ips:
            ips.append(info[4][0])
    if not ips:
        raise ValueError(f"could not resolve {p.hostname}")
    return ips, p.hostname


def _get_pinned(url: str, ip: str, host: str) -> tuple[httpx.Response, bytes]:
    """One GET to `ip` with the original Host header and TLS SNI. A fresh client per attempt means no
    connection is ever pooled across hostnames; the environment's proxy settings are ignored on purpose
    (a proxy would route by the IP we pinned and defeat the check)."""
    p = urlparse(url)
    netloc = f"[{ip}]" if ":" in ip else ip
    if p.port:
        netloc += f":{p.port}"
    host_header = host if not p.port else f"{host}:{p.port}"
    with httpx.Client(timeout=20, trust_env=False) as c:
        with c.stream("GET", p._replace(netloc=netloc).geturl(),
                      headers={"User-Agent": UA, "Host": host_header}, extensions={"sni_hostname": host}) as r:
            if r.is_redirect:
                return r, b""
            r.raise_for_status()
            data = b""
            for chunk in r.iter_bytes():
                data += chunk
                if len(data) > MAX_BYTES:
                    break
            return r, data[:MAX_BYTES]


def fetch(url: str, max_redirects: int = 5) -> tuple[str, str, str]:
    """Return (final_url, content_type, text). Redirects are followed manually so each hop is re-checked,
    and each request goes only to addresses that passed the check (trying them in order)."""
    for _ in range(max_redirects + 1):
        ips, host = resolve_public(url)
        last: Exception | None = None
        for ip in ips:
            try:
                r, data = _get_pinned(url, ip, host)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:   # try the next vetted address
                last = e
        else:
            raise last or ValueError("connection failed")
        if r.is_redirect:
            url = urljoin(url, r.headers["location"])
            continue
        return url, r.headers.get("content-type", ""), data.decode(r.encoding or "utf-8", errors="replace")
    raise ValueError("too many redirects")


def soup_of(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def extract_text(html: str) -> tuple[str, str]:
    """(title, readable text)."""
    s = soup_of(html)
    title = s.title.get_text(strip=True) if s.title else ""
    for t in s(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        t.decompose()
    root = s.find("main") or s.find("article") or s.body or s
    return title, re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t]+", " ", root.get_text("\n"))).strip()


def extract_links(html: str, base: str) -> list[str]:
    out, seen = [], set()
    for a in soup_of(html).find_all("a", href=True):
        u = urldefrag(urljoin(base, a["href"]))[0]
        if urlparse(u).scheme in ("http", "https") and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def robots_allowed(url: str) -> bool:
    p = urlparse(url)
    origin = f"{p.scheme}://{p.netloc}"
    if origin not in _robots:
        rp = RobotFileParser()
        try:
            _, _, txt = fetch(origin + "/robots.txt")
            rp.parse(txt.splitlines())
            _robots[origin] = rp
        except Exception:
            _robots[origin] = None  # no robots.txt reachable -> allowed
    rp = _robots[origin]
    return True if rp is None else rp.can_fetch(UA, url)


def parse_ddg(html: str) -> list[dict]:
    out = []
    for a in soup_of(html).select("a.result__a"):
        href = a.get("href", "")
        q = parse_qs(urlparse(href).query)
        url = q["uddg"][0] if "uddg" in q else href
        snippet = a.find_parent(class_="result")
        sn = snippet.select_one(".result__snippet") if snippet else None
        if url.startswith("http"):
            out.append({"title": a.get_text(strip=True), "url": url, "snippet": sn.get_text(strip=True) if sn else ""})
    return out


def search(query: str, n: int = 5) -> list[dict]:
    with httpx.Client(timeout=20, headers={"User-Agent": UA}) as c:
        r = c.post("https://html.duckduckgo.com/html/", data={"q": query})
        r.raise_for_status()
    return parse_ddg(r.text)[:n]


def crawl(start_url: str, max_pages: int = 10, max_depth: int = 1, same_domain: bool = True,
          delay: float | None = None) -> list[dict]:
    """Breadth-first crawl honoring robots.txt, with a politeness delay."""
    delay = float(os.environ.get("BOT_CRAWL_DELAY", "1.0")) if delay is None else delay
    host = urlparse(start_url).netloc
    queue, seen, pages = deque([(start_url, 0)]), {start_url}, []
    while queue and len(pages) < max_pages:
        url, depth = queue.popleft()
        if not robots_allowed(url):
            pages.append({"url": url, "skipped": "disallowed by robots.txt"})
            continue
        try:
            final, ctype, body = fetch(url)
        except Exception as e:
            pages.append({"url": url, "error": str(e)})
            continue
        if "html" in ctype:
            title, text = extract_text(body)
            pages.append({"url": final, "depth": depth, "title": title, "text": text[:3000]})
            if depth < max_depth:
                for u in extract_links(body, final):
                    if u not in seen and (not same_domain or urlparse(u).netloc == host):
                        seen.add(u)
                        queue.append((u, depth + 1))
        if delay:
            time.sleep(delay)
    return pages
