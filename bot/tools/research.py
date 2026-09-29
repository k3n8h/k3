"""Research, scraping and crawling tools."""
import json

from bot import web
from bot.registry import tool
from bot.tools.code import safe_path


@tool("Web search (no API key). Returns title, url, snippet for the top results.")
def web_search(query: str, max_results: int = 5) -> list:
    return web.search(query, max_results)


@tool("Fetch a web page and return its title, readable text (truncated) and outgoing links.")
def web_fetch(url: str, max_chars: int = 6000) -> dict:
    final, ctype, body = web.fetch(url)
    if "html" not in ctype:
        return {"url": final, "content_type": ctype, "text": body[:max_chars]}
    title, text = web.extract_text(body)
    return {"url": final, "title": title, "text": text[:max_chars], "links": web.extract_links(body, final)[:40]}


@tool("Scrape a page: return the text (or attribute `attr`, e.g. href) of every element matching a CSS selector.")
def scrape(url: str, selector: str, attr: str = "", limit: int = 100) -> list:
    final, _, body = web.fetch(url)
    els = web.soup_of(body).select(selector)[:limit]
    return [e.get(attr) if attr else e.get_text(" ", strip=True) for e in els]


@tool("Crawl a site breadth-first (robots.txt respected, polite delay, same domain by default). "
      "Saves results to crawls/<name>.json in the workspace when `save_as` is given.")
def crawl(start_url: str, max_pages: int = 10, max_depth: int = 1, save_as: str = "") -> dict:
    max_pages, max_depth = min(max_pages, 50), min(max_depth, 3)
    pages = web.crawl(start_url, max_pages, max_depth)
    if save_as:
        p = safe_path(f"crawls/{save_as}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(pages, indent=2))
    return {"pages": len(pages), "results": [{k: (v[:300] if k == "text" else v) for k, v in pg.items()} for pg in pages]}


@tool("Research a question: search the web, read the top sources and return excerpts with URLs to cite. "
      "Also saves a markdown research file in the workspace.")
def research(question: str, max_sources: int = 3) -> dict:
    sources = []
    for r in web.search(question, max_sources + 2):
        if len(sources) >= max_sources:
            break
        try:
            final, ctype, body = web.fetch(r["url"])
            title, text = web.extract_text(body) if "html" in ctype else (r["title"], body)
            sources.append({"url": final, "title": title or r["title"], "excerpt": text[:1500]})
        except Exception as e:
            sources.append({"url": r["url"], "title": r["title"], "excerpt": r["snippet"], "note": f"fetch failed: {e}"})
    slug = "".join(c if c.isalnum() else "-" for c in question.lower()).strip("-")[:40] or "research"
    p = safe_path(f"research/{slug}.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# {question}\n\n" + "\n\n".join(f"## {s['title']}\n{s['url']}\n\n{s['excerpt']}" for s in sources))
    return {"saved": f"research/{slug}.md", "sources": sources}
