"""Creative/design helpers. The model does the writing; these tools persist and structure results."""
from bot.registry import tool
from bot.tools.code import safe_path

_SVG_WRAP = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{body}</svg>'


@tool("Save an SVG design. `body` is the inner SVG markup (shapes/text); it is wrapped in an <svg> element.")
def save_svg(path: str, body: str, width: int = 800, height: int = 600) -> str:
    if "<script" in body.lower():
        raise ValueError("scripts are not allowed in SVG designs")
    p = safe_path(path if path.endswith(".svg") else path + ".svg")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_SVG_WRAP.format(w=width, h=height, body=body))
    return f"saved {p.name}"


@tool("Save an HTML mockup/page to the workspace as a .html file.")
def save_html(path: str, html: str) -> str:
    p = safe_path(path if path.endswith(".html") else path + ".html")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html)
    return f"saved {p.name}"


@tool("Save a brainstorm as a markdown list in the workspace; ideas is a list of strings.")
def save_brainstorm(topic: str, ideas: list) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in topic.lower()).strip("-")[:40] or "brainstorm"
    p = safe_path(f"brainstorms/{slug}.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# {topic}\n\n" + "\n".join(f"- {i}" for i in ideas) + "\n")
    return f"saved brainstorms/{slug}.md"
