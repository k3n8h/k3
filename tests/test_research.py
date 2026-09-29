import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bot import llm, memory, registry, web
from bot.agent import Agent

PAGES = {
    "/": '<html><title>Home</title><body><nav>menu</nav><main><h1>Hi</h1><p class=x>alpha</p>'
         '<a href="/a">A</a><a href="/secret">S</a><a href="http://other.example/z">Z</a></main></body></html>',
    "/a": '<html><title>A page</title><body><p class=x>beta</p></body></html>',
    "/secret": "<html><title>no</title><body>hidden</body></html>",
    "/robots.txt": "User-agent: *\nDisallow: /secret\n",
}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGES.get(self.path)
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "text/plain" if self.path == "/robots.txt" else "text/html")
        self.end_headers()
        self.wfile.write((body or "").encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def site(monkeypatch, tmp_path):
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("BOT_ALLOW_PRIVATE_NETS", "1")
    monkeypatch.setenv("BOT_CRAWL_DELAY", "0")
    monkeypatch.setenv("BOT_HOME", str(tmp_path))
    memory.connect(str(tmp_path / "t.db"))
    web._robots.clear()
    registry.load_all()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ssrf_guard(monkeypatch):
    monkeypatch.delenv("BOT_ALLOW_PRIVATE_NETS", raising=False)
    for u in ("http://127.0.0.1/", "http://localhost:8080/", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            web.check_url(u)


def test_fetch_scrape_crawl(site):
    d = json.loads(registry.call("web_fetch", {"url": site + "/"}))
    assert d["title"] == "Home" and "alpha" in d["text"] and "menu" not in d["text"]
    assert json.loads(registry.call("scrape", {"url": site + "/", "selector": "p.x"})) == ["alpha"]
    assert json.loads(registry.call("scrape", {"url": site + "/", "selector": "a", "attr": "href"}))[0] == "/a"
    r = json.loads(registry.call("crawl", {"start_url": site + "/", "max_depth": 1, "save_as": "t"}))
    urls = {p["url"]: p for p in r["results"]}
    assert site + "/a" in urls and "beta" in urls[site + "/a"]["text"]
    assert "robots" in urls[site + "/secret"]["skipped"]
    assert not any("other.example" in u for u in urls)
    assert registry.call("read_file", {"path": "crawls/t.json"}).startswith("[")


def test_parse_ddg():
    html = ('<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com%2Fp">T</a>'
            '<a class="result__snippet">snip</a></div>')
    assert web.parse_ddg(html) == [{"title": "T", "url": "https://ex.com/p", "snippet": "snip"}]


def test_research_uses_search_and_fetch(site, monkeypatch):
    monkeypatch.setattr(web, "search", lambda q, n=5: [{"title": "H", "url": site + "/a", "snippet": "s"}])
    r = json.loads(registry.call("research", {"question": "what is beta"}))
    assert r["sources"][0]["excerpt"].strip() == "beta"
    assert "beta" in registry.call("read_file", {"path": r["saved"]})


def test_offline_provider_needs_no_api(site, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BOT_BASE_URL", raising=False)
    monkeypatch.delenv("BOT_PROVIDER", raising=False)
    assert llm.get_provider().name == "offline"
    a = Agent()
    assert "alpha" in a.run(f"fetch {site}/")
    assert "14" in a.run("calc 2+3*4")
    assert "offline" in a.run("blah blah")


def test_learning_offline_and_prompt():
    memory.connect(":memory:")
    registry.load_all()
    a = Agent(provider=llm.get_provider("offline"))
    a.run('learn "lucky number" => roll 1d2')
    assert "\"total\"" in a.run("give me my lucky number please")
    from bot.tools.learning import lessons_prompt
    assert "lucky number" in lessons_prompt()
    assert "deleted" not in a.run("forget 1")  # destructive -> denied without confirmation
    assert len(memory.query("SELECT * FROM lessons")) == 1


def test_openai_message_conversion_and_provider():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "x"},
                                              {"type": "tool_use", "id": "c1", "name": "flip_coin", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "heads"}]}]
    out = llm.to_openai_messages("sys", msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["name"] == "flip_coin" and out[3]["tool_call_id"] == "c1"

    class Http:
        def post(self, url, json=None, headers=None):
            assert url == "http://local/v1/chat/completions" and json["tools"][0]["function"]["name"] == "t"
            class R:
                def raise_for_status(self): pass
                def json(self): return {"choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "9", "function": {"name": "t", "arguments": '{"a": 1}'}}]}}]}
            return R()

    p = llm.OpenAICompatProvider("m", "http://local/v1", http=Http())
    rep = p.complete("s", [{"role": "user", "content": "q"}], [{"name": "t", "description": "d", "input_schema": {}}])
    assert rep.tool_calls[0].input == {"a": 1}
