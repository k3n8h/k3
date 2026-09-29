"""Model-provider abstraction so the bot is not tied to any single API.

Internal message format is Anthropic-style blocks:
  user:      str | [{"type":"tool_result","tool_use_id","content"}]
  assistant: [{"type":"text","text"} | {"type":"tool_use","id","name","input"}]
Providers: anthropic, openai-compatible (OpenAI, Ollama, vLLM, LM Studio, ...) and offline (no model at all).
"""
import json
import os
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Reply:
    text: str = ""
    tool_calls: list = field(default_factory=list)

    def blocks(self) -> list[dict]:
        out = [{"type": "text", "text": self.text}] if self.text else []
        out += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in self.tool_calls]
        return out


class Provider:
    name = "base"

    def complete(self, system: str, messages: list, tools: list, max_tokens: int = 4096) -> Reply:
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model = client, model

    def complete(self, system, messages, tools, max_tokens=4096) -> Reply:
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, tools=tools, messages=messages,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
        return Reply(
            "".join(b.text for b in r.content if b.type == "text"),
            [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"])


def to_openai_messages(system: str, messages: list) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        c = m["content"]
        if m["role"] == "user":
            if isinstance(c, str):
                out.append({"role": "user", "content": c})
            else:
                out += [{"role": "tool", "tool_call_id": b["tool_use_id"], "content": str(b["content"])} for b in c]
        else:
            text = "".join(b["text"] for b in c if b["type"] == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in c if b["type"] == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
    return out


class OpenAICompatProvider(Provider):
    name = "openai-compatible"

    def __init__(self, model: str, base_url: str, api_key: str = "", http=None):
        import httpx
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.http = http or httpx.Client(timeout=120)

    def complete(self, system, messages, tools, max_tokens=4096) -> Reply:
        body = {"model": self.model, "max_tokens": max_tokens,
                "messages": to_openai_messages(system, messages),
                "tools": [{"type": "function", "function": {
                    "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                    for t in tools]}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = self.http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(c["id"], c["function"]["name"], args))
        return Reply(msg.get("content") or "", calls)


def get_provider(name: str = "") -> Provider:
    """BOT_PROVIDER = anthropic | openai | offline | auto (default).

    auto: Anthropic if ANTHROPIC_API_KEY, else an OpenAI-compatible endpoint if BOT_BASE_URL /
    OPENAI_API_KEY is set, else the offline rule-based engine (no network model required).
    """
    from bot import config
    name = (name or os.environ.get("BOT_PROVIDER", "auto")).lower()
    if name == "auto":
        if os.environ.get("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.environ.get("BOT_BASE_URL") or os.environ.get("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "offline"
    if name == "anthropic":
        return AnthropicProvider(config.MODEL)
    if name in ("openai", "local", "ollama"):
        base = os.environ.get("BOT_BASE_URL") or "https://api.openai.com/v1"
        return OpenAICompatProvider(os.environ.get("BOT_MODEL", "gpt-4o-mini"), base,
                                    os.environ.get("BOT_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    if name == "offline":
        from bot.offline import OfflineProvider
        return OfflineProvider()
    raise ValueError(f"unknown provider {name!r}")
