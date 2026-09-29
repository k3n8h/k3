"""Claude tool-use loop."""
from typing import Callable, Optional

from bot import config, registry


class Agent:
    def __init__(self, client=None, confirm: Optional[Callable[[str, dict], bool]] = None,
                 max_iterations: int = config.MAX_ITERATIONS, system: str = config.SYSTEM_PROMPT):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        registry.load_all()
        self.client = client
        self.confirm = confirm or (lambda name, args: False)  # deny destructive by default
        self.max_iterations = max_iterations
        self.system = system
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages.clear()

    def _create(self):
        return self.client.messages.create(
            model=config.MODEL, max_tokens=config.MAX_TOKENS,
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            tools=registry.definitions(), messages=self.messages)

    def run(self, user_text: str, on_tool: Optional[Callable[[str, dict], None]] = None) -> str:
        """Run one user turn to completion; returns the final assistant text."""
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(self.max_iterations):
            resp = self._create()
            self.messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if on_tool:
                    on_tool(block.name, block.input)
                if registry.is_destructive(block.name) and not self.confirm(block.name, block.input):
                    out = "error: user declined this destructive action"
                else:
                    out = registry.call(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
            self.messages.append({"role": "user", "content": results})
        return "(stopped: reached the tool-call limit for this turn)"
