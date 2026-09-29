"""Provider-agnostic tool-use loop."""
from typing import Callable, Optional

from bot import config, registry
from bot.llm import Provider, get_provider


class Agent:
    def __init__(self, provider: Optional[Provider] = None, confirm: Optional[Callable[[str, dict], bool]] = None,
                 max_iterations: int = config.MAX_ITERATIONS, system: str = config.SYSTEM_PROMPT):
        registry.load_all()
        self.provider = provider or get_provider()
        self.confirm = confirm or (lambda name, args: False)  # deny destructive by default
        self.max_iterations = max_iterations
        self.system = system
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages.clear()

    def _system(self) -> str:
        from bot.tools.learning import lessons_prompt
        from bot import learner
        q = next((m["content"] for m in reversed(self.messages) if m["role"] == "user" and isinstance(m["content"], str)), "")
        return self.system + lessons_prompt() + (learner.hints(q) if q else "")

    def run(self, user_text: str, on_tool: Optional[Callable[[str, dict], None]] = None) -> str:
        """Run one user turn to completion; returns the final assistant text."""
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(self.max_iterations):
            reply = self.provider.complete(self._system(), self.messages, registry.definitions(), config.MAX_TOKENS)
            self.messages.append({"role": "assistant", "content": reply.blocks() or [{"type": "text", "text": ""}]})
            if not reply.tool_calls:
                return reply.text
            results = []
            for call in reply.tool_calls:
                if on_tool:
                    on_tool(call.name, call.input)
                if registry.needs_confirmation(call.name, getattr(self.provider, "user_driven", False)) \
                        and not self.confirm(call.name, call.input):
                    out = "error: user declined this destructive action"
                else:
                    out = registry.call(call.name, call.input)
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": out})
            self.messages.append({"role": "user", "content": results})
        return "(stopped: reached the tool-call limit for this turn)"
