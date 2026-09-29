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

    MAX_HISTORY = 60

    def _trim_history(self) -> None:
        """Bound prompt growth. Only cut at a plain user message so tool_use/tool_result pairs are never split."""
        if len(self.messages) <= self.MAX_HISTORY:
            return
        keep_from = len(self.messages) - self.MAX_HISTORY
        while keep_from < len(self.messages) and not (self.messages[keep_from]["role"] == "user"
                                                      and isinstance(self.messages[keep_from]["content"], str)):
            keep_from += 1
        if keep_from < len(self.messages):
            del self.messages[:keep_from]

    def run(self, user_text: str, on_tool: Optional[Callable[[str, dict], None]] = None) -> str:
        """Run one user turn to completion; returns the final assistant text."""
        self._trim_history()
        mark = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        for _ in range(self.max_iterations):
            try:
                reply = self.provider.complete(self._system(), self.messages, registry.definitions(), config.MAX_TOKENS)
            except Exception as e:                        # network/API failure: report it, keep the history consistent
                del self.messages[mark:]
                return f"(the model call failed: {type(e).__name__}: {e})"
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
