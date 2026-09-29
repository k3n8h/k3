"""Tiny tool registry: @tool turns a typed function into an Anthropic tool definition."""
import inspect
import json
import typing
from typing import Any, Callable

_TOOLS: dict[str, dict] = {}
_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _schema_for(fn: Callable) -> dict:
    props, required = {}, []
    hints = typing.get_type_hints(fn)
    for name, param in inspect.signature(fn).parameters.items():
        t = hints.get(name, str)
        origin = typing.get_origin(t)
        if origin is typing.Union:  # Optional[X]
            args = [a for a in typing.get_args(t) if a is not type(None)]
            t = args[0]
        if typing.get_origin(t) is list:
            t = list
        props[name] = {"type": _JSON_TYPES.get(t, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def tool(description: str, destructive: bool = False):
    def deco(fn: Callable) -> Callable:
        _TOOLS[fn.__name__] = {
            "fn": fn,
            "destructive": destructive,
            "definition": {"name": fn.__name__, "description": description, "input_schema": _schema_for(fn)},
        }
        return fn
    return deco


def definitions() -> list[dict]:
    return [t["definition"] for t in _TOOLS.values()]


def is_destructive(name: str) -> bool:
    return bool(_TOOLS.get(name, {}).get("destructive"))


def call(name: str, args: dict) -> str:
    entry = _TOOLS.get(name)
    if entry is None:
        return f"error: unknown tool {name}"
    try:
        result: Any = entry["fn"](**args)
    except Exception as e:  # tool errors go back to the model, not the user's stack
        return f"error: {type(e).__name__}: {e}"
    return result if isinstance(result, str) else json.dumps(result, default=str)


def load_all() -> None:
    from bot.tools import (autonomous, calendar, code, create, data, fun,  # noqa: F401
                           learning, moderation, organize, research, training, utilities)
