"""Small utilities."""
import ast
import operator as op
from datetime import datetime, timedelta

from bot.registry import tool

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}


def safe_eval(expr: str) -> float:
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


@tool("Evaluate an arithmetic expression (+ - * / ** % //, parentheses).")
def calculate(expression: str) -> float:
    return safe_eval(expression)


@tool("Current local date and time (ISO 8601) and weekday.")
def now() -> dict:
    n = datetime.now()
    return {"iso": n.isoformat(timespec="minutes"), "weekday": n.strftime("%A")}


@tool("Add days to a YYYY-MM-DD date (negative to subtract); returns the new date and weekday.")
def date_add(date: str, days: int) -> dict:
    d = datetime.fromisoformat(date) + timedelta(days=days)
    return {"date": d.date().isoformat(), "weekday": d.strftime("%A")}


_UNITS = {"km": 1000, "m": 1, "cm": 0.01, "mi": 1609.344, "ft": 0.3048, "in": 0.0254,
          "kg": 1, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523}
_GROUPS = [{"km", "m", "cm", "mi", "ft", "in"}, {"kg", "g", "lb", "oz"}]


@tool("Convert length (km,m,cm,mi,ft,in), mass (kg,g,lb,oz) or temperature (c,f,k) between units.")
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    a, b = from_unit.lower(), to_unit.lower()
    if a in ("c", "f", "k") and b in ("c", "f", "k"):
        c = value if a == "c" else (value - 32) * 5 / 9 if a == "f" else value - 273.15
        return c if b == "c" else c * 9 / 5 + 32 if b == "f" else c + 273.15
    if not any(a in g and b in g for g in _GROUPS):
        raise ValueError("incompatible or unknown units")
    return value * _UNITS[a] / _UNITS[b]


@tool("Create a simple poll and return a formatted ballot for the user to answer.")
def make_poll(question: str, options: list) -> str:
    return question + "\n" + "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
