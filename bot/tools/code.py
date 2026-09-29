"""Workspace files and sandboxed-ish Python execution (workspace cwd, timeout, isolated mode)."""
import subprocess
import sys
from pathlib import Path

from bot import config
from bot.registry import tool


def safe_path(rel: str) -> Path:
    root = config.workspace().resolve()
    p = (root / rel).resolve()
    if p != root and root not in p.parents:
        raise ValueError("path escapes the workspace")
    return p


@tool("Write a text file inside the workspace (creates parent dirs). Use for code, SVG/HTML designs, docs.", sensitive=True)
def write_file(path: str, content: str) -> str:
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} chars to {p.relative_to(config.workspace().resolve())}"


@tool("Read a text file from the workspace.")
def read_file(path: str) -> str:
    return safe_path(path).read_text()[:20000]


@tool("List files in the workspace (optionally a subdirectory).")
def list_files(path: str = ".") -> list:
    root = config.workspace().resolve()
    return sorted(str(f.relative_to(root)) for f in safe_path(path).rglob("*") if f.is_file())


@tool("Run Python code in the workspace directory with a timeout. Returns stdout, stderr and exit code. "
      "Not a security sandbox: only run code you would run yourself.", sensitive=True)
def run_python(code: str, timeout: int = config.CODE_TIMEOUT_S) -> dict:
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], cwd=config.workspace(), capture_output=True,
                           text=True, timeout=min(timeout, 60))
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "stdout": "", "stderr": f"timed out after {timeout}s"}
    return {"exit_code": r.returncode, "stdout": r.stdout[-8000:], "stderr": r.stderr[-4000:]}
