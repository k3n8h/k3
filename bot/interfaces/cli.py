"""Terminal chat: python -m bot.interfaces.cli"""
import sys

from rich.console import Console
from rich.markdown import Markdown

from bot import registry, scheduler
from bot.agent import Agent
from bot.tools import autonomous

console = Console()


def confirm(name: str, args: dict) -> bool:
    return console.input(f"[bold red]Allow {name}({args})? [y/N] [/]").strip().lower() == "y"


def handle_slash(cmd: str, agent: Agent) -> bool:
    if cmd == "/clear":
        agent.reset()
        console.print("history cleared")
    elif cmd == "/tools":
        for d in registry.definitions():
            console.print(f"[cyan]{d['name']}[/] - {d['description']}")
    elif cmd == "/help":
        console.print("/tools  /clear  /quit  - anything else is sent to the bot")
    else:
        return False
    return True


def main() -> None:
    agent = Agent(confirm=confirm)
    sched = scheduler.start(notify=lambda m: console.print(f"\n[yellow]{m}[/]"), run_job=autonomous.run_goal)
    console.print("[bold]K3[/] ready. /help for commands.")
    try:
        while True:
            try:
                text = console.input("[bold green]you>[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in ("/quit", "/exit"):
                break
            if text.startswith("/") and handle_slash(text, agent):
                continue
            reply = agent.run(text, on_tool=lambda n, a: console.print(f"[dim]  tool: {n}[/]"))
            console.print(Markdown(reply))
    finally:
        sched.shutdown(wait=False)
    sys.exit(0)


if __name__ == "__main__":
    main()
