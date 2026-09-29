"""Autonomous work: goals run by a fresh agent with a larger step budget; can be scheduled."""
from bot import config, memory
from bot.registry import tool

_client = None  # injectable for tests


def run_goal(goal: str) -> str:
    """Run a goal to completion without user interaction. Destructive tools are auto-denied."""
    from bot.agent import Agent
    agent = Agent(client=_client, max_iterations=config.AUTONOMOUS_MAX_ITERATIONS,
                  system=config.SYSTEM_PROMPT + "\nYou are working autonomously: plan, use tools, "
                  "then finish with a short report of what you did. Do not ask questions.")
    return agent.run(goal)


@tool("Schedule an autonomous job: `goal` will be worked on unattended at run_at (ISO datetime).")
def schedule_job(goal: str, run_at: str) -> dict:
    return {"id": memory.execute("INSERT INTO jobs(goal,run_at) VALUES(?,?)", (goal, run_at)).lastrowid}


@tool("List scheduled autonomous jobs and their results.")
def list_jobs() -> list:
    return memory.query("SELECT * FROM jobs ORDER BY run_at")
