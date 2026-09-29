"""Teachable memory: the bot learns instructions, shortcuts and preferences and applies them later.

Works with every provider: the model sees lessons in its system prompt; the offline engine
expands a trigger phrase into its stored command.
"""
from bot import memory
from bot.registry import tool


def lessons_prompt() -> str:
    rows = memory.query("SELECT * FROM lessons ORDER BY id")
    if not rows:
        return ""
    lines = [f"- {r['trigger']}: {r['action']}" if r["trigger"] else f"- {r['action']}" for r in rows]
    return "\n\nThings the user has taught you (follow them):\n" + "\n".join(lines)


@tool("Learn an instruction. trigger is a phrase the user will say (or empty for a general preference/rule); "
      "action is what to do when it comes up. Use when the user says 'remember', 'from now on', "
      "'when I say X do Y', or corrects your behavior.", sensitive=True)
def learn_instruction(trigger: str, action: str) -> dict:
    row = memory.query("SELECT id FROM lessons WHERE trigger=? AND action=?", (trigger, action))
    if row:
        return {"id": row[0]["id"], "already_known": True}
    return {"id": memory.execute("INSERT INTO lessons(trigger,action) VALUES(?,?)", (trigger, action)).lastrowid}


@tool("List everything the bot has learned.")
def list_lessons() -> list:
    return memory.query("SELECT * FROM lessons ORDER BY id")


@tool("Forget a learned instruction by id.", destructive=True)
def forget_lesson(id: int) -> dict:
    return {"deleted": memory.execute("DELETE FROM lessons WHERE id=?", (id,)).rowcount}
