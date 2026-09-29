"""Tasks and notes."""
from bot import memory
from bot.registry import tool


@tool("Add a to-do task. priority 1 (high) to 3 (low); due is an optional ISO date.")
def add_task(title: str, due: str = "", priority: int = 2) -> dict:
    return {"id": memory.execute("INSERT INTO tasks(title,due,priority) VALUES(?,?,?)", (title, due, priority)).lastrowid}


@tool("List tasks, open ones by default; set include_done to see completed tasks too.")
def list_tasks(include_done: bool = False) -> list:
    sql = "SELECT * FROM tasks" + ("" if include_done else " WHERE done=0") + " ORDER BY priority, due"
    return memory.query(sql)


@tool("Mark a task done by id.")
def complete_task(id: int) -> dict:
    return {"updated": memory.execute("UPDATE tasks SET done=1 WHERE id=?", (id,)).rowcount}


@tool("Save a note with optional comma-separated tags.")
def add_note(title: str, body: str, tags: str = "") -> dict:
    return {"id": memory.execute("INSERT INTO notes(title,body,tags) VALUES(?,?,?)", (title, body, tags)).lastrowid}


@tool("Search notes by substring in title, body or tags.")
def search_notes(query: str) -> list:
    like = f"%{query}%"
    return memory.query("SELECT * FROM notes WHERE title LIKE ? OR body LIKE ? OR tags LIKE ?", (like, like, like))


@tool("Delete a note by id.", destructive=True)
def delete_note(id: int) -> dict:
    return {"deleted": memory.execute("DELETE FROM notes WHERE id=?", (id,)).rowcount}
