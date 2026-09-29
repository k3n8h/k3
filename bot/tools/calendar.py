"""Schedules, classes and appointments."""
from datetime import datetime, timedelta

from bot import memory
from bot.registry import tool

KINDS = ("class", "appointment", "event")


def find_conflicts(start: str, end: str, exclude_id: int = 0) -> list[dict]:
    s, e = datetime.fromisoformat(start), datetime.fromisoformat(end)
    return [ev for ev in memory.query("SELECT * FROM events WHERE id<>?", (exclude_id,))
            if datetime.fromisoformat(ev["start"]) < e and s < datetime.fromisoformat(ev["end"])]


@tool("Add a class, appointment or event. start/end are ISO 8601 local datetimes (YYYY-MM-DDTHH:MM). "
      "kind is one of class, appointment, event. Refuses on time conflicts unless allow_conflict is true.")
def add_event(kind: str, title: str, start: str, end: str, location: str = "", notes: str = "",
              remind_minutes: int = 0, allow_conflict: bool = False) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if datetime.fromisoformat(end) <= datetime.fromisoformat(start):
        raise ValueError("end must be after start")
    conflicts = find_conflicts(start, end)
    if conflicts and not allow_conflict:
        return {"created": False, "conflicts": conflicts}
    cur = memory.execute(
        "INSERT INTO events(kind,title,start,end,location,notes,remind_minutes) VALUES(?,?,?,?,?,?,?)",
        (kind, title, start, end, location, notes, remind_minutes))
    return {"created": True, "id": cur.lastrowid}


@tool("List events between two ISO datetimes (default: next 7 days), optionally filtered by kind.")
def list_events(start: str = "", end: str = "", kind: str = "") -> list:
    s = start or datetime.now().isoformat(timespec="minutes")
    e = end or (datetime.now() + timedelta(days=7)).isoformat(timespec="minutes")
    rows = memory.query("SELECT * FROM events WHERE start>=? AND start<=? ORDER BY start", (s, e))
    return [r for r in rows if not kind or r["kind"] == kind]


@tool("Find free slots of at least `minutes` on a date (YYYY-MM-DD) between 09:00 and 17:00.")
def find_free_slots(date: str, minutes: int = 60) -> list:
    day_start, day_end = datetime.fromisoformat(f"{date}T09:00"), datetime.fromisoformat(f"{date}T17:00")
    busy = sorted((datetime.fromisoformat(e["start"]), datetime.fromisoformat(e["end"]))
                  for e in memory.query("SELECT * FROM events") if day_start.date() == datetime.fromisoformat(e["start"]).date())
    slots, cursor = [], day_start
    for s, e in busy + [(day_end, day_end)]:
        if s - cursor >= timedelta(minutes=minutes):
            slots.append({"start": cursor.isoformat(timespec="minutes"), "end": s.isoformat(timespec="minutes")})
        cursor = max(cursor, e)
    return slots


@tool("Cancel (delete) an event by id.", destructive=True)
def cancel_event(id: int) -> dict:
    return {"deleted": memory.execute("DELETE FROM events WHERE id=?", (id,)).rowcount}
