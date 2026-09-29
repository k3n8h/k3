"""Background scheduler: appointment reminders and scheduled autonomous jobs."""
from datetime import datetime, timedelta
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler

from bot import memory

_notify: Callable[[str], None] = print


def due_reminders(now: datetime) -> list[dict]:
    out = []
    for e in memory.query("SELECT * FROM events WHERE remind_minutes>0 AND reminded=0"):
        start = datetime.fromisoformat(e["start"])
        if start - timedelta(minutes=e["remind_minutes"]) <= now < start:
            out.append(e)
    return out


def tick(now: Optional[datetime] = None, run_job: Optional[Callable[[str], str]] = None) -> None:
    now = now or datetime.now()
    for e in due_reminders(now):
        _notify(f"Reminder: {e['title']} at {e['start']}")
        memory.execute("UPDATE events SET reminded=1 WHERE id=?", (e["id"],))
    if run_job:
        for j in memory.query("SELECT * FROM jobs WHERE done=0 AND run_at<=?", (now.isoformat(),)):
            memory.execute("UPDATE jobs SET done=1 WHERE id=?", (j["id"],))
            result = run_job(j["goal"])
            memory.execute("UPDATE jobs SET result=? WHERE id=?", (result, j["id"]))
            _notify(f"Scheduled job finished: {j['goal']}\n{result}")


def start(notify: Callable[[str], None] = print, run_job: Optional[Callable[[str], str]] = None):
    global _notify
    _notify = notify
    sched = BackgroundScheduler()
    sched.add_job(lambda: tick(run_job=run_job), "interval", seconds=30)
    sched.start()
    return sched
