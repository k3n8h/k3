"""Web chat: uvicorn bot.interfaces.web:app

Single-user local server. Destructive tools are denied in the web UI (no interactive confirm).
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from bot.agent import Agent

app = FastAPI(title="K3 bot")
_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


class ChatIn(BaseModel):
    message: str


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    tools_used: list[str] = []
    reply = get_agent().run(body.message, on_tool=lambda n, a: tools_used.append(n))
    return {"reply": reply, "tools": tools_used}


@app.post("/reset")
def reset() -> dict:
    get_agent().reset()
    return {"ok": True}


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>K3</title>
<style>body{font:16px system-ui;max-width:720px;margin:0 auto;padding:16px;background:#fafafa;color:#111}
#log{min-height:60vh}.m{margin:8px 0;padding:8px 12px;border-radius:8px;white-space:pre-wrap}
.u{background:#dbeafe;text-align:right}.b{background:#fff;border:1px solid #ddd}.t{color:#666;font-size:12px}
form{display:flex;gap:8px}input{flex:1;padding:10px;font:inherit}
@media(prefers-color-scheme:dark){body{background:#111;color:#eee}.b{background:#1c1c1c;border-color:#333}.u{background:#1e3a8a}}
</style><h1>K3</h1><div id=log></div>
<form id=f><input id=i autofocus placeholder="Ask anything..."><button>Send</button></form>
<script>
const log=document.getElementById('log');
function add(c,t){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);return d}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();
const i=document.getElementById('i'),m=i.value.trim();if(!m)return;i.value='';add('u',m);
const w=add('b','...');
try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});
const j=await r.json();w.textContent=j.reply;if(j.tools.length)add('t','tools: '+j.tools.join(', '))}
catch(x){w.textContent='Error: '+x}}
</script>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE
