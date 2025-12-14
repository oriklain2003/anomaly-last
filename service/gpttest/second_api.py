# backend.py
import os
import sqlite3
import logging
from typing import Dict, Any

import requests  # only used if you want to keep DuckDuckGo; not needed here actually
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("agent")

# -------------------------
# CONFIG
# -------------------------

# Set your key as ENV:  set OPENAI_API_KEY=...  (Windows)  /  export OPENAI_API_KEY=... (Linux/Mac)
OPENAI_API_KEY = "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"
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY environment variable (do NOT hardcode it).")

client = OpenAI(api_key=OPENAI_API_KEY)

DB_PATH = r"C:\Users\macab\Desktop\fiveair\anomaly-last\service\realtime\research.db"

AGENT_PROMPT = """
You are an autonomous reasoning agent for flight anomaly analysis.

You have access to TWO tools:
1. sql(query)  - run read-only SQL SELECT queries on the flight DB.
2. search(query) - perform online web search (via OpenAI web_search) for fresh aviation/flight knowledge.

Your goals:
- Decide when you need SQL queries.
- Decide when you need web searches.
- You may call these tools as MANY times as needed.
- Reason internally; do NOT reveal chain-of-thought.
- Once you have enough information, produce a clean final answer for the user.

DATABASE SCHEMA (IMPORTANT):

TABLE: anomalies_tracks
COLUMNS:
    flight_id   TEXT
    timestamp   INTEGER
    lat         REAL
    lon         REAL
    alt         REAL
    gspeed      REAL
    vspeed      REAL
    track       REAL
    squawk      TEXT
    callsign    TEXT
    source      TEXT
PRIMARY KEY(flight_id, timestamp)


TABLE: anomaly_reports

COLUMNS:
	id	INTEGER,
	flight_id	TEXT,
	timestamp	INTEGER,
	is_anomaly	BOOLEAN,
	severity_cnn	REAL,
	severity_dense	REAL,
	full_report	JSON,
	PRIMARY KEY("id" AUTOINCREMENT)

RULES:
- You may ONLY use SELECT queries on the anomalies_tracks table.
- Use the column names EXACTLY as written.
- Include WHERE filters when needed.
- If you expect many results, include LIMIT (the environment will add LIMIT 5000 if missing).

TOOL CALL FORMAT (VERY IMPORTANT):

When you want to call a tool, you MUST respond EXACTLY in one of these formats:

<tool>
sql: SELECT ...
</tool>

or

<tool>
search: "the query text here"
</tool>

Do NOT include anything else inside <tool> tags.
Do NOT put explanations around it; just the tool call.

After the environment responds with tool results (as plain text messages),
you may call tools again or move to a final answer.

When you are completely ready to answer the user, return:

<final>
YOUR FINAL ANSWER HERE
</final>

If you accidentally answer without <final>, the environment may use that as the final answer.
"""

# -------------------------
# TOOL IMPLEMENTATIONS
# -------------------------

def execute_safe_sql(query: str) -> Dict[str, Any]:
    """
    Read-only SQL execution on research.db
    Returns a dict with either {"rows": [...]} or {"error": "..."}.
    """
    log.info(f"[SQL] Incoming query: {query}")

    forbidden = ["insert", "update", "delete", "drop", "alter", "create"]
    if any(cmd in query.lower() for cmd in forbidden):
        log.warning("[SQL] BLOCKED dangerous query")
        return {"error": "Only SELECT queries allowed."}

    # Force a LIMIT if none is present
    if "limit" not in query.lower():
        query = query.rstrip().rstrip(";")
        query += " LIMIT 5000"
        log.info(f"[SQL] LIMIT appended -> {query}")

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        rows = cur.execute(query).fetchall()
        data = [dict(r) for r in rows]
        log.info(f"[SQL] Returned {len(data)} rows")
        return {"rows": data}
    except Exception as e:
        log.error(f"[SQL ERROR] {e}")
        return {"error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def openai_web_search(query: str) -> str:
    """
    Use OpenAI's official web_search tool via the Responses API.
    Corrected implementation: Responses API does NOT accept `messages`.
    """
    log.info(f"[WEB_SEARCH] Query: {query}")

    try:
        resp = client.responses.create(
            model="gpt-4.1-mini",  # fast; can change to gpt-4.1 or gpt-5
            input=query,
            tools=[{"type": "web_search"}],   # THIS is the correct schema
        )

        # Responses API always returns results via output_text
        result_text = resp.output_text
        log.info(f"[WEB_SEARCH] Got {len(result_text)} chars")

        return result_text

    except Exception as e:
        log.error(f"[WEB_SEARCH ERROR] {e}")
        return f"Web search error: {e}"

# -------------------------
# AGENT CONTROLLER (multi-step loop)
# -------------------------

def run_agent(user_message: str, max_steps: int = 8) -> str:
    """
    Core agent loop.
    The LLM can:
      - ask for <tool> sql: ...
      - ask for <tool> search: "..."
      - finally return <final> ... </final>
    It can call tools multiple times, up to max_steps.
    """
    messages = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": user_message},
    ]

    for step in range(max_steps):
        log.info("------------------------------------------------------")
        log.info(f"[AGENT] Step {step+1}/{max_steps}, messages so far: {len(messages)}")

        resp = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
        )
        msg = resp.choices[0].message.content
        log.info(f"[LLM RAW OUTPUT]\n{msg}")

        # Always record what the assistant just said
        messages.append({"role": "assistant", "content": msg})

        # 1) Final answer?
        if "<final>" in msg:
            try:
                final = msg.split("<final>")[1].split("</final>")[0].strip()
                if final:
                    log.info(f"[AGENT FINAL ANSWER]\n{final}")
                    return final
            except Exception as e:
                log.warning(f"[AGENT] Error parsing <final>: {e}")
                return msg.strip()

        # 2) SQL tool usage?
        if "<tool>" in msg and "sql:" in msg:
            try:
                tool_block = msg.split("<tool>")[1].split("</tool>")[0]
            except Exception as e:
                log.error(f"[AGENT] Error extracting <tool> block for SQL: {e}")
                continue

            try:
                sql_query = tool_block.split("sql:", 1)[1].strip()
                log.info(f"[AGENT → SQL] LLM requested SQL:\n{sql_query}")
                sql_result = execute_safe_sql(sql_query)
                log.info(f"[AGENT ← SQL RESULT] {str(sql_result)[:5]}...")
                # Feed tool result back as a "user" message so the model can continue reasoning
                messages.append({
                    "role": "user",
                    "content": f"Tool result (sql):\n{sql_result}"
                })
                continue
            except Exception as e:
                log.error(f"[AGENT] Tool (sql) parsing error: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Tool (sql) error parsing: {e}"
                })
                continue

        # 3) Search tool usage?
        if "<tool>" in msg and "search:" in msg:
            try:
                tool_block = msg.split("<tool>")[1].split("</tool>")[0]
            except Exception as e:
                log.error(f"[AGENT] Error extracting <tool> block for search: {e}")
                continue

            try:
                search_query = tool_block.split("search:", 1)[1].strip()
                # Strip quotes if present
                if search_query.startswith('"') and search_query.endswith('"'):
                    search_query = search_query[1:-1]
                log.info(f"[AGENT → WEB_SEARCH] LLM requested search: {search_query}")
                search_result = openai_web_search(search_query)
                log.info(f"[AGENT ← WEB_SEARCH RESULT] {search_result[:400]}...")
                messages.append({
                    "role": "user",
                    "content": f"Tool result (search):\n{search_result}"
                })
                continue
            except Exception as e:
                log.error(f"[AGENT] Tool (search) parsing error: {e}")
                messages.append({
                    "role": "user",
                    "content": f"Tool (search) error parsing: {e}"
                })
                continue

        # 4) If no tools & no <final> — assume this is final text
        log.info("[AGENT] No tool call or <final> tag detected, returning raw message.")
        return msg.strip()

    # If we hit max_steps without a final tag, just return last assistant message
    log.warning("[AGENT WARNING] Max reasoning steps reached without <final>.")
    return messages[-1]["content"].strip()

# -------------------------
# FASTAPI SETUP
# -------------------------

class ChatRequest(BaseModel):
    message: str

app = FastAPI()

# CORS so you can open index.html from file:// or other origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Main endpoint: receives { "message": "..." }
    Returns { "answer": "..." }
    """
    log.info(f"[API] /chat called with message: {req.message}")
    answer = run_agent(req.message)
    log.info(f"[API] Answer: {answer[:400]}...")
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=456)
