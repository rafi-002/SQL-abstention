"""ReAct-style SQLite agent with two tools (execute_sql, submit_sql). One call of run_agent = one trajectory."""
from __future__ import annotations

import json
import time
from dataclasses import asdict

from .data import Task, schema_ddl
from .sqlexec import execute

PROMPT_VERSION = "v1"

SYSTEM = """You are a data analyst answering questions over a SQLite database.
Explore the database with the execute_sql tool as needed (results are cut to {row_cap} rows), then call
submit_sql exactly once with ONE final SQLite query whose result table answers the question.
Return the columns the question asks for. Do not explain the answer in prose; the submitted query is the answer."""

USER = """Database: {db}

Schema:
{ddl}
{knowledge}
Question: {question}"""

TOOLS = [
    {"type": "function", "function": {
        "name": "execute_sql",
        "description": "Run one read-only SQLite query and see up to the first rows of the result.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "submit_sql",
        "description": "Submit the final SQLite query. Ends the task.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
]


def build_messages(task: Task, row_cap: int) -> list[dict]:
    k = task.knowledge_text()
    knowledge = f"\nExternal knowledge ({task.external_knowledge}):\n{k}\n" if k else ""
    return [
        {"role": "system", "content": SYSTEM.format(row_cap=row_cap)},
        {"role": "user", "content": USER.format(db=task.db, ddl=schema_ddl(str(task.db_path)),
                                                knowledge=knowledge, question=task.question)},
    ]


def run_agent(task: Task, client, *, max_steps: int = 15, token_budget: int = 120_000, row_cap: int = 50,
              sql_timeout_s: float = 30.0, obs_max_chars: int = 6000, seed: int | None = None) -> dict:
    """Returns the trajectory record (without benchmark label; the runner adds it)."""
    messages = build_messages(task, row_cap)
    steps, final_sql, status = [], None, "step_limit"
    tot_in = tot_out = 0
    tot_cost = 0.0
    for i in range(max_steps):
        reply = client.chat(messages, TOOLS, seed=seed)
        tot_in += reply.input_tokens
        tot_out += reply.output_tokens
        tot_cost += reply.cost_usd
        am = {"role": "assistant", "content": reply.content or ""}
        if reply.tool_calls:
            am["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}} for c in reply.tool_calls]
        messages.append(am)
        step = {"i": i, "content": reply.content, "finish_reason": reply.finish_reason, "model": reply.model,
                "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                "cost_usd": reply.cost_usd, "cost_source": reply.cost_source,
                "latency_s": round(reply.latency_s, 3), "tool_calls": [], "observations": []}
        if not reply.tool_calls:
            messages.append({"role": "user", "content": "Use a tool: execute_sql to explore, or submit_sql to finish."})
            step["observations"].append({"tool": None, "nudge": True})
        for c in reply.tool_calls:
            step["tool_calls"].append({"id": c.id, "name": c.name, "arguments": c.arguments})
            sql = c.arguments.get("sql") if isinstance(c.arguments, dict) else None
            if c.name == "submit_sql" and sql:
                final_sql, status = sql, "submitted"
                messages.append({"role": "tool", "tool_call_id": c.id, "content": "Submitted."})
                step["observations"].append({"tool": c.name, "submitted": True})
                break
            if c.name == "execute_sql" and sql:
                r = execute(str(task.db_path), sql, timeout_s=sql_timeout_s, row_cap=row_cap)
                text = r.render(obs_max_chars)
                step["observations"].append({"tool": c.name, "ok": r.ok, "error": r.error, "columns": r.columns,
                                             "n_rows_shown": len(r.rows), "truncated": r.truncated,
                                             "seconds": round(r.seconds, 3), "text": text})
            else:
                text = f"ERROR: unknown tool or missing 'sql' argument ({c.name})"
                step["observations"].append({"tool": c.name, "ok": False, "error": text})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": text})
        steps.append(step)
        if status == "submitted":
            break
        if tot_in + tot_out >= token_budget:
            status = "token_budget"
            break
    return {"prompt_version": PROMPT_VERSION, "status": status, "final_sql": final_sql, "n_steps": len(steps),
            "input_tokens": tot_in, "output_tokens": tot_out, "cost_usd": round(tot_cost, 6),
            "steps": steps, "messages": messages,
            "settings": {"max_steps": max_steps, "token_budget": token_budget, "row_cap": row_cap,
                         "sql_timeout_s": sql_timeout_s, "obs_max_chars": obs_max_chars, "seed": seed}}
