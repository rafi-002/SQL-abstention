"""LLM judge for the hybrid audit (protocol v0.3, notes/M4_audit_protocol.md section 13).

The judge sees exactly what a human auditor sees: the blinded packet (question, knowledge document, schema,
anonymous candidate tables). It can run read-only SQL on the database and page through a candidate's full
table, then must call submit_verdicts once with a verdict for every candidate under the protocol's rules.

Outputs go to runs/<run_name>/ (git-ignored), so that the human annotator never sees judge verdicts on the
questions they will label:
  judge.jsonl        one record per question (verdicts, readings, full messages, tokens, cost)
  labels_judge.csv   same columns as audit/labels_A1.csv (blind_verdict filled)
  console.log

Run: .venv/Scripts/python.exe tools/llm_judge.py configs/judge_<name>.json
"""
from __future__ import annotations

import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import REPO, load_tasks  # noqa: E402
from sqlabst.llm import OpenRouterClient, ScriptedClient  # noqa: E402
from sqlabst.sqlexec import execute  # noqa: E402

PROMPT_VERSION = "judge-v1.1"   # v1.1: adds a submit reminder two steps before the step limit (used for retries)
VERDICTS = {"CORRECT", "CORRECT_ALT", "WRONG", "CANNOT_JUDGE"}
REASONS = {"W1", "W2", "W3", "W4", "W5", "W6"}

SYSTEM = """You are an expert SQL analyst auditing answers to a database question. Several anonymous candidate
result tables are given. Some may be right, some wrong; you are not told where they came from. Judge each one.

Procedure:
1. First decide what the question means: write the intended reading, and every other defensible reading
   (R1, R2, ...). A reading is defensible if a competent analyst could reasonably adopt it from the text and it
   contradicts no explicit instruction.
2. Check candidates against the data. Use execute_sql (read-only SQLite, results cut to 50 rows) to compute or
   verify what the correct answer should be. Use read_candidate to see rows of a candidate beyond those shown.
3. Give every candidate exactly one verdict:
   CORRECT       right under the intended reading
   CORRECT_ALT   right under another defensible reading (give it, e.g. "R2")
   WRONG         not right under any defensible reading (give a reason code)
   CANNOT_JUDGE  you cannot decide even after checking (explain)
   Reason codes for WRONG: W1 wrong logic/join/filter; W2 wrong set of entities (missing/extra rows);
   W3 incomplete output (a requested column or value missing); W4 ignores an explicit instruction;
   W5 numerical error beyond tolerance; W6 other (explain).

Rules:
- Every value the question asks for must be present and correct. Extra columns are allowed.
- Column names and column order do not matter.
- Row order matters only if the question asks for an order (top-k, sorted, rank).
- Numbers: equal after rounding to the precision the question asks for; if none is stated, within 0.5% relative
  (or 0.01 absolute for values below 2).
- Equivalent formats are fine when unambiguous (0.25 vs 25% with a clear label; date formats).
- Ties: any tie-consistent answer is correct unless the question states a tie-break.
- Explicit instructions in the question or external knowledge must be followed.
- Missing a requested column or entity is WRONG (W3).
Finish by calling submit_verdicts exactly once, covering every candidate."""

TOOLS = [
    {"type": "function", "function": {
        "name": "execute_sql", "description": "Run one read-only SQLite query on the question's database.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}}},
    {"type": "function", "function": {
        "name": "read_candidate", "description": "Show 50 rows of a candidate's full table starting at offset.",
        "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string"},
                                                        "offset": {"type": "integer"}},
                       "required": ["candidate_id"]}}},
    {"type": "function", "function": {
        "name": "submit_verdicts", "description": "Submit the final verdicts. Ends the task.",
        "parameters": {"type": "object", "properties": {
            "intended_reading": {"type": "string"},
            "readings": {"type": "string", "description": "R1: ...; R2: ... (empty if only one)"},
            "ambiguous": {"type": "boolean"}, "broken": {"type": "boolean"},
            "verdicts": {"type": "array", "items": {"type": "object", "properties": {
                "candidate_id": {"type": "string"}, "verdict": {"type": "string", "enum": sorted(VERDICTS)},
                "alt_reading": {"type": "string"}, "reason_code": {"type": "string"}, "note": {"type": "string"}},
                "required": ["candidate_id", "verdict"]}}},
            "required": ["intended_reading", "ambiguous", "broken", "verdicts"]}}},
]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_verdicts(args: dict, cids: list[str]) -> str | None:
    """Return an error message if the submission is incomplete or malformed, else None."""
    vs = args.get("verdicts") if isinstance(args, dict) else None
    if not isinstance(vs, list):
        return "verdicts must be a list"
    got = {}
    for v in vs:
        if not isinstance(v, dict) or v.get("candidate_id") not in cids:
            return f"unknown candidate_id in {v!r}; valid ids: {cids}"
        if v.get("verdict") not in VERDICTS:
            return f"invalid verdict for {v.get('candidate_id')}: {v.get('verdict')!r}"
        if v["verdict"] == "WRONG" and v.get("reason_code") not in REASONS:
            return f"WRONG verdict for {v['candidate_id']} needs a reason_code in {sorted(REASONS)}"
        got[v["candidate_id"]] = v
    missing = [c for c in cids if c not in got]
    return f"missing verdicts for candidates {missing}" if missing else None


def judge_question(qid: str, task, client, max_steps: int, token_budget: int, obs_max_chars: int, audit_dir: str = "audit") -> dict:
    packet = (REPO / audit_dir / "packets" / f"{qid}.md").read_text(encoding="utf-8")
    cdir = REPO / audit_dir / "candidates" / qid
    cids = sorted(p.stem for p in cdir.glob("*.csv"))
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": packet + f"\n\nCandidate ids: {', '.join(cids)}"}]
    steps, result, status = [], None, "step_limit"
    tin = tout = 0
    cost = 0.0
    for i in range(max_steps):
        reply = client.chat(messages, TOOLS, seed=0)
        tin, tout, cost = tin + reply.input_tokens, tout + reply.output_tokens, cost + reply.cost_usd
        am = {"role": "assistant", "content": reply.content or ""}
        if reply.tool_calls:
            am["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}} for c in reply.tool_calls]
        messages.append(am)
        step = {"i": i, "input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                "cost_usd": reply.cost_usd, "cost_source": reply.cost_source, "calls": []}
        if not reply.tool_calls:
            messages.append({"role": "user", "content": "Use a tool. Finish with submit_verdicts covering every candidate."})
        for c in reply.tool_calls:
            a = c.arguments if isinstance(c.arguments, dict) else {}
            if c.name == "submit_verdicts":
                err = check_verdicts(a, cids)
                if err:
                    text = f"ERROR: {err}. Call submit_verdicts again with a complete, valid submission."
                else:
                    result, status, text = a, "submitted", "Submitted."
            elif c.name == "execute_sql" and a.get("sql"):
                text = execute(str(task.db_path), a["sql"], timeout_s=30, row_cap=50).render(obs_max_chars)
            elif c.name == "read_candidate" and a.get("candidate_id") in cids:
                df = pd.read_csv(cdir / f"{a['candidate_id']}.csv")
                off = max(0, int(a.get("offset") or 0))
                text = (f"rows {off}-{min(off + 50, len(df)) - 1} of {len(df)}\n"
                        + df.iloc[off:off + 50].to_csv(index=False))[:obs_max_chars]
            else:
                text = f"ERROR: unknown tool or bad arguments ({c.name})"
            step["calls"].append({"name": c.name, "arguments": a, "result_head": text[:300]})
            messages.append({"role": "tool", "tool_call_id": c.id, "content": text})
            if status == "submitted":
                break
        steps.append(step)
        if status == "submitted":
            break
        if i == max_steps - 3 or (tin + tout >= 0.8 * token_budget and not step.get("_warned")):
            messages.append({"role": "user", "content": "You are almost out of steps. Call submit_verdicts now, "
                             "covering every candidate; use CANNOT_JUDGE where you could not decide."})
            step["_warned"] = True
        if tin + tout >= token_budget:
            status = "token_budget"
            break
    return {"qid": qid, "status": status, "result": result, "candidates": cids, "n_steps": len(steps),
            "input_tokens": tin, "output_tokens": tout, "cost_usd": round(cost, 6), "steps": steps,
            "messages": messages, "prompt_version": PROMPT_VERSION}


def max_cost(cfg) -> float:
    a = cfg["judge_llm"]
    return cfg["token_budget"] * a["price_in_per_m"] / 1e6 + cfg["max_steps"] * a["max_output_tokens"] * a["price_out_per_m"] / 1e6


def main(cfg_path: str):
    cfg = json.loads(Path(cfg_path).read_text())
    out = REPO / "runs" / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "console.log", "a", encoding="utf-8")

    def say(msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    tasks = {t.instance_id: t for t in load_tasks()}
    qids = cfg["questions"] if isinstance(cfg["questions"], list) else sorted(tasks)
    excl = set(cfg.get("exclude", []))
    qids = [q for q in qids if q not in excl]
    jf = out / "judge.jsonl"
    done, spent = set(), 0.0
    if jf.exists():
        for line in jf.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            spent += r.get("cost_usd", 0)
            if r["status"] == "submitted":
                done.add(r["qid"])
    todo = [q for q in qids if q not in done]
    cap, mrc, workers = cfg["spend_cap_usd"], max_cost(cfg), cfg.get("workers", 1)
    say(f"{cfg['run_name']}: {len(todo)} questions to judge, {len(done)} done, spent ${spent:.4f}, cap ${cap:.2f}, "
        f"max per question ${mrc:.3f}, workers {workers}")
    a = cfg["judge_llm"]
    if a["provider"] == "scripted":
        def make():
            return ScriptedClient(list(cfg["_script"]))
    else:
        def make():
            return OpenRouterClient(a["model"], a["price_in_per_m"], a["price_out_per_m"], a.get("temperature", 0.0),
                                    a["max_output_tokens"], a.get("retries", 5), a.get("backoff_cap_s", 60))
    lock = threading.Lock()
    in_flight = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(jf, "a", encoding="utf-8") as fh:
        futs = {}
        queue = list(todo)
        def submit_next():
            nonlocal in_flight
            if not queue:
                return False
            if spent + (in_flight + 1) * mrc > cap:
                if in_flight == 0:
                    say(f"spend cap: stopping before {queue[0]} ({len(queue)} not started)")
                    queue.clear()
                return False                   # wait for a running question to finish; never drop the item
            q = queue.pop(0)
            in_flight += 1
            futs[ex.submit(_safe, q, tasks[q], make(), cfg)] = q
            return True
        for _ in range(workers):
            if not submit_next():
                break
        while futs:
            f = next(iter(_as_completed(futs)))
            q = futs.pop(f)
            rec = f.result()
            with lock:
                in_flight -= 1
                spent += rec.get("cost_usd", 0)
                fh.write(json.dumps(rec) + "\n"); fh.flush()
            say(f"{q} {rec['status']} steps={rec.get('n_steps')} tok={rec.get('input_tokens')}+{rec.get('output_tokens')} "
                f"${rec.get('cost_usd', 0):.4f} total ${spent:.4f}")
            while len(futs) < workers and submit_next():
                pass
            if not futs and queue:
                submit_next()
    write_labels(out)
    say(f"finished; total spent ${spent:.4f}")


def _as_completed(futs):
    from concurrent.futures import as_completed
    return as_completed(list(futs))


def _safe(q, task, client, cfg):
    try:
        return judge_question(q, task, client, cfg["max_steps"], cfg["token_budget"], cfg.get("obs_max_chars", 6000), cfg.get("audit_dir", "audit"))
    except Exception as e:  # harness errors are recorded, never treated as verdicts
        return {"qid": q, "status": "harness_error", "error": repr(e)[:2000], "cost_usd": 0.0}


def write_labels(out: Path):
    rows = []
    for line in (out / "judge.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] != "submitted":
            continue
        for v in r["result"]["verdicts"]:
            rows.append({"qid": r["qid"], "candidate_id": v["candidate_id"], "blind_verdict": v["verdict"],
                         "alt_reading": v.get("alt_reading", ""), "reason_code": v.get("reason_code", ""),
                         "note": v.get("note", ""), "final_verdict": "", "change_reason": "",
                         "annotator": "judge:" + r.get("prompt_version", "?"), "date": now()[:10]})
    with open(out / "labels_judge.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["qid", "candidate_id", "blind_verdict", "alt_reading", "reason_code", "note",
                                           "final_verdict", "change_reason", "annotator", "date"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1])
