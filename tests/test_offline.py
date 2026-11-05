"""Offline tests (no key, no spend). Run: .venv/Scripts/python.exe -m pytest -q tests"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.agent import TOOLS, build_messages, run_agent  # noqa: E402
from sqlabst.data import REPO, load_tasks  # noqa: E402
from sqlabst.llm import ChatReply, ScriptedClient, ToolCall  # noqa: E402
from sqlabst.scorer import score_sql  # noqa: E402
from sqlabst.sqlexec import execute  # noqa: E402

TASKS = {t.instance_id: t for t in load_tasks()}
T = TASKS["local004"]          # gold SQL reproduces the gold answer (manifests/scorer_check.json)


def reply(name=None, sql=None, content="", it=100, ot=10, cost=0.001):
    calls = [ToolCall(f"id_{name}_{abs(hash(sql)) % 9999}", name, {"sql": sql})] if name else []
    return ChatReply(content, calls, it, ot, cost, "provider", "scripted")


def test_snapshot_counts():
    assert len(TASKS) == 135
    assert len({t.db for t in TASKS.values()}) == 30
    assert all(t.db_path.exists() for t in TASKS.values())


def test_readonly_connection_blocks_writes():
    r = execute(str(T.db_path), "CREATE TABLE zz_should_fail(x)")
    assert not r.ok and "readonly" in r.error.lower()


def test_row_cap_and_truncation():
    r = execute(str(T.db_path), "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n WHERE i<500) SELECT i FROM n",
                row_cap=10)
    assert r.ok and len(r.rows) == 10 and r.truncated
    assert "more than 10 rows" in r.render()


def test_timeout_is_reported_not_hung():
    r = execute(str(T.db_path), "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM n) SELECT count(*) FROM n",
                timeout_s=1)
    assert not r.ok and "time limit" in r.error


def test_agent_explore_then_submit_records_everything():
    c = ScriptedClient([reply("execute_sql", "SELECT 1 AS one"), reply("submit_sql", T.gold_sql())])
    tr = run_agent(T, c, seed=3)
    assert tr["status"] == "submitted" and tr["final_sql"] == T.gold_sql()
    assert tr["n_steps"] == 2 and tr["input_tokens"] == 200 and tr["output_tokens"] == 20
    assert abs(tr["cost_usd"] - 0.002) < 1e-9
    obs = tr["steps"][0]["observations"][0]
    assert obs["ok"] and obs["columns"] == ["one"] and "1" in obs["text"]
    # the second model call saw the tool result of the first
    second = c.calls[1]
    assert second[-1]["role"] == "tool" and "one" in second[-1]["content"]
    # full message list is serialisable (it goes to JSONL)
    json.dumps(tr)


def test_agent_step_limit_and_nudge():
    c = ScriptedClient([reply(content="thinking")] * 3)
    tr = run_agent(T, c, max_steps=3)
    assert tr["status"] == "step_limit" and tr["final_sql"] is None
    assert tr["steps"][0]["observations"][0]["nudge"]


def test_agent_token_budget_stops():
    c = ScriptedClient([reply("execute_sql", "SELECT 1", it=600, ot=0)] * 10)
    tr = run_agent(T, c, token_budget=1000)
    assert tr["status"] == "token_budget" and tr["n_steps"] == 2


def test_agent_sql_error_is_observed():
    c = ScriptedClient([reply("execute_sql", "SELECT * FROM no_such_table"), reply("submit_sql", "SELECT 1")])
    tr = run_agent(T, c)
    o = tr["steps"][0]["observations"][0]
    assert not o["ok"] and "no such table" in o["error"]


def test_prompt_contains_schema_question_and_knowledge():
    m = build_messages(T, 50)
    assert "CREATE TABLE" in m[1]["content"] and T.question in m[1]["content"]
    k = next(t for t in TASKS.values() if t.external_knowledge)
    assert k.knowledge_text()[:40] in build_messages(k, 50)[1]["content"]
    assert {t["function"]["name"] for t in TOOLS} == {"execute_sql", "submit_sql"}


def test_official_scorer_right_and_wrong():
    assert score_sql("local004", T.gold_sql())["score"] == 1
    assert score_sql("local004", "SELECT 1 AS x")["score"] == 0


def test_folds_hold_out_whole_databases():
    f = json.loads((REPO / "manifests" / "folds_ldo.json").read_text())
    dbs = [d for x in f["folds"] for d in x["databases"]]
    assert len(dbs) == len(set(dbs)) == 30
    ids = [i for x in f["folds"] for i in x["task_ids"]]
    assert sorted(ids) == sorted(TASKS)
    for x in f["folds"]:
        assert {TASKS[i].db for i in x["task_ids"]} == set(x["databases"])


def test_spend_cap_bound():
    sys.path.insert(0, str(REPO / "tools"))
    from run_trajectories import max_run_cost
    cfg = json.loads((REPO / "configs" / "smoke_live.json").read_text())
    b = max_run_cost(cfg)
    a, ag = cfg["agent_llm"], cfg["agent"]
    assert b >= ag["token_budget"] * a["price_in_per_m"] / 1e6
    assert max_run_cost(json.loads((REPO / "configs" / "smoke_gold.json").read_text())) == 0.0
