"""Offline checks for tools/llm_judge.py (scripted client, no key, no spend)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import REPO, load_tasks  # noqa: E402
from sqlabst.llm import ChatReply, ScriptedClient, ToolCall  # noqa: E402
from tools.llm_judge import check_verdicts, judge_question  # noqa: E402

QID = "local004"
pytestmark = pytest.mark.skipif(not (REPO / "audit" / "packets" / f"{QID}.md").exists(), reason="packets not built")
TASK = {t.instance_id: t for t in load_tasks()}[QID]
CIDS = sorted(p.stem for p in (REPO / "audit" / "candidates" / QID).glob("*.csv"))


def call(name, args, i=0):
    return ChatReply("", [ToolCall(f"c{i}", name, args)], 10, 5, 0.001, "provider", "scripted")


def full(verdict="CORRECT"):
    return {"intended_reading": "x", "ambiguous": False, "broken": False,
            "verdicts": [{"candidate_id": c, "verdict": verdict, "reason_code": "W1"} for c in CIDS]}


def test_check_verdicts():
    assert check_verdicts(full(), CIDS) is None
    assert "missing" in check_verdicts({"verdicts": []}, CIDS)
    bad = full(); bad["verdicts"][0]["verdict"] = "MAYBE"
    assert "invalid verdict" in check_verdicts(bad, CIDS)
    nored = full("WRONG"); nored["verdicts"][0]["reason_code"] = ""
    assert "reason_code" in check_verdicts(nored, CIDS)


def test_explore_then_submit():
    client = ScriptedClient([call("execute_sql", {"sql": "SELECT 1"}, 1), call("submit_verdicts", full(), 2)])
    r = judge_question(QID, TASK, client, max_steps=5, token_budget=10_000, obs_max_chars=2000)
    assert r["status"] == "submitted" and r["n_steps"] == 2
    assert {v["candidate_id"] for v in r["result"]["verdicts"]} == set(CIDS)


def test_incomplete_submission_is_rejected_then_fixed():
    client = ScriptedClient([call("submit_verdicts", {"intended_reading": "x", "ambiguous": False, "broken": False,
                                                      "verdicts": []}, 1),
                             call("submit_verdicts", full(), 2)])
    r = judge_question(QID, TASK, client, max_steps=5, token_budget=10_000, obs_max_chars=2000)
    assert r["status"] == "submitted" and r["n_steps"] == 2
    assert "ERROR" in r["messages"][3]["content"]


def test_step_limit_without_submission():
    client = ScriptedClient([call("execute_sql", {"sql": "SELECT 1"}, i) for i in range(3)])
    r = judge_question(QID, TASK, client, max_steps=3, token_budget=10_000, obs_max_chars=2000)
    assert r["status"] == "step_limit" and r["result"] is None


def test_read_candidate_pages_full_table():
    client = ScriptedClient([call("read_candidate", {"candidate_id": CIDS[0], "offset": 0}, 1), call("submit_verdicts", full(), 2)])
    r = judge_question(QID, TASK, client, max_steps=5, token_budget=10_000, obs_max_chars=2000)
    assert r["messages"][3]["content"].startswith("rows 0-")
