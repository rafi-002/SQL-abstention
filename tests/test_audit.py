"""Audit packet blinding checks (no key, no spend). Skipped until tools/make_audit_packets.py has been run."""
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKETS = REPO / "audit" / "packets"
KEY = REPO / "audit" / "key.json"
pytestmark = pytest.mark.skipif(not KEY.exists(), reason="audit packets not built")

# Markers that would reveal a candidate's source or grade (data words like GOLDFINGER are allowed).
LEAK = re.compile(r"\bgold\b|gold_files|__r\d\d|exec_result|local\d{3}_[a-d]\.csv|benchmark|harness|agent_runs", re.I)


def test_every_question_has_a_packet_and_key():
    key = json.loads(KEY.read_text())
    assert len(key) == 135
    assert {p.stem for p in PACKETS.glob("*.md")} == set(key)


def test_candidate_sections_reveal_no_source_or_grade():
    for p in PACKETS.glob("*.md"):
        cand = p.read_text(encoding="utf-8").split("## Candidates", 1)[1]
        assert not LEAK.search(cand), p.name


def test_every_gold_variant_and_answer_is_a_candidate():
    key = json.loads(KEY.read_text())
    gold = sum(len(c["gold_files"]) for q in key.values() for c in q.values())
    assert gold == 328
    assert all(c["agent_runs"] or c["gold_files"] for q in key.values() for c in q.values())


def test_label_templates_list_each_candidate_once():
    key = json.loads(KEY.read_text())
    rows = (REPO / "audit" / "labels_A1.csv").read_text(encoding="utf-8").splitlines()[1:]
    pairs = [tuple(r.split(",")[:2]) for r in rows]
    assert len(pairs) == len(set(pairs)) == sum(len(v) for v in key.values())
