"""Compare smoke_gold labels with manifests/scorer_check.json and summarise a trajectory file."""
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
run = sys.argv[1] if len(sys.argv) > 1 else "smoke_gold"
recs = [json.loads(l) for l in (REPO / "runs" / run / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
print("records", len(recs), "unique run ids", len({r["run_id"] for r in recs}))
print("status", Counter(r["status"] for r in recs))
print("label", Counter((r.get("benchmark_label") or {}).get("score") for r in recs))
print("result csv saved", Counter(r.get("result_csv", {}).get("saved") for r in recs))
chk = REPO / "manifests" / "scorer_check.json"
if run == "smoke_gold" and chk.exists():
    ref = json.loads(chk.read_text())["gold_sql_scored"]
    mism = [r["run_id"] for r in recs if r["benchmark_label"]["score"] != ref[r["instance_id"]]["score"]]
    print("label mismatches vs scorer_check:", mism or "none")
keys = sorted(recs[0].keys())
print("record keys", keys)
