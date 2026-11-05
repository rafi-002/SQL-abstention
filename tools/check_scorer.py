"""Free scorer sanity checks. Writes manifests/scorer_check.json.

1. Every gold CSV of the 135 SQLite tasks, used as a prediction, must score 1 with the official comparator.
2. Each released gold SQL (24 tasks) is executed and scored by the official evaluator. A gold SQL that does
   NOT reproduce its own gold answer is itself a first piece of evidence about benchmark-label reliability.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import REPO, load_tasks  # noqa: E402
from sqlabst.scorer import score_gold_csv_against_itself, score_sql  # noqa: E402

if __name__ == "__main__":
    tasks = load_tasks()
    self_check = {t.instance_id: score_gold_csv_against_itself(t.instance_id) for t in tasks}
    n_csv = sum(len(v) for v in self_check.values())
    bad_self = {k: v for k, v in self_check.items() if not v or not all(v)}
    print(f"gold CSV self-match: {n_csv} CSVs over {len(self_check)} tasks; failures: {bad_self or 'none'}")

    gold_sql = {}
    for t in tasks:
        sql = t.gold_sql()
        if sql is None:
            continue
        t0 = time.time()
        r = score_sql(t.instance_id, sql)
        r["seconds"] = round(time.time() - t0, 1)
        gold_sql[t.instance_id] = r
        print(t.instance_id, r)
    ok = sum(1 for r in gold_sql.values() if r["score"] == 1)
    print(f"gold SQL reproduces gold answer: {ok}/{len(gold_sql)}")
    out = {
        "n_tasks": len(tasks), "n_gold_csv": n_csv, "gold_csv_self_match_failures": bad_self,
        "gold_sql_scored": gold_sql, "gold_sql_reproduces": f"{ok}/{len(gold_sql)}",
    }
    (REPO / "manifests").mkdir(exist_ok=True)
    (REPO / "manifests" / "scorer_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
