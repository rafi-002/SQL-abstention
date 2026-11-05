"""Count Spider 2.0-Lite tasks by backend and list local (SQLite) databases."""
import json, collections, sys, pathlib
root = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in open(root / "spider2-lite.jsonl", encoding="utf-8")]
kind = collections.Counter(r["instance_id"].rstrip("0123456789_") for r in rows)
print("total", len(rows), dict(kind))
local = [r for r in rows if r["instance_id"].startswith("local")]
dbs = collections.Counter(r["db"] for r in local)
print("local tasks", len(local), "local dbs", len(dbs))
print(sorted(dbs.items(), key=lambda x: -x[1]))
gold_sql = root / "evaluation_suite/gold/sql"
gold_csv = root / "evaluation_suite/gold/exec_result"
ids = {r["instance_id"] for r in local}
print("local gold sql", sum(1 for p in gold_sql.glob("local*.sql")))
print("local gold csv tasks", len({p.stem.split("_")[0] if p.stem.count("_") and not p.stem.startswith("local") else p.stem.rsplit("_",1)[0] if p.stem.rsplit("_",1)[-1].isalpha() and len(p.stem.rsplit("_",1)[-1])==1 else p.stem for p in gold_csv.glob("local*.csv")}))
print("sample csv names", sorted(p.name for p in gold_csv.glob("local*.csv"))[:8])
ev = [json.loads(l) for l in open(root / "evaluation_suite/gold/spider2lite_eval.jsonl", encoding="utf-8")] if (root / "evaluation_suite/gold/spider2lite_eval.jsonl").exists() else []
print("eval rows", len(ev), "local eval rows", sum(1 for e in ev if e["instance_id"].startswith("local")))
print(ev[0] if ev else None)
