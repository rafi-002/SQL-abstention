"""Fixed leave-database-out folds, written BEFORE any model run: manifests/folds_ldo.json.

Whole databases are assigned to K folds, balancing task counts (largest databases first, each to the
currently smallest fold; ties broken by a seeded shuffle). Deterministic for a given seed.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import REPO, load_tasks  # noqa: E402

K, SEED = 5, 20261103

if __name__ == "__main__":
    tasks = load_tasks()
    per_db = Counter(t.db for t in tasks)
    rng = random.Random(SEED)
    dbs = sorted(per_db)
    rng.shuffle(dbs)
    dbs.sort(key=lambda d: -per_db[d])          # stable: shuffle order breaks ties
    folds = [[] for _ in range(K)]
    for d in dbs:
        j = min(range(K), key=lambda j: (sum(per_db[x] for x in folds[j]), len(folds[j]), j))
        folds[j].append(d)
    out = {"k": K, "seed": SEED, "method": "greedy balance of task counts over whole databases",
           "folds": [{"fold": j, "databases": sorted(f), "n_databases": len(f),
                      "n_tasks": sum(per_db[d] for d in f),
                      "task_ids": sorted(t.instance_id for t in tasks if t.db in f)} for j, f in enumerate(folds)]}
    assert sorted(d for f in folds for d in f) == sorted(per_db)
    assert sum(x["n_tasks"] for x in out["folds"]) == len(tasks)
    (REPO / "manifests" / "folds_ldo.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for x in out["folds"]:
        print(x["fold"], x["n_tasks"], "tasks", x["n_databases"], "dbs", x["databases"])
