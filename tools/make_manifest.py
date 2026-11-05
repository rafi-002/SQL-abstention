"""Write manifests/spider2_lite_snapshot.json: pinned commit, download source, sha256 of every input file."""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import (DATA, DOCS_DIR, EVAL_FILE, GOLD_CSV_DIR, GOLD_SQL_DIR, LITE, REPO,  # noqa: E402
                          SQLITE_DIR, TASKS_FILE, load_tasks)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    tasks = load_tasks()
    commit = subprocess.run(["git", "-C", str(DATA / "spider2"), "log", "-1", "--format=%H %cI"],
                            capture_output=True, text=True, check=True).stdout.split()
    dbs = Counter(t.db for t in tasks)
    m = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_repo": "https://github.com/xlang-ai/Spider2",
        "commit": commit[0], "commit_date": commit[1],
        "checkout": "git clone --depth 1 --filter=blob:none --sparse; sparse-checkout set spider2-lite methods/spider-agent-lite",
        "sqlite_zip": {
            "url": "https://drive.usercontent.google.com/download?id=1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG&export=download",
            "sha256": sha(DATA / "local_sqlite.zip"), "bytes": (DATA / "local_sqlite.zip").stat().st_size,
            "note": "unzipped to data/sqlite; __MACOSX/ resource forks deleted",
        },
        "counts": {"sqlite_tasks": len(tasks), "databases": len(dbs),
                   "tasks_with_gold_sql": sum(1 for t in tasks if t.gold_sql() is not None),
                   "tasks_with_external_knowledge": sum(1 for t in tasks if t.external_knowledge)},
        "tasks_per_db": dict(sorted(dbs.items(), key=lambda x: (-x[1], x[0]))),
        "files": {},
    }
    files = [TASKS_FILE, EVAL_FILE, LITE / "evaluation_suite" / "evaluate.py", SQLITE_DIR / "local-map.jsonl"]
    files += sorted(GOLD_CSV_DIR.glob("local*.csv")) + sorted(GOLD_SQL_DIR.glob("local*.sql"))
    files += sorted({DOCS_DIR / t.external_knowledge for t in tasks if t.external_knowledge})
    files += sorted({t.db_path for t in tasks})
    for p in files:
        m["files"][p.relative_to(DATA).as_posix()] = sha(p)
    out = REPO / "manifests" / "spider2_lite_snapshot.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps({k: m[k] for k in ("commit", "commit_date", "counts")}, indent=1), len(m["files"]), "files hashed")
