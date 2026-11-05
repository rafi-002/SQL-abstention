"""Data access for the Spider 2.0-Lite SQLite subset (pinned snapshot, see manifests/)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
LITE = DATA / "spider2" / "spider2-lite"
SQLITE_DIR = DATA / "sqlite"                      # unzipped local_sqlite.zip
TASKS_FILE = LITE / "spider2-lite.jsonl"
EVAL_FILE = LITE / "evaluation_suite" / "gold" / "spider2lite_eval.jsonl"
GOLD_CSV_DIR = LITE / "evaluation_suite" / "gold" / "exec_result"
GOLD_SQL_DIR = LITE / "evaluation_suite" / "gold" / "sql"
DOCS_DIR = LITE / "resource" / "documents"


@dataclass(frozen=True)
class Task:
    instance_id: str
    db: str
    question: str
    external_knowledge: str | None

    @property
    def db_path(self) -> Path:
        return SQLITE_DIR / f"{self.db}.sqlite"

    def knowledge_text(self) -> str | None:
        if not self.external_knowledge:
            return None
        return (DOCS_DIR / self.external_knowledge).read_text(encoding="utf-8")

    def gold_sql(self) -> str | None:
        p = GOLD_SQL_DIR / f"{self.instance_id}.sql"
        return p.read_text(encoding="utf-8") if p.exists() else None


def load_tasks(tasks_file: Path = TASKS_FILE) -> list[Task]:
    """All SQLite ('local*') tasks, in file order."""
    out = []
    with open(tasks_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["instance_id"].startswith("local"):
                out.append(Task(r["instance_id"], r["db"], r["question"], r.get("external_knowledge")))
    return out


@lru_cache(maxsize=None)
def schema_ddl(db_path: str) -> str:
    """CREATE statements read from the database file itself (tables and views)."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY type, name"
        ).fetchall()
    finally:
        con.close()
    return "\n\n".join(r[0].strip() + ";" for r in rows)
