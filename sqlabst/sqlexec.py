"""Read-only SQLite execution for the agent: timeout, row cap, text rendering of results."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field


@dataclass
class ExecResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)   # at most row_cap rows
    n_rows_seen: int = 0                              # rows fetched, capped at row_cap + 1
    truncated: bool = False
    error: str | None = None
    seconds: float = 0.0

    def render(self, max_chars: int = 6000) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        if not self.columns:
            return "OK (statement returned no columns)"
        lines = [" | ".join(self.columns)]
        lines += [" | ".join("NULL" if v is None else str(v) for v in r) for r in self.rows]
        more = f"\n... (more than {len(self.rows)} rows; showing the first {len(self.rows)})" if self.truncated else ""
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (output cut at %d characters)" % max_chars
        return f"{text}{more}\n({len(self.rows)}{'+' if self.truncated else ''} rows)"


def execute(db_path: str, sql: str, timeout_s: float = 30.0, row_cap: int = 50) -> ExecResult:
    """Run one statement on a read-only connection. Writes fail at the SQLite level (mode=ro)."""
    t0 = time.time()
    deadline = t0 + timeout_s
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(row_cap + 1) if cols else []
        trunc = len(rows) > row_cap
        return ExecResult(True, cols, [tuple(r) for r in rows[:row_cap]], len(rows), trunc, None, time.time() - t0)
    except sqlite3.OperationalError as e:
        msg = f"query exceeded {timeout_s:.0f}s time limit" if str(e) == "interrupted" else str(e)
        return ExecResult(False, error=msg, seconds=time.time() - t0)
    except Exception as e:
        return ExecResult(False, error=f"{type(e).__name__}: {e}", seconds=time.time() - t0)
    finally:
        con.close()
