"""Benchmark label = the OFFICIAL Spider 2.0-Lite evaluator, imported unmodified from the pinned snapshot.

evaluate.py has import-time side effects (it redirects stdout to ./log.txt and imports google.cloud.bigquery).
We stub bigquery (never used for local* tasks), import it from inside a scratch directory, and restore stdout.
The official code applies NO timeout to SQLite queries, so each scoring call runs in a child process with a
hang guard; a timeout is reported as status 'timeout', never silently as 'wrong'.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import sys
import tempfile
import types
from pathlib import Path

from .data import EVAL_FILE, GOLD_CSV_DIR, LITE, SQLITE_DIR, TASKS_FILE

EVALUATE_PY = LITE / "evaluation_suite" / "evaluate.py"
HANG_GUARD_S = 1800

_official = None


def official():
    """Import the official evaluate.py once per process."""
    global _official
    if _official is not None:
        return _official
    google = sys.modules.setdefault("google", types.ModuleType("google"))
    cloud = sys.modules.setdefault("google.cloud", types.ModuleType("google.cloud"))
    bq = types.ModuleType("google.cloud.bigquery")
    bq.Client = None
    sys.modules["google.cloud.bigquery"] = bq
    google.cloud = cloud
    cloud.bigquery = bq
    out, err, cwd = sys.stdout, sys.stderr, os.getcwd()
    side = tempfile.mkdtemp(prefix="evalimport_")
    try:
        os.chdir(side)
        spec = importlib.util.spec_from_file_location("spider2_official_evaluate", EVALUATE_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        tee = sys.stdout
        sys.stdout, sys.stderr = out, err
        os.chdir(cwd)
        if hasattr(tee, "close") and tee is not out:
            tee.close()
    mod._eval_standard = mod.load_jsonl_to_dict(str(EVAL_FILE))
    mod._metadata = mod.load_jsonl_to_dict(str(TASKS_FILE))
    _official = mod
    return mod


def _score_inprocess(instance_id: str, sql: str) -> dict:
    mod = official()
    with tempfile.TemporaryDirectory(prefix="score_") as tmp:
        tmp = Path(tmp)
        pred = tmp / "pred"
        pred.mkdir()
        (pred / f"{instance_id}.sql").write_text(sql, encoding="utf-8")
        res = mod.evaluate_single_sql_instance(
            instance_id, mod._eval_standard, mod._metadata, str(pred), str(GOLD_CSV_DIR),
            temp_dir=tmp / "work", result_csv_dir=None, sqlite_base_dir=SQLITE_DIR,
        )
    return {"status": "scored", "score": int(res["score"]), "error_info": res["error_info"]}


def _child(q, instance_id, sql):
    try:
        q.put(_score_inprocess(instance_id, sql))
    except Exception as e:  # pragma: no cover - reported, not hidden
        q.put({"status": "scorer_error", "score": None, "error_info": repr(e)})


def score_sql(instance_id: str, sql: str, hang_guard_s: int = HANG_GUARD_S) -> dict:
    """Benchmark label for one predicted SQL. score is 1/0, or None if not scored (timeout/scorer error)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(q, instance_id, sql), daemon=True)
    p.start()
    p.join(hang_guard_s)
    if p.is_alive():
        p.kill()
        p.join()
        return {"status": "timeout", "score": None, "error_info": f"scoring exceeded {hang_guard_s}s"}
    if q.empty():
        return {"status": "scorer_error", "score": None, "error_info": f"child exit code {p.exitcode}"}
    return q.get()


def score_gold_csv_against_itself(instance_id: str) -> list[int]:
    """Sanity check: each gold CSV, used as a prediction, must match. Uses the official comparator."""
    mod = official()
    paths, _ = mod.resolve_gold_paths(instance_id, str(GOLD_CSV_DIR))
    std = mod._eval_standard.get(instance_id, {})
    out = []
    for p in paths:
        pred = mod.pd.read_csv(p)
        golds = [mod.load_gold_csv(str(g)) for g in paths]
        out.append(mod.compare_multi_pandas_table(pred, golds, std.get("condition_cols"), std.get("ignore_order", False)))
    return out
