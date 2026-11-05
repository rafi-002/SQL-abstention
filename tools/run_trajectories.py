"""Trajectory logger / runner.

    python tools/run_trajectories.py configs/<name>.json

Writes runs/<run_name>/:
  config.json             copy of the config + snapshot commit + prompt version
  trajectories.jsonl      one record per (task, run index); appended by the main thread only
  results/<run_id>.csv    full result table of the submitted SQL (for later label audits)
  console.log             everything printed

Safety: a hard spend cap. A new run is started only if   spent + in_flight * max_run_cost <= cap,
where max_run_cost is an upper bound from the token budget and prices. Resume: finished (task, run) pairs are
skipped. Harness errors are recorded as status 'harness_error' and never counted as wrong answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.agent import PROMPT_VERSION, run_agent  # noqa: E402
from sqlabst.data import REPO, load_tasks  # noqa: E402
from sqlabst.llm import GoldClient, OpenRouterClient  # noqa: E402
from sqlabst.scorer import score_sql  # noqa: E402

LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        with LOCK:
            self.out.write(s)
            self.f.write(s)
            self.f.flush()

    def flush(self):
        self.out.flush()


def make_client(cfg, task):
    a = cfg["agent_llm"]
    if a["provider"] == "gold":
        return GoldClient(task.gold_sql())
    if a["provider"] == "openrouter":
        return OpenRouterClient(a["model"], a["price_in_per_m"], a["price_out_per_m"], a.get("temperature", 0.7),
                                a.get("max_output_tokens", 2048), a.get("retries", 5), a.get("backoff_cap_s", 60),
                                a.get("provider_order"))
    raise ValueError(a["provider"])


def max_run_cost(cfg) -> float:
    a, ag = cfg["agent_llm"], cfg["agent"]
    if a["provider"] != "openrouter":
        return 0.0
    # token budget is checked after each call, so one extra call can overshoot it
    worst_in = ag["token_budget"] + ag["token_budget"]
    worst_out = ag["max_steps"] * a.get("max_output_tokens", 2048)
    return worst_in * a["price_in_per_m"] / 1e6 + worst_out * a["price_out_per_m"] / 1e6


def save_result_csv(task, sql, path, row_limit=100_000):
    import pandas as pd
    try:
        con = sqlite3.connect(f"file:{task.db_path}?mode=ro", uri=True)
        try:
            df = pd.read_sql_query(sql, con)
        finally:
            con.close()
        df.head(row_limit).to_csv(path, index=False)
        return {"saved": True, "n_rows": int(len(df)), "n_cols": int(df.shape[1]), "cut": len(df) > row_limit}
    except Exception as e:
        return {"saved": False, "error": repr(e)[:500]}


def one_run(cfg, task, run_idx, out_dir):
    run_id = f"{task.instance_id}__r{run_idx:02d}"
    seed = cfg.get("seed_base", 0) + run_idx
    rec = {"run_id": run_id, "instance_id": task.instance_id, "db": task.db, "run_idx": run_idx, "seed": seed,
           "provider": cfg["agent_llm"]["provider"], "model": cfg["agent_llm"].get("model"),
           "temperature": cfg["agent_llm"].get("temperature"), "started_utc": now()}
    t0 = time.time()
    try:
        traj = run_agent(task, make_client(cfg, task), seed=seed, **cfg["agent"])
        rec.update(traj)
        if traj["final_sql"]:
            rec["result_csv"] = save_result_csv(task, traj["final_sql"], out_dir / "results" / f"{run_id}.csv")
            rec["benchmark_label"] = score_sql(task.instance_id, traj["final_sql"])
        else:
            rec["benchmark_label"] = {"status": "no_submission", "score": 0, "error_info": traj["status"]}
    except Exception as e:
        rec.update({"status": "harness_error", "error": repr(e)[:2000], "traceback": traceback.format_exc()[-4000:]})
        rec.setdefault("cost_usd", 0.0)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["ended_utc"] = now()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    out = REPO / "runs" / cfg["run_name"]
    (out / "results").mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = Tee(out / "console.log")
    snap = json.loads((REPO / "manifests" / "spider2_lite_snapshot.json").read_text(encoding="utf-8"))
    meta = dict(cfg, snapshot_commit=snap["commit"], prompt_version=PROMPT_VERSION,
                config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(), written_utc=now())
    (out / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    tasks = load_tasks()
    if cfg.get("task_ids"):
        want = set(cfg["task_ids"])
        tasks = [t for t in tasks if t.instance_id in want]
    if cfg.get("only_with_gold_sql"):
        tasks = [t for t in tasks if t.gold_sql() is not None]
    jobs = [(t, k) for t in tasks for k in range(cfg["runs_per_task"])]

    log = out / "trajectories.jsonl"
    done, spent = set(), 0.0
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            spent += r.get("cost_usd", 0.0) or 0.0
            if r.get("status") != "harness_error":
                done.add(r["run_id"])
    jobs = [(t, k) for t, k in jobs if f"{t.instance_id}__r{k:02d}" not in done]
    cap, mrc, workers = cfg["spend_cap_usd"], max_run_cost(cfg), cfg.get("workers", 1)
    print(f"[{now()}] {cfg['run_name']}: {len(jobs)} runs to do, {len(done)} already done, spent so far "
          f"${spent:.4f}, cap ${cap:.2f}, max cost per run ${mrc:.4f}, workers {workers}")

    pending, i, n_done = {}, 0, 0
    stop_reason = None
    with ThreadPoolExecutor(max_workers=workers) as ex, open(log, "a", encoding="utf-8") as f:
        while i < len(jobs) or pending:
            while i < len(jobs) and len(pending) < workers and stop_reason is None:
                if spent + (len(pending) + 1) * mrc > cap:
                    stop_reason = f"spend cap: spent ${spent:.4f} + in-flight bound would exceed ${cap:.2f}"
                    break
                t, k = jobs[i]
                pending[ex.submit(one_run, cfg, t, k, out)] = (t, k)
                i += 1
            if not pending:
                break
            fin, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fu in fin:
                pending.pop(fu)
                rec = fu.result()
                spent += rec.get("cost_usd", 0.0) or 0.0
                n_done += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                lab = rec.get("benchmark_label", {})
                print(f"[{now()}] {rec['run_id']} {rec['status']} label={lab.get('score')} ({lab.get('status')}) "
                      f"steps={rec.get('n_steps')} tok={rec.get('input_tokens', 0)}+{rec.get('output_tokens', 0)} "
                      f"${rec.get('cost_usd', 0):.4f} total ${spent:.4f} [{n_done}/{len(jobs)}]")
            if stop_reason and not pending:
                break
    print(f"[{now()}] finished: {n_done} runs this session, total spent ${spent:.4f}"
          + (f"; STOPPED EARLY: {stop_reason}" if stop_reason else ""))


if __name__ == "__main__":
    main()
