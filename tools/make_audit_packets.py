"""Build blinded audit packets for M4 (see notes/M4_audit_protocol.md).

For every question: the distinct agent result tables from an M3 run plus every gold CSV variant are merged
into anonymous candidates (A, B, C, ...) in a seeded random order. The packet shows the question, the
external-knowledge document, a schema summary and the candidate tables only (no grades, no run counts,
no SQL, nothing marking gold). The key (candidate -> agent runs / gold files) is written separately and
must stay out of the auditor's view until the unblinding step.

Outputs (audit/packets, audit/candidates and audit/key.json are git-ignored; they contain benchmark data):
  audit/packets/<qid>.md            one packet per question
  audit/candidates/<qid>/<cid>.csv  full candidate tables
  audit/key.json                    hidden key
  audit/labels_<ann>.csv            label template (one row per candidate; committed)
  audit/questions_<ann>.csv         question template (committed)
  manifests/audit_freeze.json       order, A2 subset, pilot, seeds, sha256 of key and packets (committed)

Run: .venv/Scripts/python.exe tools/make_audit_packets.py [run_name]
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import GOLD_CSV_DIR, REPO, load_tasks  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "m3_haiku55_t07"
SEED = 20261104
SHOW_ROWS = 50
AUDIT = REPO / "audit"
ANNOTATORS = ["A1"]                 # A2 template is written for the A2 subset only
A2_PER_FOLD = 8                     # ~30% of 135, stratified by fold
N_PILOT = 5


def canon_key(df: pd.DataFrame) -> str:
    """Same grouping as LOG L14: round to 2 decimals, sort rows, ignore column names."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].round(2)
    rows = sorted(map(tuple, d.astype(str).values.tolist()))
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names so naming style does not reveal the source (gold files often use UPPER_CASE)."""
    out = df.copy()
    out.columns = [re.sub(r"\s+", "_", str(c).strip()).lower() for c in out.columns]
    return out


def fmt(v) -> str:
    if isinstance(v, float):
        if pd.isna(v):
            return "NULL"
        return f"{v:.6g}" if abs(v) < 1e6 else f"{v:.6e}"
    s = str(v)
    return s.replace("|", "\\|").replace("\n", " ")[:120]


def md_table(df: pd.DataFrame, n: int) -> str:
    head = df.head(n)
    lines = ["| " + " | ".join(map(str, head.columns)) + " |", "|" + "---|" * len(head.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in head.itertuples(index=False)]
    return "\n".join(lines)


def schema_summary(db_path: Path) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out = []
        for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                                   "AND name NOT LIKE 'sqlite_%' ORDER BY name"):
            try:
                cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
                out.append(f"- `{name}`: {', '.join(cols)}")
            except sqlite3.OperationalError as e:      # e.g. a view referencing a missing column
                out.append(f"- `{name}`: (columns unavailable: {e})")
        return "\n".join(out)
    finally:
        con.close()


def gold_files(qid: str) -> list[Path]:
    pat = re.compile(rf"^{re.escape(qid)}(_[a-z])?\.csv$")
    return sorted(p for p in GOLD_CSV_DIR.iterdir() if pat.match(p.name))


def main():
    tasks = {t.instance_id: t for t in load_tasks()}
    run_dir = REPO / "runs" / RUN
    final = {}
    for line in (run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] != "harness_error":
            final[r["run_id"]] = r
    folds = json.loads((REPO / "manifests" / "folds_ldo.json").read_text())["folds"]

    (AUDIT / "packets").mkdir(parents=True, exist_ok=True)
    (AUDIT / "candidates").mkdir(parents=True, exist_ok=True)
    key, label_rows, stats = {}, [], {"questions": 0, "candidates": 0, "agent_only": 0, "gold_only": 0, "both": 0}

    for qid in sorted(tasks):
        t = tasks[qid]
        cands: dict[str, dict] = {}
        for rid, r in sorted(final.items()):
            if r["instance_id"] != qid or r["status"] != "submitted":
                continue
            p = run_dir / "results" / f"{rid}.csv"
            if not p.exists():
                continue                                  # non-executing SQL: wrong by construction, not audited
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            k = canon_key(df)
            c = cands.setdefault(k, {"df": df, "runs": [], "gold": []})
            c["runs"].append(rid)
        for g in gold_files(qid):
            df = pd.read_csv(g)
            k = canon_key(df)
            c = cands.setdefault(k, {"df": df, "runs": [], "gold": []})
            c["gold"].append(g.name)

        rng = random.Random(f"{SEED}:{qid}")
        order = sorted(cands)
        rng.shuffle(order)
        ids = [chr(ord("A") + i) if i < 26 else f"Z{i}" for i in range(len(order))]
        qdir = AUDIT / "candidates" / qid
        qdir.mkdir(parents=True, exist_ok=True)
        for old in qdir.glob("*.csv"):
            old.unlink()

        parts = [f"# {qid}  (database: {t.db})", "", "## Question", "", t.question.strip(), ""]
        if t.external_knowledge:
            parts += [f"## External knowledge ({t.external_knowledge})", "", t.knowledge_text().strip(), ""]
        parts += ["## Schema", "", schema_summary(t.db_path), "",
                  "## Step 1: write the intended reading(s) in questions_<you>.csv BEFORE reading the candidates", "",
                  "## Candidates", ""]
        key[qid] = {}
        for cid, k in zip(ids, order):
            c = cands[k]
            df = norm_columns(c["df"])
            df.to_csv(qdir / f"{cid}.csv", index=False)
            key[qid][cid] = {"agent_runs": c["runs"], "gold_files": c["gold"], "canon": k}
            stats["candidates"] += 1
            stats["both" if c["runs"] and c["gold"] else ("agent_only" if c["runs"] else "gold_only")] += 1
            shown = min(SHOW_ROWS, len(df))
            parts += [f"### Candidate {cid}", "", f"{len(df)} rows × {df.shape[1]} columns"
                      + (f" (first {shown} shown; full table: candidates/{qid}/{cid}.csv)" if len(df) > shown else ""), "",
                      md_table(df, SHOW_ROWS) if df.shape[1] else "_(empty result)_", ""]
            label_rows.append({"qid": qid, "candidate_id": cid})
        (AUDIT / "packets" / f"{qid}.md").write_text("\n".join(parts), encoding="utf-8")
        stats["questions"] += 1

    # --- order, A2 subset, pilot (all seeded, fixed before any label) ---
    rng = random.Random(SEED)
    fold_order = list(range(len(folds)))
    rng.shuffle(fold_order)
    q_order, a2 = [], []
    for f in fold_order:
        ids_f = sorted(folds[f]["task_ids"])
        rng.shuffle(ids_f)
        q_order += ids_f
        a2 += sorted(rng.sample(ids_f, A2_PER_FOLD))
    pilot = sorted(random.Random(SEED + 1).sample(sorted(tasks), N_PILOT))
    fallback_folds = sorted(random.Random(SEED + 2).sample(range(len(folds)), 2))

    key_bytes = json.dumps(key, indent=1, sort_keys=True).encode()
    (AUDIT / "key.json").write_bytes(key_bytes)
    rank = {q: i for i, q in enumerate(q_order)}

    def write_csv(path, fields, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    lab_fields = ["qid", "candidate_id", "blind_verdict", "alt_reading", "reason_code", "note",
                  "final_verdict", "change_reason", "annotator", "date"]
    q_fields = ["qid", "intended_reading", "readings", "ambiguous", "broken", "minutes", "annotator", "date"]
    rows_sorted = sorted(label_rows, key=lambda r: (rank[r["qid"]], r["candidate_id"]))
    write_csv(AUDIT / "labels_A1.csv", lab_fields, rows_sorted)
    write_csv(AUDIT / "questions_A1.csv", q_fields, [{"qid": q} for q in q_order])
    write_csv(AUDIT / "labels_A2.csv", lab_fields, [r for r in rows_sorted if r["qid"] in set(a2)])
    write_csv(AUDIT / "questions_A2.csv", q_fields, [{"qid": q} for q in q_order if q in set(a2)])

    packet_hashes = {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((AUDIT / "packets").glob("*.md"))}
    freeze = {
        "protocol": "notes/M4_audit_protocol.md", "protocol_commit": None, "run": RUN, "seed": SEED,
        "fold_order": fold_order, "question_order": q_order, "a2_subset": sorted(a2), "pilot": pilot,
        "fallback_folds": fallback_folds, "show_rows": SHOW_ROWS, "stats": stats,
        "key_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "packets_sha256": hashlib.sha256(json.dumps(packet_hashes, sort_keys=True).encode()).hexdigest(),
    }
    (REPO / "manifests" / "audit_freeze.json").write_text(json.dumps(freeze, indent=1), encoding="utf-8")
    print(json.dumps({k: freeze[k] for k in ["stats", "pilot", "fallback_folds", "fold_order", "key_sha256"]}, indent=1))
    print("A2 subset:", len(a2), "questions; label rows A1:", len(rows_sorted))


if __name__ == "__main__":
    main()
