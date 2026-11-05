"""Go/no-go analysis (M5): does standard risk-controlled abstention keep its promise on held-out databases?

Setting (fixed before looking at results; see notes/M5_analysis_plan.md):
- Unit = question. The agent's answer to a question is its MODAL result table over the 8 runs
  (no-answer runs never count as an answer; ties are broken by the canonical table key).
- Confidence = how many of the 8 runs produced that modal table (k = 0..8). Abstain if k < threshold.
- Promise: P( error rate among answered questions <= alpha ) >= 1 - delta.
- Calibration = Learn-then-Test (Angelopoulos et al., 2021) with fixed-sequence testing over thresholds
  k = 8, 7, ..., 1 (most to least conservative) and the exact binomial p-value
  p(k) = P[Bin(n_k, alpha) <= e_k], where n_k / e_k are the answered / wrong calibration questions.
  Stop at the first threshold with p > delta; use the loosest threshold that passed (none -> abstain on all).
  Under exchangeable questions this gives the finite-sample (alpha, delta) promise.
- Labels: --calib-labels and --test-labels are each 'benchmark' or 'audit' (audit = primary blind verdict,
  notes/M4_audit_protocol.md section 6). The go/no-go is calib=benchmark, test=audit.

Splits:
- ldo5:  the 5 fixed leave-database-out folds (manifests/folds_ldo.json).
- rdb:   R random database-held-out splits (whole databases; test set >= 27 questions).
- rq:    R random question splits with the same test sizes as rdb (exchangeable reference: the promise
         holds here by theory, so its violation rate is the noise baseline for small test sets).

Run: .venv/Scripts/python.exe tools/gonogo.py --calib-labels benchmark --test-labels benchmark
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import REPO  # noqa: E402

RUN = "m3_haiku55_t07"


def canon_key(df: pd.DataFrame) -> str:
    """Identical to tools/make_audit_packets.py, so modal answers map onto audit candidates."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].round(2)
    rows = sorted(map(tuple, d.astype(str).values.tolist()))
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def binom_cdf(e: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(e + 1))


def load_questions(run: str = RUN) -> dict:
    """qid -> {db, modal_key, k, bench_correct}. bench_correct is the benchmark label of the modal table."""
    run_dir = REPO / "runs" / run
    final = {}
    for line in (run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] != "harness_error":
            final[r["run_id"]] = r
    by_q = defaultdict(list)
    for rid, r in final.items():
        by_q[r["instance_id"]].append(r)
    out = {}
    for qid, recs in sorted(by_q.items()):
        tables = Counter()
        label_of = {}
        for r in recs:
            p = run_dir / "results" / f"{r['run_id']}.csv"
            if r["status"] != "submitted" or not p.exists():
                continue                                   # no answer / non-executing SQL: not a candidate answer
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            key = canon_key(df)
            tables[key] += 1
            label_of[key] = (r.get("benchmark_label") or {}).get("score") == 1
        if tables:
            k = max(tables.values())
            modal = min(key for key, c in tables.items() if c == k)
            out[qid] = {"db": recs[0]["db"], "modal_key": modal, "k": k, "bench_correct": label_of[modal]}
        else:
            out[qid] = {"db": recs[0]["db"], "modal_key": None, "k": 0, "bench_correct": None}
    return out


def load_audit_labels(qs: dict, annotator: str = "A1", path: Path | None = None) -> dict:
    """qid -> primary (blind) label (True/False) of the modal table, or None if unlabeled / excluded.
    `path` overrides the label file (e.g. runs/<judge>/labels_judge.csv or audit/labels_H1.csv)."""
    key = json.loads((REPO / "audit" / "key.json").read_text())
    verdict = {}
    with open(path or (REPO / "audit" / f"labels_{annotator}.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            verdict[(row["qid"], row["candidate_id"])] = row["blind_verdict"].strip().upper()
    out = {}
    for qid, q in qs.items():
        out[qid] = None
        if q["modal_key"] is None:
            continue
        for cid, c in key[qid].items():
            if c["canon"] == q["modal_key"]:
                v = verdict.get((qid, cid), "")
                out[qid] = True if v in ("CORRECT", "CORRECT_ALT") else False if v == "WRONG" else None
    return out


def modal_verdicts(qs: dict, path: Path) -> dict:
    """qid -> raw blind verdict string of the modal table ('' if not yet labelled). For completeness checks:
    CANNOT_JUDGE counts as labelled (it is excluded from risk, protocol section 6), an empty cell does not."""
    key = json.loads((REPO / "audit" / "key.json").read_text())
    raw = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw[(row["qid"], row["candidate_id"])] = row["blind_verdict"].strip().upper()
    out = {}
    for qid, q in qs.items():
        if q["modal_key"] is None:
            continue
        cid = next((c for c, v in key[qid].items() if v["canon"] == q["modal_key"]), None)
        out[qid] = raw.get((qid, cid), "")
    return out


def ltt_threshold(calib: list[tuple[int, bool]], alpha: float, delta: float) -> int | None:
    """Loosest k-threshold passing fixed-sequence LTT; None = abstain on everything."""
    best = None
    for thr in range(8, 0, -1):
        ans = [c for k, c in calib if k >= thr]
        n, e = len(ans), sum(1 for c in ans if not c)
        if n == 0:
            continue                                       # nothing answered at this level: trivially safe, keep going
        if binom_cdf(e, n, alpha) > delta:
            break
        best = thr
    return best


def evaluate(test: list[tuple[int, bool]], thr: int | None) -> tuple[int, int]:
    if thr is None:
        return 0, 0
    ans = [c for k, c in test if k >= thr]
    return len(ans), sum(1 for c in ans if not c)


def items(qs, labels, ids):
    """(k, correct) for questions with a usable label; questions with no answer have k = 0 (always abstain)."""
    out = []
    for q in ids:
        lab = labels.get(q)
        if qs[q]["k"] == 0:
            out.append((0, True))                          # never answered, label irrelevant
        elif lab is not None:
            out.append((qs[q]["k"], lab))
    return out


def run_split(qs, cal_lab, test_lab, calib_ids, test_ids, alpha, delta, correction=None):
    """correction(thr) -> additive PPI correction of the test risk (protocol v0.3 / plan section 7), or None."""
    thr = ltt_threshold(items(qs, cal_lab, calib_ids), alpha, delta)
    n, e = evaluate(items(qs, test_lab, test_ids), thr)
    risk = (e / n) if n else 0.0
    if correction is not None and n:
        risk = min(1.0, max(0.0, risk + correction(thr)))
    return {"thr": thr, "n_test": len(test_ids), "answered": n, "wrong": round(risk * n, 3),
            "risk": risk, "violated": bool(n and risk > alpha)}


def ppi_correction(qs, judge_lab, human_lab, sample):
    """Delta(thr) = mean over human-sample questions answered at thr of (human error - judge error)."""
    def corr(thr):
        if thr is None:
            return 0.0
        d = [(not human_lab[q]) - (not judge_lab[q]) for q in sample
             if qs[q]["k"] >= thr and human_lab.get(q) is not None and judge_lab.get(q) is not None]
        return sum(d) / len(d) if d else 0.0
    return corr


def summarise(rows, alpha):
    R = len(rows)
    ans = [r for r in rows if r["answered"]]
    return {"splits": R,
            "violation_rate": round(sum(r["violated"] for r in rows) / R, 3),
            "mean_test_risk_when_answering": round(sum(r["risk"] for r in ans) / len(ans), 3) if ans else None,
            "pooled_test_risk": round(sum(r["wrong"] for r in rows) / max(1, sum(r["answered"] for r in rows)), 3),
            "mean_coverage": round(sum(r["answered"] / r["n_test"] for r in rows) / R, 3),
            "share_abstain_all": round(sum(r["thr"] is None for r in rows) / R, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-labels", default="benchmark", choices=["benchmark", "audit"])
    ap.add_argument("--test-labels", default="benchmark", choices=["benchmark", "audit", "judge", "ppi"],
                    help="ppi = judge labels corrected with the human sample (protocol v0.3)")
    ap.add_argument("--judge-run", default="judge_gpt61sol")
    ap.add_argument("--human-file", default="audit/labels_H1.csv")
    ap.add_argument("--alphas", default="0.1,0.2,0.3,0.4")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    qs = load_questions()
    bench = {q: v["bench_correct"] for q, v in qs.items()}
    audit = load_audit_labels(qs) if "audit" in (a.calib_labels, a.test_labels) else None
    judge = human = correction = None
    need = [q for q, v in qs.items() if v["k"] > 0]
    if a.test_labels in ("judge", "ppi"):
        jpath = REPO / "runs" / a.judge_run / "labels_judge.csv"
        judge = load_audit_labels(qs, path=jpath)
        jraw = modal_verdicts(qs, jpath)
        miss = [q for q in need if not jraw.get(q)]
        print(f"judge CANNOT_JUDGE on modal answers (excluded): {sum(1 for q in need if jraw.get(q) == 'CANNOT_JUDGE')}")
        print(f"judge-labelled modal answers: {len(need) - len(miss)}/{len(need)}")
        if miss:
            sys.exit(f"judge labels incomplete ({len(miss)} missing, e.g. {miss[:3]})")
    if a.test_labels == "ppi":
        sample = json.loads((REPO / "manifests" / "audit_freeze.json").read_text())["human_sample"]
        human = load_audit_labels(qs, path=REPO / a.human_file)
        hraw = modal_verdicts(qs, REPO / a.human_file)
        miss = [q for q in sample if qs[q]["k"] > 0 and not hraw.get(q)]
        print(f"human-labelled sample modal answers: {len([q for q in sample if qs[q]['k'] > 0]) - len(miss)}/{len([q for q in sample if qs[q]['k'] > 0])}")
        if miss:
            sys.exit("human sample incomplete: per the protocol, no go/no-go numbers before it is complete")
        correction = ppi_correction(qs, judge, human, sample)
    cal_lab = bench if a.calib_labels == "benchmark" else audit
    test_lab = {"benchmark": bench, "audit": audit, "judge": judge, "ppi": judge}[a.test_labels]
    if audit is not None:
        n_lab = sum(v is not None for q, v in audit.items() if qs[q]["k"] > 0)
        n_need = sum(1 for q in qs.values() if q["k"] > 0)
        print(f"audited modal answers: {n_lab}/{n_need}")
        if n_lab < n_need:
            sys.exit("audit incomplete: per the protocol, no go/no-go numbers before the audit is complete")

    folds = json.loads((REPO / "manifests" / "folds_ldo.json").read_text())["folds"]
    all_ids = sorted(qs)
    dbs = sorted({v["db"] for v in qs.values()})
    by_db = defaultdict(list)
    for q, v in qs.items():
        by_db[v["db"]].append(q)
    rng = random.Random(a.seed)
    rdb_splits = []
    for _ in range(a.R):
        order = dbs[:]
        rng.shuffle(order)
        test = []
        for d in order:
            if len(test) >= 27:
                break
            test += by_db[d]
        rdb_splits.append(sorted(test))
    rq_splits = [sorted(rng.sample(all_ids, len(t))) for t in rdb_splits]

    k_dist = Counter(v["k"] for v in qs.values())
    base_err = sum(1 for v in qs.values() if v["k"] > 0 and v["bench_correct"] is False) / sum(1 for v in qs.values() if v["k"] > 0)
    report = {"run": RUN, "calib_labels": a.calib_labels, "test_labels": a.test_labels, "delta": a.delta,
              "R": a.R, "seed": a.seed, "n_questions": len(qs), "k_distribution": dict(sorted(k_dist.items())),
              "modal_answer_benchmark_error": round(base_err, 3), "results": {}}
    for alpha in [float(x) for x in a.alphas.split(",")]:
        thr_full = ltt_threshold(items(qs, cal_lab, all_ids), alpha, a.delta)
        ldo = [run_split(qs, cal_lab, test_lab, [q for q in all_ids if q not in set(f["task_ids"])], f["task_ids"], alpha, a.delta, correction)
               for f in folds]
        rdb = [run_split(qs, cal_lab, test_lab, [q for q in all_ids if q not in set(t)], t, alpha, a.delta, correction) for t in rdb_splits]
        rq = [run_split(qs, cal_lab, test_lab, [q for q in all_ids if q not in set(t)], t, alpha, a.delta, correction) for t in rq_splits]
        report["results"][str(alpha)] = {
            "threshold_on_all_135": thr_full,
            "ldo5_folds": ldo, "ldo5": summarise(ldo, alpha),
            "random_database_splits": summarise(rdb, alpha),
            "random_question_splits": summarise(rq, alpha),
        }
    if human is not None:
        sample = json.loads((REPO / "manifests" / "audit_freeze.json").read_text())["human_sample"]
        brng = random.Random(a.seed + 1)
        for alpha in [float(x) for x in a.alphas.split(",")]:
            pooled_n = pooled_e = 0
            for f in folds:
                thr = ltt_threshold(items(qs, cal_lab, [q for q in all_ids if q not in set(f["task_ids"])]), alpha, a.delta)
                n, e = evaluate(items(qs, human, [q for q in f["task_ids"] if q in set(sample)]), thr)
                pooled_n, pooled_e = pooled_n + n, pooled_e + e
            thr_full = report["results"][str(alpha)]["threshold_on_all_135"]
            boots = []
            if thr_full is not None:
                for _ in range(2000):
                    bs = [brng.choice(sample) for _ in sample]
                    boots.append(ppi_correction(qs, judge, human, bs)(thr_full))
                boots.sort()
            report["results"][str(alpha)]["human_sample_only_ldo5"] = {
                "answered": pooled_n, "wrong": pooled_e, "risk": round(pooled_e / pooled_n, 3) if pooled_n else None}
            report["results"][str(alpha)]["ppi_delta_at_full_threshold"] = {
                "delta": round(correction(thr_full), 3) if thr_full is not None else None,
                "ci90": [round(boots[int(0.05 * len(boots))], 3), round(boots[int(0.95 * len(boots)) - 1], 3)] if boots else None}
    txt = json.dumps(report, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    for alpha, r in report["results"].items():
        print(f"alpha={alpha} thr(all)={r['threshold_on_all_135']}")
        for name in ["ldo5", "random_database_splits", "random_question_splits"]:
            print(f"   {name:24s} {r[name]}")
    print("k distribution:", report["k_distribution"], "modal-answer benchmark error:", report["modal_answer_benchmark_error"])


if __name__ == "__main__":
    main()
