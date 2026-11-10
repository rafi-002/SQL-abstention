"""M6: score x calibrator grid on the existing runs (pre-registered in notes/M6_score_calibrator_grid.md).

Free: uses only the existing 1,080 runs and the existing label files, no API calls. Reports aggregates only
(no per-question verdicts), so the blind human sample stays unbiased.

Scores   S1 k, S2 k_loose, S3 cluster entropy (answer iff H <= h).
Calibr.  C1 Learn-then-Test fixed sequence (exact binomial), C2 Clopper-Pearson + Bonferroni, C3 Hoeffding + Bonferroni.
Labels   key -> key; key -> each of 4 judge variants; judge -> same judge variant (exploratory).
Splits   5 fixed LDO folds, R random database splits, R random question splits (same construction as tools/gonogo.py).

Before the grid runs, S1 x C1 must reproduce results/gonogo_bench_bench.json and the Sol cells of
results/gonogo_two_judges_PROVISIONAL.json exactly; otherwise the script stops.

Run: .venv/Scripts/python.exe tools/score_grid.py   (writes results/score_grid.json)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from math import comb, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import REPO  # noqa: E402
from tools.explore_confidence import auroc, col_sigs, loose_equal  # noqa: E402
from tools.gonogo import canon_key, summarise  # noqa: E402

RUN = "m3_haiku55_t07"
ALPHAS = [0.1, 0.2, 0.3, 0.4]
DELTA = 0.1
H_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.1]
JUDGES = {"sol": "judge_gpt61sol", "gem": "judge_gemini38flash"}
AUDIT_DIR = "audit"
AGENTS = {"haiku": (RUN, JUDGES, "audit", "gonogo_bench_bench.json"),
          "glm53": ("m7_glm53_full", {"sol": "judge_gpt61sol_glm53", "gem": "judge_gemini38flash_glm53"}, "audit_glm53", "gonogo_glm53_bench_bench.json")}


@lru_cache(maxsize=None)
def cdf(e: int, n: int, a: float) -> float:
    return sum(comb(n, i) * a**i * (1 - a) ** (n - i) for i in range(e + 1))


def select(nc, ec, alpha, delta, cal, m):
    """Index (0 = strictest) of the loosest threshold the calibrator accepts, or None (= abstain on everything)."""
    best = None
    for j in range(m):
        n, e = nc[j], ec[j]
        if n == 0:
            continue                                   # nothing answered at this level: trivially safe, skip
        if cal == "C1":
            if cdf(e, n, alpha) > delta:
                break                                  # fixed sequence: stop at the first failure
            best = j
        elif cal == "C2":
            if cdf(e, n, alpha) <= delta / m:
                best = j
        else:                                          # C3
            if e / n + sqrt(log(m / delta) / (2 * n)) <= alpha:
                best = j
    return best


def load_questions():
    run_dir = REPO / "runs" / RUN
    final = {}
    for line in (run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] != "harness_error":
            final[r["run_id"]] = r
    by_q = defaultdict(list)
    for r in final.values():
        by_q[r["instance_id"]].append(r)
    qs = {}
    for qid, recs in sorted(by_q.items()):
        tabs, label_of, sigs, noans = Counter(), {}, {}, 0
        for r in recs:
            p = run_dir / "results" / f"{r['run_id']}.csv"
            if r["status"] != "submitted" or not p.exists():
                noans += 1
                continue
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            key = canon_key(df)
            tabs[key] += 1
            label_of[key] = (r.get("benchmark_label") or {}).get("score") == 1
            sigs[key] = (col_sigs(df), len(df))
        if not tabs:
            qs[qid] = {"db": recs[0]["db"], "modal_key": None, "k": 0, "k_loose": 0, "H": float("inf"), "bench": None}
            continue
        k = max(tabs.values())
        modal = min(t for t, c in tabs.items() if c == k)
        k_loose = sum(c for t, c in tabs.items() if t == modal or loose_equal(sigs[modal], sigs[t]))
        counts = list(tabs.values()) + ([noans] if noans else [])
        tot = sum(counts)
        H = -sum(c / tot * log(c / tot) for c in counts)
        qs[qid] = {"db": recs[0]["db"], "modal_key": modal, "k": k, "k_loose": k_loose, "H": abs(H),
                   "bench": label_of[modal]}
    return qs


def judge_labels(qs, run, strict):
    """qid -> True/False/None for the modal table. lenient: CORRECT+CORRECT_ALT correct; strict: only CORRECT."""
    key = json.loads((REPO / AUDIT_DIR / "key.json").read_text())
    verdict = {}
    with open(REPO / "runs" / run / "labels_judge.csv", encoding="utf-8") as fh:
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
                if strict:
                    out[qid] = True if v == "CORRECT" else False if v in ("WRONG", "CORRECT_ALT") else None
                else:
                    out[qid] = True if v in ("CORRECT", "CORRECT_ALT") else False if v == "WRONG" else None
    return out


def make_splits(qs, seed, R):
    """Identical construction (and RNG order) to tools/gonogo.py."""
    folds = json.loads((REPO / "manifests" / "folds_ldo.json").read_text())["folds"]
    ids = sorted(qs)
    idx = {q: i for i, q in enumerate(ids)}
    dbs = sorted({v["db"] for v in qs.values()})
    by_db = defaultdict(list)
    for q, v in qs.items():
        by_db[v["db"]].append(q)
    rng = random.Random(seed)
    rdb = []
    for _ in range(R):
        order = dbs[:]
        rng.shuffle(order)
        test = []
        for d in order:
            if len(test) >= 27:
                break
            test += by_db[d]
        rdb.append(sorted(test))
    rq = [sorted(rng.sample(ids, len(t))) for t in rdb]

    def mat(lists):
        S = np.zeros((len(lists), len(ids)))
        for r, t in enumerate(lists):
            for q in t:
                S[r, idx[q]] = 1.0
        return S
    return {"ldo5": mat([f["task_ids"] for f in folds]), "rdb": mat(rdb), "rq": mat(rq)}


def run_cell(A, m, cal_lab, te_lab, splits, alpha, cal):
    """Return {split_type: summary} for one (score thresholds A, calibrator, label pair, alpha)."""
    out = {}
    for name, (nc, ec, nt, et, nsz) in splits.items():
        rows = []
        for r in range(nc.shape[0]):
            j = select(nc[r].tolist(), ec[r].tolist(), alpha, DELTA, cal, m)
            if j is None:
                rows.append({"thr": None, "n_test": int(nsz[r]), "answered": 0, "wrong": 0, "risk": 0.0, "violated": False})
            else:
                n, e = int(nt[r, j]), int(et[r, j])
                risk = e / n if n else 0.0
                rows.append({"thr": j, "n_test": int(nsz[r]), "answered": n, "wrong": e, "risk": risk,
                             "violated": bool(n and risk > alpha)})
        out[name] = summarise(rows, alpha)
    return out


def counts_for(A, cal_use, cal_wrong, te_use, te_wrong, S):
    Au = (A & cal_use[:, None]).astype(float)
    Ew = (A & cal_wrong[:, None]).astype(float)
    Tu = (A & te_use[:, None]).astype(float)
    Tw = (A & te_wrong[:, None]).astype(float)
    nc = np.rint(Au.sum(0)[None, :] - S @ Au).astype(int)
    ec = np.rint(Ew.sum(0)[None, :] - S @ Ew).astype(int)
    nt = np.rint(S @ Tu).astype(int)
    et = np.rint(S @ Tw).astype(int)
    return nc, ec, nt, et, S.sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--agent", default="haiku", choices=sorted(AGENTS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global RUN, JUDGES, AUDIT_DIR
    RUN, JUDGES, AUDIT_DIR, bench_file = AGENTS[a.agent]
    if a.out is None:
        a.out = str(REPO / "results" / ("score_grid.json" if a.agent == "haiku" else f"score_grid_{a.agent}.json"))

    qs = load_questions()
    ids = sorted(qs)
    Q = len(ids)
    k = np.array([qs[q]["k"] for q in ids])
    kl = np.array([qs[q]["k_loose"] for q in ids])
    H = np.array([qs[q]["H"] for q in ids])
    has = k > 0
    scores = {
        "S1": ("k", list(range(8, 0, -1)), np.stack([(k >= t) & has for t in range(8, 0, -1)], 1), k.astype(float)),
        "S2": ("k_loose", list(range(8, 0, -1)), np.stack([(kl >= t) & has for t in range(8, 0, -1)], 1), kl.astype(float)),
        "S3": ("cluster entropy", H_GRID, np.stack([(H <= h + 1e-9) & has for h in H_GRID], 1), -H),
    }
    label_sets = {"key": [qs[q]["bench"] for q in ids]}
    for jn, run in JUDGES.items():
        for strict in (False, True):
            lab = judge_labels(qs, run, strict)
            label_sets[f"{jn}_{'strict' if strict else 'lenient'}"] = [lab[q] for q in ids]
    L = {}
    for name, labs in label_sets.items():
        use = np.array([(l is not None) and h for l, h in zip(labs, has)])
        wrong = np.array([(l is False) and h for l, h in zip(labs, has)])
        correct = np.array([(l is True) and h for l, h in zip(labs, has)])
        L[name] = {"use": use, "wrong": wrong, "correct": correct}
        print(f"labels {name}: usable modal answers {int(use.sum())}, wrong {int(wrong.sum())}", flush=True)

    split_mats = make_splits(qs, a.seed, a.R)

    def cell(score, cal, cl, tl, alpha):
        _, grid, A, _ = scores[score]
        c, t = L[cl], L[tl]
        sp = {n: counts_for(A, c["use"], c["wrong"], t["use"], t["wrong"], S) for n, S in split_mats.items()}
        return run_cell(A, len(grid), cl, tl, sp, alpha, cal)

    # ---- sanity: S1 x C1 must reproduce the existing results exactly --------------------------------------
    bench = json.loads((REPO / "results" / bench_file).read_text())["results"]
    bad = []
    for alpha in ALPHAS:
        got = cell("S1", "C1", "key", "key", alpha)
        for ours, theirs in (("ldo5", "ldo5"), ("rdb", "random_database_splits"), ("rq", "random_question_splits")):
            if got[ours] != bench[str(alpha)][theirs]:
                bad.append(("key->key", alpha, ours, got[ours], bench[str(alpha)][theirs]))
    two = json.loads((REPO / "results" / "gonogo_two_judges_PROVISIONAL.json").read_text())
    for var, jkey, tag in () if a.agent != "haiku" else (("sol_lenient", "sol_lenient", "lenient"), ("sol_strict", "sol_strict", "strict"),
                           ("gem_lenient", "gemini_lenient", "lenient"), ("gem_strict", "gemini_strict", "strict")):
        for alpha in (0.2, 0.3, 0.4):
            blk = two[jkey][f"a={alpha}"]
            for pair, cl in ((f"bench->{tag}", "key"), (f"judge->{tag}", var)):
                got = cell("S1", "C1", cl, var, alpha)
                for ours, theirs in (("ldo5", "ldo"), ("rdb", "rdb")):
                    if got[ours] != blk[pair][theirs]:
                        bad.append((f"{var} {pair}", alpha, ours, got[ours], blk[pair][theirs]))
    if bad:
        for b in bad[:6]:
            print("MISMATCH", b)
        sys.exit(f"sanity check failed: {len(bad)} mismatches. Fix the code before reading any grid result.")
    print("sanity check passed: S1 x C1 reproduces gonogo_bench_bench.json and the two-judge cells exactly.", flush=True)
    if a.check_only:
        return

    # ---- the grid ----------------------------------------------------------------------------------------
    combos = [("key", "key")] + [("key", v) for v in L if v != "key"] + [(v, v) for v in L if v != "key"]
    cells = {}
    for score in scores:
        for cal in ("C1", "C2", "C3"):
            for cl, tl in combos:
                for alpha in ALPHAS:
                    res = cell(score, cal, cl, tl, alpha)
                    rdb, rq, ldo = res["rdb"], res["rq"], res["ldo5"]
                    vacuous = rdb["mean_coverage"] < 0.10
                    brk = (rdb["violation_rate"] - rq["violation_rate"] > 0.1 and rdb["violation_rate"] > DELTA) \
                        or rdb["pooled_test_risk"] > alpha + 0.05
                    cells[f"{score}|{cal}|{cl}->{tl}|a={alpha}"] = {**res, "vacuous": vacuous, "breaks": bool(brk and not vacuous)}
        print(f"done {score}", flush=True)

    auc = {}
    for score, (name, _, _, conf) in scores.items():
        auc[score] = {}
        for lname, l in L.items():
            m_ = l["use"]
            auc[score][lname] = round(auroc(conf[m_], l["correct"][m_]), 3)

    report = {"run": RUN, "R": a.R, "seed": a.seed, "delta": DELTA, "alphas": ALPHAS, "H_grid": H_GRID,
              "note": "pre-registered in notes/M6_score_calibrator_grid.md; aggregates only",
              "auroc_confidence_vs_label": auc, "cells": cells}
    Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out} ({len(cells)} cells)", flush=True)


if __name__ == "__main__":
    main()
