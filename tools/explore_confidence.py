"""EXPLORATORY (not part of the pre-registered go/no-go): can information already in the M3 logs predict
whether the modal answer is benchmark-correct better than k-of-8 agreement?

Uses benchmark labels only and reports aggregates only (no per-question lists), so the blind audit is unaffected.
Combined scores are fit by logistic regression *out of fold* over the 5 leave-database-out folds,
so every score is evaluated on databases it was not fit on.

Run: .venv/Scripts/python.exe tools/explore_confidence.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import REPO, load_tasks  # noqa: E402
from tools.gonogo import canon_key  # noqa: E402

RUN = "m3_haiku55_t07"


def col_sigs(df: pd.DataFrame) -> list[tuple]:
    """Each column as a sorted tuple of rounded values (column names ignored), like the official matcher."""
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            s = s.round(2)
        out.append(tuple(sorted(s.fillna(0).astype(str).tolist())))
    return out


def loose_equal(a, b) -> bool:
    """Same row count and every column of the narrower table appears in the wider one."""
    (sa, na), (sb, nb) = a, b
    if na != nb:
        return False
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    big_c = Counter(big)
    return all(big_c[c] > 0 for c in small) and len(small) > 0


def auroc(score, y) -> float:
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def logreg_fit(X, y, l2=1.0, iters=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z = np.c_[np.ones(len(Z)), Z]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * g
    return lambda Xn: 1 / (1 + np.exp(-np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ w))


def main():
    tasks = {t.instance_id: t for t in load_tasks()}
    run_dir = REPO / "runs" / RUN
    final = {}
    for line in (run_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] != "harness_error":
            final[r["run_id"]] = r
    by_q = defaultdict(list)
    for r in final.values():
        by_q[r["instance_id"]].append(r)

    rows = []
    for qid, recs in sorted(by_q.items()):
        tabs, labels, sigs = Counter(), {}, {}
        steps, errs, otoks, noans = [], [], [], 0
        for r in recs:
            obs = [o for s in r.get("steps", []) for o in (s.get("observations") or []) if isinstance(o, dict)]
            steps.append(r.get("n_steps") or 0)
            errs.append(sum(1 for o in obs if not o.get("ok", True)))
            otoks.append(r.get("output_tokens") or 0)
            p = run_dir / "results" / f"{r['run_id']}.csv"
            if r["status"] != "submitted" or not p.exists():
                noans += 1
                continue
            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
            k = canon_key(df)
            tabs[k] += 1
            labels[k] = (r.get("benchmark_label") or {}).get("score") == 1
            sigs[k] = (col_sigs(df), len(df))
        if not tabs:
            continue
        kmax = max(tabs.values())
        modal = min(t for t, c in tabs.items() if c == kmax)
        k_loose = sum(c for t, c in tabs.items() if t == modal or loose_equal(sigs[modal], sigs[t]))
        rows.append({
            "qid": qid, "db": tasks[qid].db, "y": labels[modal],
            "k": kmax, "k_loose": k_loose, "n_distinct": len(tabs), "noans": noans,
            "mean_steps": np.mean(steps), "mean_sql_errors": np.mean(errs), "mean_out_tokens": np.mean(otoks),
            "modal_rows": sigs[modal][1], "modal_cols": len(sigs[modal][0]),
            "has_doc": int(bool(tasks[qid].external_knowledge)), "q_len": len(tasks[qid].question),
        })
    D = pd.DataFrame(rows)
    y = D["y"].values.astype(float)
    feats = ["k", "k_loose", "n_distinct", "noans", "mean_steps", "mean_sql_errors", "mean_out_tokens",
             "modal_rows", "modal_cols", "has_doc", "q_len"]
    print(f"questions with an answer: {len(D)}; benchmark-correct modal answers: {int(y.sum())}")
    print("\nSingle features (AUROC; >0.5 = higher value predicts correct, <0.5 = lower value does):")
    single = {f: round(auroc(D[f], y), 3) for f in feats}
    for f, a in sorted(single.items(), key=lambda x: -abs(x[1] - 0.5)):
        print(f"  {f:16s} {a}")

    folds = json.loads((REPO / "manifests" / "folds_ldo.json").read_text())["folds"]
    fold_of = {q: i for i, f in enumerate(folds) for q in f["task_ids"]}
    D["fold"] = D["qid"].map(fold_of)
    combos = {"k only": ["k"], "k_loose only": ["k_loose"],
              "agreement (k, k_loose, n_distinct, noans)": ["k", "k_loose", "n_distinct", "noans"],
              "all features": feats}
    res = {}
    for name, cols in combos.items():
        oof = np.zeros(len(D))
        for f in range(len(folds)):
            tr, te = D["fold"] != f, D["fold"] == f
            model = logreg_fit(D.loc[tr, cols].values.astype(float), y[tr.values])
            oof[te.values] = model(D.loc[te, cols].values.astype(float))
        D[f"s_{name}"] = oof
        sel = {}
        order = np.argsort(-oof, kind="stable")
        for cov in (0.25, 0.5, 0.75):
            n = int(round(cov * len(D)))
            sel[cov] = round(1 - y[order[:n]].mean(), 3)
        res[name] = {"auroc_oof": round(auroc(oof, y), 3), "benchmark_error_at_coverage": sel}
    print("\nOut-of-fold (leave-database-out) combined scores:")
    for n, r in res.items():
        print(f"  {n:45s} AUROC {r['auroc_oof']}  error@cov25/50/75% {r['benchmark_error_at_coverage']}")
    out = {"run": RUN, "labels": "benchmark", "n": len(D), "single_feature_auroc": single, "combined": {
        n: {"auroc_oof": r["auroc_oof"], "error_at_coverage": {str(k): v for k, v in r["benchmark_error_at_coverage"].items()}}
        for n, r in res.items()}}
    (REPO / "results" / "explore_confidence.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
