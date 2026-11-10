"""Leave-one-database-out check of a judge-calibrated cell (GLM-5.3, alpha 0.2, S1 x LTT, lenient labels).
Run: .venv/Scripts/python.exe tools/per_db_check.py results/glm53_per_db_alpha02.json"""
import sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import score_grid as g
g.RUN, g.JUDGES, g.AUDIT_DIR = g.AGENTS["glm53"][0], g.AGENTS["glm53"][1], "audit_glm53"
qs = g.load_questions()
alpha, delta = 0.2, 0.1
T = list(range(8, 0, -1))
out = {}
for name, run in g.JUDGES.items():
    lab = g.judge_labels(qs, run, False)
    use = {q: lab[q] for q in qs if lab[q] is not None and qs[q]["k"] > 0}
    dbs = sorted({qs[q]["db"] for q in use})
    rows = []
    tot_n = tot_e = 0
    for d in dbs:
        cal = [q for q in use if qs[q]["db"] != d]
        te = [q for q in use if qs[q]["db"] == d]
        nc = [sum(qs[q]["k"] >= t for q in cal) for t in T]
        ec = [sum(qs[q]["k"] >= t and use[q] is False for q in cal) for t in T]
        j = g.select(nc, ec, alpha, delta, "C1", 8)
        if j is None:
            rows.append((d, None, len(te), 0, 0)); continue
        thr = T[j]
        ans = [q for q in te if qs[q]["k"] >= thr]
        wr = sum(use[q] is False for q in ans)
        rows.append((d, thr, len(te), len(ans), wr)); tot_n += len(ans); tot_e += wr
    viol = [r for r in rows if r[3] and r[4] / r[3] > alpha]
    print(f"== {name} lenient, leave-one-database-out, alpha {alpha}: {len(dbs)} databases; thresholds chosen:",
          collections.Counter(r[1] for r in rows))
    print(f"   answered {tot_n}, wrong {tot_e}, pooled risk {tot_e/max(tot_n,1):.3f}; databases with test risk > alpha: {len(viol)} of {sum(1 for r in rows if r[3])} that answered")
    for r in sorted(viol, key=lambda r: -r[4]/r[3])[:8]:
        print(f"   {r[0]:28s} thr k>={r[1]} questions {r[2]} answered {r[3]} wrong {r[4]}")
    out[name] = {"rows": rows}
json.dump(out, open(sys.argv[1], "w"), indent=1)
