"""Figure for RESULTS.md: error per leave-database-out fold at alpha=0.4, benchmark vs LLM-judge grading."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
SRC = str(REPO / "results")
DST = str(REPO / "figures" / "guarantee_by_fold.png")
b = [f["risk"] for f in json.load(open(SRC + "/gonogo_bench_bench.json"))["results"]["0.4"]["ldo5_folds"]]
j = [f["risk"] for f in json.load(open(SRC + "/gonogo_bench_judge_PROVISIONAL.json"))["results"]["0.4"]["ldo5_folds"]]
x = range(1, 6)
w = 0.38
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)
ax.bar([i - w / 2 for i in x], b, w, label="graded by benchmark answer key", color="#d08770")
ax.bar([i + w / 2 for i in x], j, w, label="graded by LLM judge (provisional)", color="#5e81ac")
ax.axhline(0.4, ls="--", color="k", lw=1)
ax.text(3.5, 0.41, "promise: at most 40% wrong", ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels([f"fold {i - 1}\n(unseen DBs)" for i in x], fontsize=8)
ax.set_ylabel("error among answered questions")
ax.set_ylim(0, 0.6)
ax.set_title("alpha = 0.4: calibrated on benchmark labels, tested on held-out databases", fontsize=9)
ax.legend(fontsize=8, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
fig.savefig(DST)
print("saved", DST)
