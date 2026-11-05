"""Show, for failing gold-SQL tasks, the gold SQL output vs the gold CSV(s) (shape, columns, head)."""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sqlabst.data import GOLD_CSV_DIR, load_tasks  # noqa: E402
from sqlabst.scorer import official  # noqa: E402

ids = sys.argv[1:]
tasks = {t.instance_id: t for t in load_tasks()}
mod = official()
for i in ids:
    t = tasks[i]
    con = sqlite3.connect(f"file:{t.db_path}?mode=ro", uri=True)
    df = pd.read_sql_query(t.gold_sql(), con)
    con.close()
    std = mod._eval_standard[i]
    print("=" * 30, i, "condition_cols", std.get("condition_cols"), "ignore_order", std.get("ignore_order"))
    print("GOLD-SQL OUTPUT", df.shape, list(df.columns)); print(df.head(4).to_string()[:700])
    for p in sorted(GOLD_CSV_DIR.glob(f"{i}*.csv")):
        g = pd.read_csv(p)
        print("--", p.name, g.shape, list(g.columns)); print(g.head(4).to_string()[:700])
