# SQL-abstention

Does a risk-controlled abstention guarantee for text-to-SQL agents still hold under **unreliable benchmark labels** and **unseen databases**? Measured on Spider 2.0-Lite.

**Results: see [RESULTS.md](RESULTS.md).**

## Contents
- `sqlabst/`: data access, the official-scorer wrapper, the read-only SQL tool, the ReAct agent and the model clients
- `tools/`:
  - `run_trajectories.py`: agent runs
  - `summarize_run.py`
  - `gonogo.py`: the guarantee test
  - `llm_judge.py`: blind LLM re-grading
  - `make_audit_packets.py`
  - `explore_confidence.py`
  - `make_guarantee_figure.py`
  - plus manifest, fold and scorer checks
- `tests/`: offline tests (no API key needed)
- `configs/`: run configurations, each with a hard `spend_cap_usd`
- `manifests/`: pinned data snapshot (commit plus sha256 of every input), run records, leave-database-out folds, audit sampling
- `results/`: go/no-go outputs (JSON)
- `figures/`

## Setup
1. `uv venv --python 3.12 .venv` and `uv pip install --python .venv/Scripts/python.exe -r requirements.txt`
2. Data (not included; see `manifests/spider2_lite_snapshot.json` for the commit, URL and hashes): sparse-clone `xlang-ai/Spider2` into `data/spider2`, unzip `local_sqlite.zip` into `data/sqlite`, and delete `__MACOSX/`.
3. Check (needs the data from step 2): `.venv/Scripts/python.exe -m pytest -q tests`. Tests that need local audit files are skipped when those files are absent.
4. Paid runs: set `OPENROUTER_API_KEY` in the shell, then run a config.

Agent trajectories and judge outputs (`runs/`) are not included.
