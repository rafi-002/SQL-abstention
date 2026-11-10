"""Tests for the M6 calibrators (tools/score_grid.py): C1 must equal the existing LTT, C2/C3 must only accept
thresholds that satisfy their own inequality, and an empty calibration set abstains."""
import random
import sys
from math import log, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gonogo import ltt_threshold  # noqa: E402
from tools.score_grid import cdf, select  # noqa: E402


def random_calib(rng, n=60):
    return [(rng.randint(1, 8), rng.random() > 0.35) for _ in range(n)]


def counts(calib, m=8):
    nc, ec = [], []
    for thr in range(8, 8 - m, -1):
        ans = [c for k, c in calib if k >= thr]
        nc.append(len(ans))
        ec.append(sum(1 for c in ans if not c))
    return nc, ec


def test_c1_matches_existing_ltt():
    rng = random.Random(1)
    for _ in range(200):
        calib = random_calib(rng, rng.randint(5, 80))
        for alpha in (0.1, 0.2, 0.3, 0.4):
            nc, ec = counts(calib)
            j = select(nc, ec, alpha, 0.1, "C1", 8)
            thr = ltt_threshold(calib, alpha, 0.1)
            assert (None if j is None else 8 - j) == thr


def test_c2_and_c3_accept_only_valid_thresholds():
    rng = random.Random(2)
    for _ in range(200):
        calib = random_calib(rng, rng.randint(5, 80))
        nc, ec = counts(calib)
        for alpha in (0.2, 0.3, 0.4):
            j = select(nc, ec, alpha, 0.1, "C2", 8)
            if j is not None:
                assert cdf(ec[j], nc[j], alpha) <= 0.1 / 8
            j = select(nc, ec, alpha, 0.1, "C3", 8)
            if j is not None:
                assert ec[j] / nc[j] + sqrt(log(8 / 0.1) / (2 * nc[j])) <= alpha


def test_empty_calibration_abstains():
    for cal in ("C1", "C2", "C3"):
        assert select([0] * 8, [0] * 8, 0.3, 0.1, cal, 8) is None


def test_fixed_sequence_stops_at_first_failure_but_bonferroni_does_not():
    # No errors, but the strictest thresholds answer few questions. C1 stops at the first failure (n = 3 gives
    # p = 0.8**3 = 0.51 > 0.1) and accepts nothing; C2 needs P <= 0.1 / 8, i.e. n >= 20, and accepts index 7 (n = 24).
    nc, ec = [3, 6, 9, 12, 15, 18, 21, 24], [0] * 8
    assert select(nc, ec, 0.2, 0.1, "C1", 8) is None
    assert select(nc, ec, 0.2, 0.1, "C2", 8) == 7
