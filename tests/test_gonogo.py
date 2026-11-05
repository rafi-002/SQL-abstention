"""Checks for tools/gonogo.py on synthetic data (no key, no spend)."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.gonogo import binom_cdf, evaluate, ltt_threshold, ppi_correction  # noqa: E402


def test_ppi_correction_measures_judge_bias():
    qs = {f"q{i}": {"k": 8 if i < 6 else 2} for i in range(10)}
    judge = {f"q{i}": True for i in range(10)}                  # judge says everything is correct
    human = {f"q{i}": i not in (0, 1) for i in range(10)}        # humans find q0, q1 wrong
    corr = ppi_correction(qs, judge, human, sorted(qs))
    assert abs(corr(8) - 2 / 6) < 1e-12                          # among answered at k>=8: 2 of 6 missed by judge
    assert abs(corr(1) - 2 / 10) < 1e-12
    assert corr(None) == 0.0


def test_binom_cdf_edges():
    assert abs(binom_cdf(10, 10, 0.3) - 1.0) < 1e-12
    assert abs(binom_cdf(0, 3, 0.5) - 0.125) < 1e-12


def test_all_correct_uses_loosest_threshold():
    calib = [(k, True) for k in range(1, 9) for _ in range(40)]   # 0.9**40 < 0.1, so even k=8 alone passes
    assert ltt_threshold(calib, 0.1, 0.1) == 1


def test_all_wrong_abstains():
    calib = [(8, False)] * 50
    thr = ltt_threshold(calib, 0.1, 0.1)
    assert thr is None and evaluate(calib, thr) == (0, 0)


def test_ltt_controls_population_risk_on_exchangeable_data():
    """Population selective risk of the chosen threshold must exceed alpha in at most ~delta of draws."""
    rng = random.Random(0)
    p_correct = {k: 0.3 + 0.08 * k for k in range(1, 9)}           # higher agreement -> more often right
    alpha, delta = 0.2, 0.1
    w = {k: 1 / 8 for k in range(1, 9)}
    def pop_risk(thr):
        ks = [k for k in w if k >= thr]
        return sum(w[k] * (1 - p_correct[k]) for k in ks) / sum(w[k] for k in ks)
    bad = 0
    for _ in range(400):
        calib = []
        for _ in range(108):
            k = rng.randint(1, 8)
            calib.append((k, rng.random() < p_correct[k]))
        thr = ltt_threshold(calib, alpha, delta)
        if thr is not None and pop_risk(thr) > alpha:
            bad += 1
    assert bad / 400 <= delta + 0.03
