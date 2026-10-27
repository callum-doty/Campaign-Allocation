"""Fast checks on the math, using synthetic data only."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign import load_presidential as lp  # noqa: E402
from campaign.allocate import allocate  # noqa: E402
from campaign.fit_baseline import fit_baseline  # noqa: E402
from campaign.predict_margin import predict_margin  # noqa: E402
from campaign.seat_distribution import seat_distribution  # noqa: E402
from campaign.spending_effect import spending_effect  # noqa: E402


def synthetic_races(n=500, seed=0):
    rng = np.random.default_rng(seed)
    status = rng.choice(3, n)
    races = pd.DataFrame({"pvi": rng.uniform(-25, 25, n), "d_incumbent": (status == 0) * 1.0,
                          "open_seat": (status == 1) * 1.0, "floor_share": rng.uniform(-2, -0.1, n),
                          "gb": rng.choice([-2.0, 0.0, 7.0], n), "cycle": 2020})
    races["margin"] = (1 + 1.7 * races.pvi + 12 * races.d_incumbent + 5 * races.open_seat
                       + 3 * races.floor_share + races.gb + rng.normal(0, 6, n))
    return races


def test_baseline_recovers_known_coefficients():
    model = fit_baseline(synthetic_races())
    assert model["coef"]["pvi"] == pytest.approx(1.7, abs=0.05)
    assert model["coef"]["d_incumbent"] == pytest.approx(12, abs=1.5)
    assert model["sigma"] == pytest.approx(6, rel=0.1)


def test_error_splits_into_national_and_district():
    races = synthetic_races(n=2000)
    shocks = {2014: -4.0, 2016: 0.0, 2018: 3.0, 2020: -2.0, 2022: 4.0, 2024: -1.0}
    races["cycle"] = np.resize(list(shocks), len(races))
    races["margin"] += races.cycle.map(shocks)
    model = fit_baseline(races)
    assert model["sigma_national"] == pytest.approx(np.std(list(shocks.values()), ddof=1), abs=0.5)
    assert model["sigma_district"] == pytest.approx(6, rel=0.1)
    assert model["sigma"] ** 2 == pytest.approx(model["sigma_national"] ** 2 + model["sigma_district"] ** 2)


def test_seat_distribution_mean_spread_and_symmetry():
    mu = np.linspace(-20, 20, 41)
    independent = seat_distribution(mu, {"sigma_national": 0.0, "sigma_district": 8.0}, need=21)
    shared = seat_distribution(mu, {"sigma_national": 4.0, "sigma_district": np.sqrt(64 - 16)}, need=21)
    assert independent["mean"] == pytest.approx(norm.cdf(mu / 8).sum(), abs=1e-6)
    assert shared["mean"] == pytest.approx(independent["mean"], abs=1e-6)        # same expected seats...
    assert shared["p90"] - shared["p10"] > independent["p90"] - independent["p10"]  # ...but a wider total
    assert independent["p_majority"] == pytest.approx(0.5, abs=0.02)


def test_generic_ballot_is_a_one_for_one_swing():
    races = synthetic_races()
    model = fit_baseline(races)
    assert np.allclose(predict_margin(races.assign(gb=3.0), model) - predict_margin(races.assign(gb=0.0), model), 3.0)


def test_spending_effect_is_zero_at_zero_and_concave():
    assert spending_effect(0) == 0
    a, b, c = spending_effect([1e6, 2e6, 3e6])
    assert 0 < c - b < b - a < a


def test_observed_allocation_reproduces_the_baseline():
    mu, observed = np.array([-5.0, 0.0, 8.0]), np.array([1e6, 3e6, 0.0])
    shift = spending_effect(observed) - spending_effect(observed)
    assert np.allclose(norm.cdf((mu + shift) / 6), norm.cdf(mu / 6))


def test_allocation_respects_budget_and_cap_and_favors_close_races():
    x = allocate(np.array([-20.0, 0.0, 20.0]), 6.0, np.zeros(3), budget=3e6, cap=2e6)
    assert x.sum() <= 3e6 and x.max() <= 2e6
    assert x[1] == 2e6


def test_greedy_matches_brute_force():
    mu, sigma, step, budget, cap = np.array([-9.0, -2.0, 4.0]), 5.0, 5e5, 3e6, 2e6
    x = allocate(mu, sigma, np.zeros(3), budget, cap, step=step, scale_pts=4.0)

    def seats(alloc):
        return norm.cdf((mu + spending_effect(alloc, 4.0)) / sigma).sum()

    best = max(seats(np.array(a)) for a in itertools.product(np.arange(0, cap + step, step), repeat=3)
               if sum(a) <= budget)
    assert seats(x) == pytest.approx(best, abs=1e-9)


def test_presidential_sheet_parsing(tmp_path, monkeypatch):
    (tmp_path / "lines_2026.csv").write_text(
        "SPONSORED,,,,,,\nDistrict,Incumbent,Party,2024,,,2020\n,,,Harris,Trump,Margin,Biden,Trump\n"
        "AK-AL,Someone,(R),45,55,-10,43,53\nNY-03,Someone,(D),60,40,20,,\n")
    monkeypatch.setattr(lp, "RAW", tmp_path)
    out = lp.load_presidential(2026).set_index(["district_id", "pres_year"]).d_share
    assert out[("AK-00", 2024)] == pytest.approx(0.45)
    assert out[("NY-03", 2024)] == pytest.approx(0.60)
    assert ("NY-03", 2020) not in out.index
