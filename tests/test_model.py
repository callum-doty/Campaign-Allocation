"""Fast checks on the math, using synthetic data only."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign import load_candidates as lc  # noqa: E402
from campaign import load_early_money as lem  # noqa: E402
from campaign import load_presidential as lp  # noqa: E402
from campaign.allocate import allocate  # noqa: E402
from campaign.break_even import break_even_scale  # noqa: E402
from campaign.fit_baseline import fit_baseline  # noqa: E402
from campaign.implied_belief import implied_belief  # noqa: E402
from campaign.predict_margin import predict_margin  # noqa: E402
from campaign.reallocation_gain import reallocation_gain  # noqa: E402
from campaign.seat_distribution import seat_distribution  # noqa: E402
from campaign.solve_game import solve_game  # noqa: E402
from campaign.spending_effect import spending_effect  # noqa: E402


STATES = [a + b for a in "ABCDE" for b in "ABCDEFGH"]       # 40 two-letter "states"


def synthetic_races(n=500, seed=0):
    rng = np.random.default_rng(seed)
    status = rng.choice(3, n)
    races = pd.DataFrame({"pvi": rng.uniform(-25, 25, n), "d_incumbent": (status == 0) * 1.0,
                          "open_seat": (status == 1) * 1.0, "floor_share": rng.uniform(-2, -0.1, n),
                          "gb": rng.choice([-2.0, 0.0, 7.0], n), "cycle": 2020,
                          "district_id": [f"{STATES[i % 40]}-{i:03d}" for i in range(n)]})
    races["fresh_d"] = races.d_incumbent * (rng.random(n) < 0.2)
    races["fresh_r"] = (status == 2) * (rng.random(n) < 0.2) * 1.0
    races["over_inc"] = (status != 1) * rng.normal(0, 6, n)
    races["over_open"] = (status == 1) * rng.normal(0, 6, n)
    races["margin"] = (1 + 1.7 * races.pvi + 12 * races.d_incumbent + 5 * races.open_seat
                       + 3 * races.floor_share - 5 * races.fresh_d + 0.25 * races.over_inc
                       + races.gb + rng.normal(0, 6, n))
    return races


def test_baseline_recovers_known_coefficients():
    model = fit_baseline(synthetic_races())
    assert model["coef"]["pvi"] == pytest.approx(1.7, abs=0.05)
    assert model["coef"]["d_incumbent"] == pytest.approx(12, abs=1.5)
    assert model["coef"]["fresh_d"] == pytest.approx(-5, abs=2)
    assert model["coef"]["over_inc"] == pytest.approx(0.25, abs=0.1)
    assert model["sigma"] == pytest.approx(6, rel=0.1)


def test_error_splits_into_national_state_and_district():
    races = synthetic_races(n=6000)
    shocks = {2014: -4.0, 2016: 0.0, 2018: 3.0, 2020: -2.0, 2022: 4.0, 2024: -1.0}
    races["cycle"] = np.resize(list(shocks), len(races))
    races["margin"] += races.cycle.map(shocks)
    state = races.district_id.str[:2] + races.cycle.astype(str)        # 40 states x 6 cycles, 25 races each
    state_shocks = np.random.default_rng(1).normal(0, 3, state.nunique())
    races["margin"] += state.map(dict(zip(state.unique(), state_shocks)))
    model = fit_baseline(races)
    assert model["sigma_national"] == pytest.approx(np.std(list(shocks.values()), ddof=1), abs=0.5)
    assert model["sigma_state"] == pytest.approx(np.std(state_shocks, ddof=1), abs=0.5)
    assert model["sigma_district"] == pytest.approx(6, rel=0.1)
    assert model["sigma"] ** 2 == pytest.approx(
        model["sigma_national"] ** 2 + model["sigma_state"] ** 2 + model["sigma_district"] ** 2)


def test_seat_distribution_mean_spread_and_symmetry():
    mu = np.linspace(-20, 20, 41)
    independent = seat_distribution(mu, {"sigma_national": 0.0, "sigma_district": 8.0}, need=21)
    shared = seat_distribution(mu, {"sigma_national": 4.0, "sigma_district": np.sqrt(64 - 16)}, need=21)
    assert independent["mean"] == pytest.approx(norm.cdf(mu / 8).sum(), abs=1e-6)
    assert shared["mean"] == pytest.approx(independent["mean"], abs=1e-6)        # same expected seats...
    assert shared["p90"] - shared["p10"] > independent["p90"] - independent["p10"]  # ...but a wider total
    assert independent["p_majority"] == pytest.approx(0.5, abs=0.02)


def test_state_error_widens_the_seat_total_but_not_its_mean():
    mu, states = np.linspace(-20, 20, 41), np.repeat(np.arange(8), 6)[:41]
    national = {"sigma_national": 3.0, "sigma_state": 0.0, "sigma_district": np.sqrt(64 - 9)}
    with_state = {"sigma_national": 3.0, "sigma_state": 4.0, "sigma_district": np.sqrt(64 - 9 - 16)}
    a = seat_distribution(mu, national, need=21, states=states)
    b = seat_distribution(mu, with_state, need=21, states=states)
    assert b["mean"] == pytest.approx(a["mean"], abs=1e-6)
    assert b["p90"] - b["p10"] > a["p90"] - a["p10"]
    assert seat_distribution(mu, national, need=21)["p10"] == a["p10"]     # no state error: grouping is irrelevant


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


def test_costlier_race_gets_less_money():
    # two identical toss-ups; the second's dollars buy half as much (higher saturation point)
    x = allocate(np.zeros(2), 6.0, np.zeros(2), budget=4e6, cap=4e6,
                 half_saturation=np.array([5.5e6, 11e6]))
    assert x[0] > x[1]


def test_symmetric_game_has_no_gain_left():
    # races mirrored around a toss-up, equal budgets: by symmetry the equilibrium
    # leaves expected seats at exactly half, and neither side can gain by deviating
    races = pd.DataFrame({"mu": [-4.0, 0.0, 4.0], "dccc": [1e6] * 3, "nrcc": [1e6] * 3})
    game = solve_game(races, sigma=6.0, rounds=20, step=250_000)
    assert game["seats_equilibrium"] == pytest.approx(1.5, abs=0.02)
    assert game["exploit_equilibrium"] < 0.01


def test_greedy_matches_brute_force():
    mu, sigma, step, budget, cap = np.array([-9.0, -2.0, 4.0]), 5.0, 5e5, 3e6, 2e6
    x = allocate(mu, sigma, np.zeros(3), budget, cap, step=step, scale_pts=4.0)

    def seats(alloc):
        return norm.cdf((mu + spending_effect(alloc, 4.0)) / sigma).sum()

    best = max(seats(np.array(a)) for a in itertools.product(np.arange(0, cap + step, step), repeat=3)
               if sum(a) <= budget)
    assert seats(x) == pytest.approx(best, abs=1e-9)


def test_break_even_scale_is_where_the_gain_reaches_the_target():
    # money parked in two safe seats; moving it to the close races pays only if money is strong
    mu, observed = np.array([-10.0, -3.0, 0.0, 3.0, 8.0, 12.0, 15.0]), np.array([0, 0, 0, 0, 0, 2e6, 2e6])
    scale = break_even_scale(mu, 6.0, observed, cap=4e6, half_saturation=5.5e6, target=0.5)

    def gain(s):
        return reallocation_gain(mu, 6.0, observed, 4e6, scale_pts=s, half_saturation=5.5e6)[0]

    assert gain(scale) == pytest.approx(0.5, abs=0.01)
    assert gain(0.9 * scale) < 0.5


def test_implied_multiplier_is_one_at_an_optimum_and_free_of_scale():
    # the model's own optimum, read backwards, should need no extra belief in any band
    mu0, sigma = np.array([-14.0, -9.0, -4.0, 0.0, 4.0, 9.0, 14.0]), 6.0
    x = allocate(mu0, sigma, np.zeros(7), budget=30e6, cap=30e6, step=50_000)
    mu = mu0 + spending_effect(x)            # the forecast at that allocation
    belief = implied_belief(mu, sigma, x)
    assert (belief.dollars > 0).all()        # every band is funded
    assert np.allclose(belief.multiplier, 1.0, atol=0.01)
    lopsided = np.array([1e6, 0, 0, 2e6, 0, 6e6, 4e6])
    assert np.allclose(implied_belief(mu, sigma, lopsided, scale_pts=0.5).multiplier,
                       implied_belief(mu, sigma, lopsided, scale_pts=4.5).multiplier)


def test_presidential_sheet_parsing(tmp_path, monkeypatch):
    (tmp_path / "lines_2026.csv").write_text(
        "SPONSORED,,,,,,\nDistrict,Incumbent,Party,2024,,,2020\n,,,Harris,Trump,Margin,Biden,Trump\n"
        "AK-AL,Someone,(R),45,55,-10,43,53\nNY-03,Someone,(D),60,40,20,,\n")
    monkeypatch.setattr(lp, "RAW", tmp_path)
    out = lp.load_presidential(2026).set_index(["district_id", "pres_year"]).d_share
    assert out[("AK-00", 2024)] == pytest.approx(0.45)
    assert out[("NY-03", 2024)] == pytest.approx(0.60)
    assert ("NY-03", 2020) not in out.index


def test_early_money_picks_the_report_a_september_forecaster_had(tmp_path, monkeypatch):
    (tmp_path / "reports_2018.csv").write_text(
        "committee_id,report_type,coverage_end_date,receipt_date,file_number,total_receipts_ytd,"
        "total_disbursements_ytd,candidate_contribution_ytd,loans_made_by_candidate_ytd\n"
        "C1,Q2,2018-06-30T00:00:00,2018-07-15,1,100,40,0,10\n"
        "C1,Q2,2018-06-30T00:00:00,2018-08-01,2,120,40,0,10\n"        # amendment: the latest version counts
        "C2,12R,2018-07-02T00:00:00,2018-07-10,3,50,20,5,0\n"          # runoff report stands in for the July one
        "C3,12P,2018-08-10T00:00:00,2018-08-20,4,70,20,0,0\n"          # ends too late to stand in
        "C4,Q2,2018-06-30T00:00:00,2018-07-15,5,80,30,0,0\n"
        "C5,Q2,2018-06-30T00:00:00,2018-07-15,6,30,10,0,0\n"
        "C6,Q2,2018-06-30T00:00:00,2018-07-15,7,999,0,0,0\n")
    (tmp_path / "ccl18.txt").write_text(
        "H1|2018|2018|C1|H|P|1\nH1|2018|2018|C6|H|J|2\nH2|2018|2018|C2|H|P|3\nH4|2018|2018|C3|H|P|4\n"
        "H3|2018|2018|C4|H|A|5\nH3|2018|2018|C5|H|P|6\n")
    monkeypatch.setattr(lem, "RAW", tmp_path)
    money = lem.load_early_money(2018)
    assert money.raised.to_dict() == {"H1": 120, "H2": 50, "H3": 80}    # H3: its biggest committee, not a sum
    assert money.own["H1"] == 10 and money.own["H2"] == 5


def test_hand_checked_nominees_replace_the_top_spender_guess(tmp_path, monkeypatch):
    def weball_row(cand_id, name, ici, party, disbursed, state, district):
        row = [""] * 30
        row[0], row[1], row[2], row[4], row[7], row[18], row[19] = cand_id, name, ici, party, disbursed, state, district
        return "|".join(row)
    (tmp_path / "weball26.txt").write_text("\n".join([
        weball_row("H6AA01001", "BIGSPENDER, PAT", "C", "DEM", "900000", "AA", "01"),   # lost the primary
        weball_row("H6AA01002", "NOMINEE, SAM", "C", "DEM", "300000", "AA", "01"),
        weball_row("H6AA01003", "MEMBER, ALEX", "I", "REP", "500000", "AA", "01")]) + "\n")
    (tmp_path / "lines_2026.csv").write_text("District,Incumbent,Party\nAA-01,Alex Member,(R)\n")
    (tmp_path / "nominees_2026.csv").write_text("district_id,party,cand_id,name,ici,source\nAA-01,D,H6AA01002,Sam Nominee,C,x\n")
    for attr in ("RAW", "SHEETS", "DATA"):
        monkeypatch.setattr(lc, attr, tmp_path)
    picked = lc.load_candidates(2026, pd.DataFrame()).set_index("party")
    assert picked.cand_id.to_dict() == {"D": "H6AA01002", "R": "H6AA01003"}
    assert picked.ici["R"] == "I"

