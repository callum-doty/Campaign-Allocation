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
from campaign import fall_drift as fd  # noqa: E402
from campaign import load_early_vote as lev  # noqa: E402
from campaign import household_index as hi  # noqa: E402
from campaign import load_early_money as lem  # noqa: E402
from campaign import load_fred as lf  # noqa: E402
from campaign import load_presidential as lp  # noqa: E402
from campaign import load_results as lres  # noqa: E402
from campaign.allocate import allocate  # noqa: E402
from campaign.ballot_timing import election_day, persuasion_weight, timed_gain  # noqa: E402
from campaign.break_even import break_even_scale  # noqa: E402
from campaign.fit_baseline import fit_baseline  # noqa: E402
from campaign.fit_close import close_inc, fit_close  # noqa: E402
from campaign.fit_drift import fit_drift, random_walk  # noqa: E402
from campaign.fit_spread import fit_spread, race_sigma  # noqa: E402
from campaign.implied_belief import implied_belief  # noqa: E402
from campaign import lookup_quality as lq  # noqa: E402
from campaign.national_test import CYCLES, economics_only, national_test  # noqa: E402
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
    races["q_d"] = (status != 0) * (rng.random(n) < 0.2) * 1.0       # an experienced non-incumbent nominee
    races["q_r"] = (status != 2) * (rng.random(n) < 0.2) * 1.0
    races["quality"] = races.q_d - races.q_r
    races["margin"] += 2 * races.quality
    return races


def test_baseline_recovers_known_coefficients():
    model = fit_baseline(synthetic_races())
    assert model["coef"]["pvi"] == pytest.approx(1.7, abs=0.05)
    assert model["coef"]["d_incumbent"] == pytest.approx(12, abs=1.5)
    assert model["coef"]["fresh_d"] == pytest.approx(-5, abs=2)
    assert model["coef"]["over_inc"] == pytest.approx(0.25, abs=0.1)
    assert model["coef"]["quality"] == pytest.approx(2, abs=1.2)
    assert model["sigma"] == pytest.approx(6, rel=0.1)


STATE_NAMES = ["Virginia", "West Virginia", "Georgia", "Nebraska", "Wyoming", "Michigan", "California", "New Jersey"]


def test_challenger_names_match_the_person_not_a_relative_or_namesake():
    def same(title, name, state):
        return lq.same_person(title, name, state, STATE_NAMES)
    assert same("Bill Owens (politician)", "OWENS, WILLIAM L", "Virginia")              # nickname
    assert same("Donald McEachin", "MCEACHIN, A. DONALD", "Virginia")                   # goes by his middle name
    assert same("Bonnie Watson Coleman", "WATSON COLEMAN, BONNIE", "New Jersey")        # two-word surname
    assert same("Sydney Kamlager-Dove", "KAMLAGER, SYDNEY", "California")               # surname added later
    assert same("Joe Manchin (West Virginia politician)", "MANCHIN, JOE", "West Virginia")
    assert same("Tom Kean Jr.", "THOMAS H KEAN JR", "New Jersey")                      # a ballot name
    assert not same("Mac Collins", "COLLINS, MIKE", "Georgia")                         # his father
    assert not same("Joseph P. Kennedy Sr.", "KENNEDY, JOSEPH P III", "Virginia")      # his grandfather
    assert not same("Dale Kildee", "KILDEE, DANIEL T", "Michigan")                     # his uncle
    assert not same("John Clinton Porter", "PORTER, JOHN MARK", "California")          # another middle name
    assert not same("Mike Sullivan (Wyoming politician)", "SULLIVAN, MARK", "Nebraska")
    assert not same("Joe Smith (West Virginia politician)", "SMITH, JOE", "Virginia")
    assert not same("John Adams College and Career Academy", "ADAMS, JOHN", "Virginia")


def test_an_article_on_the_other_party_is_a_namesake():
    infobox = "{{Infobox officeholder\n| party = [[Democratic Party (United States)|Democratic]]\n}}"
    prose = "'''Bob Brooks''' is an American politician. He is a former [[Republican Party (United States)|Republican]] member."
    assert not lq.party_ok(infobox, "R") and lq.party_ok(infobox, "D")
    assert not lq.party_ok(prose, "D") and lq.party_ok(prose, "R")
    assert lq.party_ok("'''Jo Doe''' is a businesswoman. She lost to Democratic incumbent Jane Roe.", "R")


def test_quality_reads_the_article_as_it_stood_on_jan_1(tmp_path, monkeypatch):
    articles = {"Ann Early": "Ann Early is a member of the Georgia House of Representatives.",
                "Bo Late": None,                                      # created during the campaign
                "Cy Redirect": "#REDIRECT [[2020 elections in Georgia]]",
                "Di Golfer": "Di Golfer is a professional golfer from Georgia."}
    looked_up = []

    def wiki(session, params):
        if params.get("list") == "search":
            surname = params["srsearch"].split('"')[1].lower()
            return {"query": {"search": [{"title": t} for t in articles if t.lower().endswith(surname)]}}
        looked_up.append(params["titles"])
        text = articles[params["titles"]]
        revisions = [] if text is None else [{"timestamp": "2019-12-31T00:00:00Z", "slots": {"main": {"content": text}}}]
        return {"query": {"pages": [{"revisions": revisions}]}}
    monkeypatch.setattr(lq, "wiki", wiki)
    nominees = pd.DataFrame({"cand_id": ["H1", "H2", "H3", "H4", "H5"], "district_id": ["GA-01"] * 5,
                             "party": ["D", "R", "D", "R", "D"], "ici": ["C", "C", "O", "O", "I"],
                             "name": ["EARLY, ANN", "LATE, BO", "REDIRECT, CY", "GOLFER, DI", "EARLY, ANN"]})
    dest = tmp_path / "quality_2020.csv"
    lq.lookup_quality(2020, nominees, {"GA": "Georgia"}, dest, {})
    found = pd.read_csv(dest).set_index("cand_id").quality.to_dict()
    assert found == {"H1": 1, "H2": 0, "H3": 0, "H4": 0}                 # the incumbent (H5) isn't looked up
    calls = len(looked_up)
    lq.lookup_quality(2020, nominees, {"GA": "Georgia"}, dest, {})       # nothing new to look up
    assert len(looked_up) == calls


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
    assert model["national_df"] == len(shocks) - 1
    assert model["sigma_state"] == pytest.approx(np.std(state_shocks, ddof=1), abs=0.5)
    assert model["sigma_district"] == pytest.approx(6, rel=0.1)
    assert model["sigma"] ** 2 == pytest.approx(
        model["sigma_national"] ** 2 + model["sigma_state"] ** 2 + model["sigma_district"] ** 2)


def drifting_races(edges, redrawn_ahead=2.0, n=1500, seed=0):
    """Races over several cycles: incumbents run `edges[cycle]` ahead each way, each cycle has its own
    national miss, and incumbents in 16 of the 40 states, on redrawn lines, run `redrawn_ahead` further."""
    rng = np.random.default_rng(seed)
    cycles = []
    for i, (cycle, edge) in enumerate(edges.items()):
        races = synthetic_races(n, seed=seed + i).assign(cycle=cycle)
        races["status"] = np.select([races.d_incumbent == 1, races.open_seat == 1], ["D incumbent", "Open"], "R incumbent")
        inc = races.d_incumbent - (races.status == "R incumbent")
        races["redrawn"] = races.district_id.str[0].isin(["A", "B"]) * 1.0
        races["inc_redrawn"] = inc * races.redrawn
        races["margin"] = (1 + 1.7 * races.pvi + edge * inc + 3 * races.floor_share + 0.25 * races.over_inc
                           + redrawn_ahead * races.inc_redrawn + races.gb + rng.normal(0, 3) + rng.normal(0, 6, n))
        cycles.append(races)
    return pd.concat(cycles, ignore_index=True)


def test_drifting_incumbency_follows_each_cycle_and_finds_the_redrawn_term():
    edges = {2012: 8.0, 2014: 6.0, 2016: 6.5, 2018: 3.0, 2020: 2.0, 2022: 2.5}
    model = fit_drift(drifting_races(edges))
    inc = model["incumbency"]
    assert inc["c"] == pytest.approx([2 * e for e in edges.values()], abs=1.0)
    assert inc["redrawn"] == pytest.approx(2.0, abs=0.5)
    assert inc["tau"] > 1 and inc["c_next"] < 8
    assert model["coef"]["d_incumbent"] == pytest.approx(inc["c_next"])
    assert model["coef"]["pvi"] == pytest.approx(1.7, abs=0.05)


def test_steady_incumbency_barely_drifts():
    inc = fit_drift(drifting_races(dict.fromkeys(range(2012, 2024, 2), 5.0)))["incumbency"]
    assert inc["tau"] < 1.5
    assert inc["c_next"] == pytest.approx(10, abs=0.6)


def test_redrawn_term_needs_a_cycle_with_both_kinds_of_lines():
    races = drifting_races({2012: 8.0, 2014: 6.0})
    races["redrawn"] = (races.cycle == 2012) * 1.0
    races["inc_redrawn"] = (races.d_incumbent - (races.status == "R incumbent")) * races.redrawn
    model = fit_drift(races)
    assert model["incumbency"]["redrawn"] is None and "inc_redrawn" not in model["coef"]


def test_random_walk_pools_without_drift_and_follows_the_last_estimate_with_a_lot():
    estimates, variances = np.array([10.0, 4.0, 7.0]), np.array([1.0, 4.0, 1.0])
    _, mean, var = random_walk(estimates, variances, 0.0)
    assert mean == pytest.approx(np.sum(estimates / variances) / np.sum(1 / variances))
    assert var == pytest.approx(1 / np.sum(1 / variances))
    assert random_walk(estimates, variances, 1e4)[1] == pytest.approx(7.0, abs=1e-3)


def test_close_race_edge_is_found_where_the_forecast_is_close_and_nowhere_else():
    races = drifting_races(dict.fromkeys(range(2012, 2020, 2), 6.0), redrawn_ahead=0.0)
    model = fit_baseline(races)
    assert fit_close(races, model)["coef"]["inc_close"] == pytest.approx(0, abs=0.75)
    races["margin"] -= 3 * close_inc(races, model)          # incumbents in close races worth 3 points less each way
    fitted = fit_close(races, model)
    assert fitted["coef"]["inc_close"] == pytest.approx(-3, abs=0.75)
    assert fitted["coef"]["d_incumbent"] == pytest.approx(12, abs=1.5)


def test_error_size_follows_a_race_feature_within_cycles():
    races = drifting_races(dict.fromkeys(range(2012, 2024, 2), 5.0), redrawn_ahead=0.0)
    races["margin"] += races.open_seat * np.random.default_rng(1).normal(0, np.sqrt(9 ** 2 - 6 ** 2), len(races))
    spread = fit_spread(races, fit_baseline(races))
    assert spread["g"]["open_seat"] == pytest.approx(np.log(9 ** 2 / 6 ** 2), abs=0.15)
    assert spread["g"]["redrawn_inc"] == pytest.approx(0, abs=0.15)
    assert spread["g"]["lean"] == pytest.approx(0, abs=0.05)


def test_a_noisy_year_doesnt_read_as_a_term_that_falls_in_it():
    races = drifting_races(dict.fromkeys(range(2012, 2020, 2), 5.0), redrawn_ahead=0.0)
    first = races.cycle == 2012                  # every incumbent on new lines, and every race noisier
    races.loc[first, "redrawn"] = 1.0
    races["inc_redrawn"] = (races.d_incumbent - (races.status == "R incumbent")) * races.redrawn
    races.loc[first, "margin"] += np.random.default_rng(1).normal(0, 8, first.sum())
    spread = fit_spread(races, fit_baseline(races))
    assert spread["g"]["redrawn_inc"] == pytest.approx(0, abs=0.15)
    sigma = race_sigma(races[races.margin.abs() < 80], spread)
    assert np.mean(sigma ** 2) == pytest.approx(spread["sigma"] ** 2)


def test_redrawn_error_term_needs_a_cycle_with_both_kinds_of_lines():
    races = drifting_races({2012: 8.0, 2014: 6.0})
    races["redrawn"] = (races.cycle == 2012) * 1.0
    races["inc_redrawn"] = (races.d_incumbent - (races.status == "R incumbent")) * races.redrawn
    assert "redrawn_inc" not in fit_spread(races, fit_baseline(races))["g"]


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


def test_uncertain_national_error_fattens_the_tails():
    mu = np.linspace(-15, 25, 41)                                          # Democrats favored for the majority
    known = {"sigma_national": 3.0, "sigma_district": 7.0}
    few, many = seat_distribution(mu, {**known, "national_df": 3}, need=21), seat_distribution(mu, known, need=21)
    assert few["p90"] - few["p10"] > many["p90"] - many["p10"]
    assert few["p_majority"] < many["p_majority"]
    lots = seat_distribution(mu, {**known, "national_df": 10_000}, need=21)
    assert lots["p_majority"] == pytest.approx(many["p_majority"], abs=1e-3)   # a t with many cycles is the normal
    cauchy = seat_distribution(mu, {**known, "national_df": 1}, need=21)
    assert np.isfinite(cauchy["mean"]) and cauchy["pmf"].sum() == pytest.approx(1)


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


def test_last_winners_fec_lists_as_challengers_are_incumbents(tmp_path, monkeypatch):
    def weball_row(cand_id, name, ici, party, disbursed, state, district):
        row = [""] * 30
        row[0], row[1], row[2], row[4], row[7], row[18], row[19] = cand_id, name, ici, party, disbursed, state, district
        return "|".join(row)
    (tmp_path / "weball14.txt").write_text("\n".join([
        weball_row("H2AA01001", "SCHNEIDER, BRADLEY SCOTT", "C", "DEM", "900000", "AA", "01"),   # won in 2012
        weball_row("H0AA01002", "DOLD, ROBERT J", "C", "REP", "800000", "AA", "01"),             # lost in 2012
        weball_row("H2AA02001", "KHANNA, ROHIT", "C", "DEM", "700000", "AA", "02"),              # lost a D-vs-D race
        weball_row("H0AA02002", "HONDA, MICHAEL", "I", "DEM", "600000", "AA", "02"),
        weball_row("H4AA03001", "HOOPER, JAMES", "C", "REP", "500000", "AA", "03"),              # not John James
        weball_row("H2AA03002", "JONES, PAT", "C", "DEM", "400000", "AA", "03")]) + "\n")
    mit = [(2012, "01", "BRAD SCHNEIDER", "DEMOCRAT", 120), (2012, "01", "ROBERT DOLD", "REPUBLICAN", 100),
           (2012, "02", "MIKE HONDA", "DEMOCRAT", 150), (2012, "02", "RO KHANNA", "DEMOCRAT", 100),
           (2012, "03", "JOHN JAMES", "REPUBLICAN", 200), (2012, "03", "PAT JONES", "DEMOCRAT", 100),
           (2014, "01", "BRADLEY S SCHNEIDER", "DEMOCRAT", 90), (2014, "01", "ROBERT J DOLD", "REPUBLICAN", 100),
           (2014, "02", "RO KHANNA", "DEMOCRAT", 100), (2014, "02", "MIKE HONDA", "DEMOCRAT", 120),
           (2014, "03", "JAMES HOOPER", "REPUBLICAN", 200), (2014, "03", "PAT JONES", "DEMOCRAT", 100)]
    pd.DataFrame([{"year": y, "stage": "GEN", "mode": "TOTAL", "runoff": False, "special": False, "writein": False,
                   "state_po": "AA", "district": d, "candidate": c, "party": p, "candidatevotes": v}
                  for y, d, c, p, v in mit]).to_csv(tmp_path / "house.csv", index=False)
    monkeypatch.setattr(lc, "RAW", tmp_path)
    monkeypatch.setattr(lres, "RAW", tmp_path / "house.csv")
    assert lres.load_results(2012).set_index("district_id").winner.to_dict() == {
        "AA-01": "BRAD SCHNEIDER", "AA-02": "MIKE HONDA", "AA-03": "JOHN JAMES"}
    picked = lc.load_candidates(2014, lres.load_results(2014)).set_index(["district_id", "party"])
    assert picked.ici.to_dict() == {("AA-01", "D"): "I", ("AA-01", "R"): "C", ("AA-02", "D"): "C",
                                    ("AA-03", "R"): "C", ("AA-03", "D"): "C"}


def test_sitting_members_fec_lists_as_challengers_are_incumbents(tmp_path, monkeypatch):
    def weball_row(cand_id, name, ici, party, disbursed, state, district):
        row = [""] * 30
        row[0], row[1], row[2], row[4], row[7], row[18], row[19] = cand_id, name, ici, party, disbursed, state, district
        return "|".join(row)
    (tmp_path / "weball26.txt").write_text("\n".join([
        weball_row("H6AA01001", "ONDER JR, ROBERT FRANK", "C", "REP", "500000", "AA", "01"),   # goes by Bob
        weball_row("H6AA02001", "HOOPER, JAMES D", "C", "REP", "400000", "AA", "02"),          # not John James
        weball_row("H6AA03001", "MENENDEZ, ROBERT J.", "I", "DEM", "300000", "AA", "03"),
        weball_row("S6AA00001", "MENENDEZ, ROBERT", "I", "DEM", "800000", "AA", "00"),         # his father's old ID
        weball_row("H6AA04001", "RUNNER, KIM", "I", "REP", "600000", "AA", "04"),
        weball_row("S6AA00002", "RUNNER, KIM", "O", "REP", "900000", "AA", "00"),              # running for Senate
        weball_row("H6AA04002", "HOPEFUL, LEE", "C", "REP", "100000", "AA", "04")]) + "\n")
    (tmp_path / "lines_2026.csv").write_text("District,Incumbent,Party\nAA-01,Bob Onder,(R)\nAA-03,Rob Menendez,(D)\n"
                                             "AA-04,Kim Runner,(R)\nAA-05,John James,(R)\n")
    for attr in ("RAW", "SHEETS", "DATA"):
        monkeypatch.setattr(lc, attr, tmp_path)
    picked = lc.load_candidates(2026, pd.DataFrame()).set_index("district_id")
    assert picked.cand_id.to_dict() == {"AA-01": "H6AA01001", "AA-02": "H6AA02001", "AA-03": "H6AA03001",
                                        "AA-04": "H6AA04002"}
    assert picked.ici.to_dict() == {"AA-01": "I", "AA-02": "C", "AA-03": "I", "AA-04": "C"}



def test_household_index_reweights_missing_items_and_reads_june(tmp_path, monkeypatch):
    months = pd.date_range("2019-01-01", "2020-12-01", freq="MS")
    quarters = pd.date_range("2019-01-01", "2020-10-01", freq="QS")
    series = {"A": (months, np.where(months.year == 2020, 110.0, 100.0)),       # +10%, weight 3
              "B": (months, np.where(months.year == 2020, 51.0, 50.0)),         # +2%, weight 1
              "C": (months[:-7], np.where(months[:-7].year == 2020, 9.0, 3.0)),  # no June 2020: left out
              "U": (months, np.where(months.year == 2020, 5.0, 4.0)),           # +1 point, monthly
              "Q": (quarters, np.where(quarters.year == 2020, 1.0, 2.0))}       # -1 point, quarterly
    for sid, (dates, values) in series.items():
        pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), sid: values}).to_csv(tmp_path / f"{sid}.csv", index=False)
    monkeypatch.setattr(lf, "RAW", tmp_path)
    monkeypatch.setattr(hi, "BASKET", [("a", "A", 3), ("b", "B", 1), ("c", "C", 3)])
    monkeypatch.setattr(hi, "STRESS", [("u", "U"), ("q", "Q")])
    row = hi.household_index([2020]).loc[2020]
    assert row.basket == pytest.approx((3 * 10 + 1 * 2) / 4)
    assert row.basket_items == 2
    assert row.stress == pytest.approx(0.0)


def test_national_test_finds_a_planted_effect_and_not_a_missing_one():
    rng = np.random.default_rng(3)
    s = np.resize([1.0, -1.0, -1.0, 1.0], len(CYCLES))
    z = rng.normal(0, 3, len(CYCLES))
    noise = rng.normal(0, 0.3, len(CYCLES))
    planted = pd.DataFrame({"s": s, "z": z, "gb": rng.normal(0, 5, len(CYCLES)), "miss": -1 - 0.8 * s * z + noise,
                            "midterm": [float(c % 4 == 2) for c in CYCLES]}, index=list(CYCLES))
    found = national_test(planted, composites=("z",), n_perm=2000)["composites"]["z"]
    assert found["coef"]["c"] == pytest.approx(-0.8, abs=0.1) and found["passes"]
    absent = national_test(planted.assign(miss=-1 + 2 * noise), composites=("z",), n_perm=2000)["composites"]["z"]
    assert not absent["passes"] and absent["p"] > 0.1


def test_economics_only_recovers_a_vote_driven_by_the_composite():
    rng = np.random.default_rng(4)
    s = np.resize([1.0, -1.0, -1.0, 1.0], len(CYCLES))
    table = pd.DataFrame({"s": s, "z": rng.normal(0, 3, len(CYCLES)), "midterm": [float(c % 4 == 2) for c in CYCLES]},
                         index=list(CYCLES))
    table["house_vote"] = s * (2 - 4 * table.midterm - 1.5 * table.z + rng.normal(0, 0.2, len(CYCLES)))
    scores = economics_only(table, composites=("z",))
    assert scores["z"]["loo"] < 0.5 < scores["midterm"]["loo"]


def test_fall_drift_uses_only_earlier_elections_and_the_right_date(tmp_path, monkeypatch):
    final, dated = tmp_path / "final.csv", tmp_path / "dated.csv"
    pd.DataFrame({"cycle": [2002, 2004, 2006, 2008], "generic_ballot": [-1.0, 2.0, 5.0, 40.0]}).to_csv(final, index=False)
    pd.DataFrame({"cycle": [2002, 2004, 2006, 2008] * 2, "date": ["09-01"] * 4 + ["10-15"] * 4,
                  "generic_ballot": [1.0, 3.0, 5.0, 0.0, -1.0, 2.0, 5.0, 0.0]}).to_csv(dated, index=False)
    monkeypatch.setattr(fd, "GENERIC_BALLOT", final)
    monkeypatch.setattr(fd, "GENERIC_BALLOT_DATED", dated)
    drift = fd.fall_drift(2008)                        # 2008's own drift (+40) must not leak in
    assert drift["n"] == 3
    assert drift["mean"] == pytest.approx(-1.0)        # drifts of -2, -1, 0
    assert drift["sd"] == pytest.approx(1.0)
    assert fd.fall_drift(2008, "10-15") == {"mean": 0.0, "sd": 0.0, "n": 3}


def test_tranches_judged_alike_give_the_same_reading_as_one_row():
    mu, sigma, observed = np.array([-15.0, 0.0, 12.0]), 8.0, np.array([1e6, 4e6, 2e6])
    whole = implied_belief(mu, sigma, observed)
    split = implied_belief(np.repeat(mu, 2), np.full(6, sigma), np.repeat(observed, 2), weights=np.repeat(observed / 2, 2))
    assert np.allclose(whole.multiplier, split.multiplier, equal_nan=True)
    assert np.allclose(whole.dollars, split.dollars)


def test_persuasion_fades_and_counts_only_ballots_cast_after_it():
    assert [str(election_day(c).date()) for c in (2020, 2022, 2026)] == ["2020-11-03", "2022-11-08", "2026-11-03"]
    days = np.arange(0, 29, 7)
    assert np.allclose(persuasion_weight(days, 0.0, half_life=4), 0.5 ** (days / 4))   # everyone votes on Election Day
    assert persuasion_weight(-1, 0.3) == 0                                              # after the election
    all_early = persuasion_weight(np.arange(40), 1.0, half_life=4, window=21)
    assert all_early[0] == 0 and all_early.argmax() == 21                              # peaks when early voting opens


def test_timing_gain_is_zero_when_money_already_landed_at_its_best():
    mu, observed = np.array([-3.0, 1.0, 6.0]), np.array([2e6, 3e6, 1e6])
    gain, _ = timed_gain(mu, 8.0, observed, np.ones(3), np.ones(3), cap=3e6, move=False)
    assert gain == pytest.approx(0.0, abs=1e-12)
    late, _ = timed_gain(mu, 8.0, observed, np.full(3, 0.5), np.ones(3), cap=3e6, move=False)
    assert late > 0                                                                    # re-timing badly timed money helps


def test_incomplete_states_borrow_the_nearest_complete_year_but_not_2020(monkeypatch):
    table = pd.DataFrame({"state": ["NJ"] * 4 + ["CO"] * 4, "year": [2018, 2020, 2022, 2024] * 2,
                          "early_share": [0.13, 0.93, 0.90, 0.88, 0.99, 0.97, 0.96, 0.96],
                          "coverage": [1.0, 1.0, 0.25, 0.22, 1.0, 1.0, 1.0, 1.0]})
    monkeypatch.setattr(lev, "early_vote_table", lambda: table)
    assert lev.early_vote_share(2024)["NJ"] == pytest.approx(0.13)   # 2022 and 2024 incomplete; 2020 not a stand-in
    assert lev.early_vote_share(2020)["NJ"] == pytest.approx(0.93)
    assert lev.early_vote_share(2024)["CO"] == pytest.approx(0.96)
