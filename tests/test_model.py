"""Fast checks on the math, using synthetic data only."""

import itertools
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from campaign import load_candidates as lc  # noqa: E402
from campaign import load_coordinated as lco  # noqa: E402
from campaign import fall_drift as fd  # noqa: E402
from campaign import load_early_vote as lev  # noqa: E402
from campaign import household_index as hi  # noqa: E402
from campaign import load_early_money as lem  # noqa: E402
from campaign import load_fred as lf  # noqa: E402
from campaign import load_presidential as lp  # noqa: E402
from campaign import load_results as lres  # noqa: E402
from campaign.allocate import allocate  # noqa: E402
from campaign.ballot_timing import election_day, persuasion_weight, timed_gain  # noqa: E402
from campaign.brand_test import add_money_bands, national_passthrough, scaled  # noqa: E402
from campaign.break_even import break_even_scale  # noqa: E402
from campaign.classify_purpose import by_rules, classify_text, classify_vendors, vendor_categories, vendor_key  # noqa: E402
from campaign.competitive_response import response  # noqa: E402
from campaign.export_dataset import display_name  # noqa: E402
from campaign.fit_baseline import fit_baseline  # noqa: E402
from campaign.fit_close import close_inc, fit_close  # noqa: E402
from campaign.fit_drift import fit_drift, random_walk  # noqa: E402
from campaign.fit_spread import fit_spread, race_sigma  # noqa: E402
from campaign.fetch import schedule_f_rows  # noqa: E402
from campaign.freeze_forecast import freeze, money_signal, money_terms, score  # noqa: E402
from campaign.implied_belief import implied_belief  # noqa: E402
from campaign import lookup_quality as lq  # noqa: E402
from campaign.load_oppexp import swap_memos  # noqa: E402
from campaign.market_spillover import spillover_fit  # noqa: E402
from campaign.mine_spending import agreement, benchmark_k, clr, cluster_from_seat, one_cloud, rules, share_clear, shares, shuffled_columns  # noqa: E402
from campaign.national_test import CYCLES, economics_only, national_test  # noqa: E402
from campaign.predict_margin import predict_margin  # noqa: E402
from campaign.fcc_prices import airtime_multiple, busiest_station, folder_check, party_files, price_ratio, reading_at, slot_cells  # noqa: E402
from campaign import fetch_fcc as ff  # noqa: E402
from campaign.fetch_fcc import market_key, station_markets  # noqa: E402
from campaign.load_fcc import days_mask, parse_lines, sponsor_kind  # noqa: E402
from campaign.load_donors import committee_donors, donor_key, gifts_through_june, parse_line, race_measures  # noqa: E402
from campaign.google_ads import add_google  # noqa: E402
from campaign import redrawn_retention as rr  # noqa: E402
from campaign.redrawn_retention import carried_over, changed_states, overlap, retention_bands, retention_slope, state_blocks, state_blocks_2010  # noqa: E402
from campaign.spending_features import committee_measures  # noqa: E402
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
        weball_row("H0AA02002", "HONDA, MICHAEL", "C", "DEM", "600000", "AA", "02"),             # won it, spent less
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
    assert picked.ici.to_dict() == {("AA-01", "D"): "I", ("AA-01", "R"): "C", ("AA-02", "D"): "I",
                                    ("AA-03", "R"): "C", ("AA-03", "D"): "C"}
    assert picked.cand_id[("AA-02", "D")] == "H0AA02002"         # the sitting member speaks for the party, not the bigger spender


def test_close_names_catch_misspellings_but_not_other_people():
    assert lc.close_name("DENHAM, JEFF", "JEFF DENHEM")
    assert lc.close_name("DELARUO, ROSA L", "ROSA L DELAURO")
    assert lc.close_name("ESTY, ELIZABETH", "ELIZABETH H ETSY")
    assert lc.close_name("LAMALFA, DOUG", "DOUG LA MALFA")
    assert lc.close_name("DEAN, MADELEINE", "MADELEINE DEAN CUNNANE")
    assert lc.close_name("HIGGINS, CAPTAIN CLAY", "CLAY HIGGINS")
    assert lc.close_name("JOHNSON, JAMES MICHAEL", '\\"MIKE\\" JOHNSON')
    assert not lc.close_name("JOHN, RICHARD MICHAEL MR.", '\\"MIKE\\" JOHNSON')
    assert not lc.close_name("DEAN, MARK", "MADELEINE DEAN CUNNANE")
    assert display_name("MCGUIRE, JOHN J. MR. III") == "JOHN J. MCGUIRE"


def test_nominees_get_their_own_fec_record(tmp_path, monkeypatch):
    def weball_row(cand_id, name, ici, party, disbursed, state, district):
        row = [""] * 30
        row[0], row[1], row[2], row[4], row[7], row[18], row[19] = cand_id, name, ici, party, disbursed, state, district
        return "|".join(row)
    (tmp_path / "weball14.txt").write_text("\n".join([
        weball_row("H0AA01001", "DENHAM, JEFF", "I", "REP", "900000", "AA", "01"),                # JEFF DENHEM on the ballot
        weball_row("H0AA02001", "FLEMING, JOHN C", "I", "REP", "2700000", "AA", "02"),            # not "RICK" JOHN
        weball_row("H4AA02002", "JOHN, RICHARD", "O", "REP", "190000", "AA", "02"),
        weball_row("H0AA09001", "SANCHEZ, LINDA", "I", "DEM", "1000000", "AA", "09"),             # filed under her old district
        weball_row("H0AA04001", "SANCHEZ, LORETTA", "I", "DEM", "2000000", "AA", "04"),
        weball_row("H4AA05001", "MCAFFREY, ALVIS", "C", "DEM", "67000", "AA", "05"),              # one person, two records
        weball_row("H4AA05002", "MCAFFREY, AL", "O", "DEM", "700", "AA", "05"),
        weball_row("H4AA08001", "MARTINEZ, DIEGO", "C", "REP", "2000", "AA", "08")]) + "\n")     # no record for Daniel
    (tmp_path / "weball16.txt").write_text(weball_row("H2AA06001", "SELF, KEITH ALAN", "I", "REP", "800000", "AA", "06") + "\n"
                                           + weball_row("H6AA08001", "SELF, KENNETH", "C", "REP", "800", "AA", "08") + "\n")
    (tmp_path / "ccl14.txt").write_text("H2AA06001|2014|2014|C1|H|P|1\nH6AA08001|2014|2014|C2|H|P|2\n")
    (tmp_path / "reports_2014.csv").write_text(
        "committee_id,report_type,coverage_end_date,receipt_date,file_number,total_receipts_ytd,"
        "total_disbursements_ytd,candidate_contribution_ytd,loans_made_by_candidate_ytd\n"
        "C1,Q2,2014-06-30T00:00:00,2014-07-15,1,554000,516000,0,0\nC2,Q2,2014-06-30T00:00:00,2014-07-15,2,900,800,0,0\n")
    mit = [(2012, "06", "KEITH SELF", "REPUBLICAN"),
           (2014, "01", "JEFF DENHEM", "REPUBLICAN"), (2014, "02", '\\"RICK\\" JOHN', "REPUBLICAN"),
           (2014, "03", "LINDA SANCHEZ", "DEMOCRAT"), (2014, "04", "LORETTA SANCHEZ", "DEMOCRAT"),
           (2014, "05", "AL MCAFFREY", "DEMOCRAT"), (2014, "06", "KEITH SELF", "REPUBLICAN"),
           (2014, "07", "DANIEL MARTINEZ", "REPUBLICAN")]
    pd.DataFrame([{"year": y, "stage": "GEN", "mode": "TOTAL", "runoff": False, "special": False, "writein": False, "state_po": "AA",
                   "district": d, "candidate": c, "party": p, "candidatevotes": 100} for y, d, c, p in mit]).to_csv(tmp_path / "house.csv", index=False)
    for module in (lc, lem):
        monkeypatch.setattr(module, "RAW", tmp_path)
    monkeypatch.setattr(lres, "RAW", tmp_path / "house.csv")
    picked = lc.load_candidates(2014, lres.load_results(2014)).set_index("district_id")
    assert picked.cand_id.to_dict() == {"AA-01": "H0AA01001", "AA-02": "H4AA02002", "AA-03": "H0AA09001", "AA-04": "H0AA04001",
                                        "AA-05": "H4AA05001", "AA-06": "H2AA06001"}
    assert picked.ici["AA-02"] == "O"
    assert picked.ici["AA-06"] == "I" and picked.disbursements["AA-06"] == 516000      # won in 2012; June 30 spending stands in


def test_later_files_link_only_campaigns_with_no_earlier_link(tmp_path, monkeypatch):
    (tmp_path / "reports_2018.csv").write_text(
        "committee_id,report_type,coverage_end_date,receipt_date,file_number,total_receipts_ytd,"
        "total_disbursements_ytd,candidate_contribution_ytd,loans_made_by_candidate_ytd\n"
        "C1,Q2,2018-06-30T00:00:00,2018-07-15,1,100,40,0,0\nC2,Q2,2018-06-30T00:00:00,2018-07-15,2,900,70,0,0\n"
        "C3,Q2,2018-06-30T00:00:00,2018-07-15,3,300,50,0,0\n")
    (tmp_path / "ccl18.txt").write_text("H1|2018|2018|C1|H|P|1\n")
    (tmp_path / "ccl22.txt").write_text("H1|2022|2022|C2|H|P|2\nH2|2022|2022|C3|H|P|3\n")
    monkeypatch.setattr(lem, "RAW", tmp_path)
    money = lem.load_early_money(2018)
    assert money.raised.to_dict() == {"H1": 100, "H2": 300}


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


def money_rows(raised_d, raised_r, spent_d=0.0, spent_r=0.0):
    zero = np.zeros(len(raised_d))
    return pd.DataFrame({"raised_jun_d": raised_d, "raised_jun_r": raised_r, "spent_jun_d": spent_d, "spent_jun_r": spent_r,
                         "own_jun_d": zero, "own_jun_r": zero, "outside_aug_d": zero, "outside_aug_r": zero})


def test_money_terms_treat_the_two_parties_alike():
    terms = money_terms(money_rows(np.array([1.5e6, 0.0]), np.array([0.0, 1.5e6])))
    assert terms.money_odds[0] == pytest.approx(-terms.money_odds[1])          # floor_share gives -0.007 and -5.0
    assert terms.surplus[0] == pytest.approx(-terms.surplus[1])
    spent = money_terms(money_rows(np.array([1.5e6]), np.array([1.5e6]), np.array([1.3e6]), np.array([2.0e6])))
    assert spent.money_odds[0] == pytest.approx(0.0)                           # raising the same, spending differently
    assert spent.surplus[0] == pytest.approx(np.log(210_000 / 10_000))         # overspending counts as no surplus


def test_a_freeze_is_not_overwritten(tmp_path):
    (tmp_path / "freeze.csv").write_text("already here")
    with pytest.raises(SystemExit):
        freeze({}, stem=tmp_path / "freeze")
    assert (tmp_path / "freeze.csv").read_text() == "already here"


def test_score_compares_the_same_close_races_and_can_take_out_the_national_miss():
    mu = np.array([5.0, -5.0, 30.0, -30.0])
    frozen = pd.DataFrame({"district_id": list("abcd") + ["no result"], "mu_baseline": [*mu, 0.0],
                           "mu_shifted": [*(mu + 4.0), 0.0], "mu_far": [20.0, -20.0, 30.0, -30.0, 0.0]})
    fit = {"sigma": 8.0, "sigma_national": 3.0}
    margin = pd.Series([6.0, -4.0, 31.0, -29.0], index=list("abcd"))
    scores = score(frozen, margin, {"baseline": fit, "shifted": fit, "far": fit})
    assert scores.national_miss["baseline"] == pytest.approx(1.0)
    assert scores.national_miss["shifted"] == pytest.approx(-3.0)
    assert scores.log_loss_within["shifted"] == pytest.approx(scores.log_loss_within["baseline"])
    assert scores.log_loss["shifted"] != pytest.approx(scores.log_loss["baseline"])
    p = norm.cdf(20.0 / 8.0)                                                   # "far" is judged on a and b, not its own close set
    assert scores.log_loss_close["far"] == pytest.approx(-np.log(p))


def test_memo_rows_on_schedule_f_are_not_counted_twice():
    row = ["SF"] + [""] * 43
    paid, memo = list(row), list(row)
    paid[2], paid[28] = "SF-1", "100.00"
    memo[2], memo[28], memo[42] = "SF-1-10000", "100.00", "X"
    other = ["SB2"] + [""] * 43
    lines = [",".join(r) for r in (paid, memo, other)]
    assert [r["tran_id"] for r in schedule_f_rows(lines)] == ["SF-1"]


def test_coordinated_money_goes_to_the_nominee_race_and_splits_only_media_from_mail(tmp_path, monkeypatch):
    purposes = ["MEDIA BUY", "MEDIA", "ADVERTISING", "MEDIA BUY MEDIA PRODUCTION", "MAIL SERVICES", "POSTAGE/ PRINTING",
                "MEDIA PRODUCTION", "TRAVEL"]
    rows = pd.DataFrame({"bucket": "dccc", "cand_id": ["H1", "H2"] + ["H1"] * 6, "state": "FL",
                         "district": ["23", "07"] + ["23"] * 6, "amount": 1.0, "date": 20261001, "purpose": purposes})
    rows.to_csv(tmp_path / "coordinated_2026.csv", index=False)
    monkeypatch.setattr(lco, "RAW", tmp_path)
    monkeypatch.setattr(lco, "load_results", lambda cycle: pd.DataFrame())
    monkeypatch.setattr(lco, "load_candidates", lambda cycle, results: pd.DataFrame(
        {"cand_id": ["H1"], "party": ["D"], "district_id": ["FL-25"]}))
    out = lco.load_coordinated(2026)
    assert out.district_id.tolist() == ["FL-25", "FL-07"] + ["FL-25"] * 6     # a new number, then a primary loser's filing
    assert out.channel.tolist() == ["Media"] * 4 + ["Mail"] * 2 + ["Other"] * 2


def test_money_signal_compares_races_by_who_spent_after_taking_out_the_miss():
    frozen = pd.DataFrame({"district_id": list("abcd"), "mu_baseline": 10.0, "p_baseline": 0.9,
                           "dccc_coordinated": [0.0, 1.0, 0.0, 1.0], "nrcc_coordinated": [0.0, 0.0, 1.0, 1.0]})
    margin = pd.Series([12.0, 10.0, 8.0, 6.0], index=list("abcd"))           # the average miss is -1
    signal = money_signal(frozen, margin).loc["80-99% D"]
    assert signal.resid["neither"] == pytest.approx(3.0)
    assert signal.resid["both"] == pytest.approx(-3.0)
    assert signal.n.sum() == 4


@pytest.mark.parametrize("purpose, category", [
    ("DIRECT MAIL FUNDRAISING", "Fundraising"), ("EMAIL", "Digital"), ("PAYROLL PROCESSING FEES", "Payroll/admin"),
    ("CREDIT CARD PROCESSING FEE", "Processing fees"), ("CREDIT CARD PAYMENT - SEE BELOW", "Pass-through"), ("POSTAGE", "Mail/print"),
    ("MEDIA BUY", "Broadcast"), ("TV AND RADIO ADVERTISING PRODUCTION", "Production"), ("CELL PHONE SERVICE", "Payroll/admin"),
    ("PHONE BANK", "Field/phones"), ("AIRLINE TICKETS", "Payroll/admin"), ("EVENT TICKETS", "Events"), ("STRATEGY CONSULTING", "Consulting/unspecified"),
    ("POLLING", "Polling/research"), ("DUES", "Contributions"), ("", "Other")])
def test_purposes_fall_in_the_right_category(purpose, category):
    assert by_rules(purpose) == category


def test_fec_codes_settle_only_purposes_with_no_rule():
    found = classify_text(pd.Series(["STRATEGY CONSULTING", "MISC", "MISC"]), pd.Series(["001", "005", None]))
    assert found.tolist() == ["Consulting/unspecified", "Polling/research", "Other"]     # filers code consultants as overhead


def test_only_a_vendor_with_a_clear_record_settles_vague_payments():
    vendors = pd.Series(["TV CO", "TV CO", "MIXED", "MIXED", "SMALL"])
    known = vendor_categories(vendors, pd.Series(["Broadcast", "Broadcast", "Broadcast", "Digital", "Broadcast"]),
                              pd.Series([20_000.0, 20_000.0, 30_000.0, 30_000.0, 1_000.0]))
    assert known.to_dict() == {"TV CO": "Broadcast"}
    out = classify_vendors(pd.Series(["TV CO", "TV CO", "MIXED"]), pd.Series(["Consulting/unspecified", "Digital", "Consulting/unspecified"]), known)
    assert out.tolist() == ["Broadcast", "Digital", "Consulting/unspecified"]
    assert vendor_key(pd.Series(["The Frost Group, LLC"])).iloc[0] == "FROST GROUP"


def test_a_payments_memo_rows_replace_it_without_changing_the_total():
    rows = pd.DataFrame({"CMTE_ID": "C1", "FILE_NUM": "1", "TRAN_ID": ["P1", "M1", "M2", "P2", "P3", "M3", "ORPHAN"],
                         "BACK_REF_TRAN_ID": [None, "P1", "P1", None, None, "P3", "NOPE"], "MEMO_CD": [None, "X", "X", None, None, "X", "X"],
                         "amount": [100.0, 30.0, 50.0, 40.0, 10.0, 20.0, 99.0]})
    out = swap_memos(rows).set_index("TRAN_ID").amount
    assert out.sum() == pytest.approx(150.0)
    assert out.to_dict() == pytest.approx({"P1": 20.0, "M1": 30.0, "M2": 50.0, "P2": 40.0, "M3": 10.0})   # P3's memo exceeded it, so it scales down


def test_centered_log_ratios_ignore_budget_and_sum_to_zero():
    rows = pd.DataFrame({"committee_id": ["a", "a", "b", "b"], "category": ["Broadcast", "Digital"] * 2, "amount": [1.0, 1.0, 1000.0, 1000.0]})
    table = shares(rows)
    assert table.total.to_dict() == {"a": 2.0, "b": 2000.0}
    ratios = clr(table)
    assert ratios.sum(axis=1).to_numpy() == pytest.approx([0.0, 0.0])
    assert ratios.loc["a"].to_numpy() == pytest.approx(ratios.loc["b"].to_numpy())


def test_conditional_lift_removes_what_budget_explains_and_rules_run_forward():
    rng = np.random.default_rng(0)
    big = rng.random(4000) < 0.5                         # big campaigns buy both, independently of each other
    table = pd.DataFrame({"Q2 Polling": rng.random(4000) < np.where(big, 0.8, 0.1), "Q3 Broadcast": rng.random(4000) < np.where(big, 0.8, 0.1)})
    found = rules(table, pd.Series(big), min_support=0.01, min_confidence=0.1).set_index("if")
    assert found.lift["Q2 Polling"] > 1.5
    assert found.conditional_lift["Q2 Polling"] == pytest.approx(1.0, abs=0.05)
    assert "Q3 Broadcast" not in found.index             # a later phase can't lead to an earlier one


def test_campaign_measures_read_combinations_order_and_vendors():
    rows = pd.DataFrame({"committee_id": ["a", "a", "a", "b", "b"], "vendor": ["V1", "V2", "V3", "V1", "V1"], "amount": [50.0, 25.0, 25.0, 90.0, 10.0],
                         "category": ["Digital", "Mail/print", "Polling/research", "Broadcast", "Polling/research"],
                         "date": pd.to_datetime(["2024-03-01", "2024-05-01", "2024-01-01", "2024-01-01", "2024-02-01"])})
    out, used = committee_measures(2024, rows, {("a", "V1")})
    assert out.digital_mail.to_dict() == {"a": 1.0, "b": 0.0}
    assert out.polling_digital["a"] == 1.0
    assert out.poll_first.to_dict() == {"a": 1.0, "b": 0.0}                    # a polled in January, b started on TV first
    assert out.vendor_hhi["b"] == pytest.approx(1.0)
    assert out.vendor_hhi["a"] == pytest.approx(0.375)
    assert out.vendor_retention.to_dict() == pytest.approx({"a": 0.5, "b": 0.0})
    assert out.contact_lead["b"] == pytest.approx(181 / 365)
    assert ("b", "V1") in used


def test_a_real_response_shows_in_last_weeks_money_and_not_next_weeks():
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product([[f"R{i}" for i in range(150)], range(12)], names=["district_id", "week"])
    nrcc = pd.Series(np.sinh(rng.normal(8, 2, len(index))), index=index)
    dccc = pd.Series(np.sinh(0.5 * np.arcsinh(nrcc.groupby(level=0).shift(1).fillna(0)) + rng.normal(4, 1, len(index))), index=index)
    found = response({2024: pd.DataFrame({"dccc": dccc, "nrcc": nrcc})}, "dccc", "nrcc")
    assert found["lag"][0] == pytest.approx(0.5, abs=0.05)
    assert abs(found["lead"][0]) < 0.05


def test_shared_markets_and_retention_recover_what_was_built_in():
    rng = np.random.default_rng(0)
    close = pd.DataFrame({"mu": rng.normal(0, 5, 400), "reach": rng.uniform(1, 30, 400), "n_other": rng.integers(0, 5, 400), "cycle": 2024, "market": np.arange(400)})
    close["committee"] = np.expm1(10 + 0.4 * close.n_other + rng.normal(0, 0.2, 400))
    assert spillover_fit(close)["n_other"] == pytest.approx(0.4, abs=0.05)
    blocks = pd.DataFrame({"old": ["A", "A", "B", "B"], "new": ["X", "X", "X", "Y"], "people": [10.0, 10.0, 10.0, 5.0]})
    shares_kept = overlap(blocks).set_index(["old", "new"]).retained
    assert shares_kept.to_dict() == pytest.approx({("A", "X"): 2 / 3, ("B", "X"): 1 / 3, ("B", "Y"): 1.0})
    table = pd.DataFrame({"cycle": 2022, "district_id": ["AL-01", "AL-02", "GA-01", "GA-02"], "retained": [0.2, 0.5, 0.8, 1.0], "resid": [-1.6, -1.0, -0.4, 0.0]})
    assert retention_slope(table)["slope"] == pytest.approx(2.0)
    apart = pd.concat([table, table.assign(cycle=2024, resid=table.resid + 10.0, retained=table.retained - 0.1)], ignore_index=True)
    assert retention_slope(apart)["slope"] == pytest.approx(2.0)
    assert retention_slope(table, within_state=True)["slope"] == pytest.approx(2.0)
    assert retention_slope(table.assign(resid=table.resid + [5.0, 5.0, -5.0, -5.0]), within_state=True)["slope"] == pytest.approx(2.0)
    assert retention_bands(table)["size"].tolist() == [1, 1, 1, 1]
    before, after = carried_over(table, table)
    assert after < before


def test_pass_through_scales_only_the_elections_it_names_and_is_recovered_nationally():
    races = {2022: pd.DataFrame({"gb": [4.0]}), 2024: pd.DataFrame({"gb": [4.0]})}
    out = scaled(races, midterm=1.0, presidential=0.5)
    assert out[2022].gb.iloc[0] == 4.0 and out[2024].gb.iloc[0] == 2.0
    polls = np.array([-9.0, -3.0, -1.0, 0.0, 2.0, 4.0, 6.0, 8.0])
    table = pd.DataFrame({"gb": polls, "gb_sept": polls + 1.0, "house_vote": 0.8 * polls - 1.0}, index=range(2010, 2026, 2))
    found = national_passthrough(table).set_index(["poll", "group"])
    assert found.k["gb", "All"] == pytest.approx(0.8)
    assert found.k["gb", "Presidential"] == pytest.approx(0.8)
    assert found.n["gb", "Midterm"] == 4


def test_money_bands_mark_low_money_and_separate_hindsight_from_what_september_sees():
    x = pd.DataFrame({"pvi": [10.0, -10.0], "raised_jun_d": [100_000.0, 2_000_000.0], "raised_jun_r": [100_000.0, 0.0], "outside_aug_d": [0.0, 0.0], "outside_aug_r": [0.0, 500_000.0],
                      "disbursements_d": [3_000_000.0, 100.0], "disbursements_r": [0.0, 100.0], "outside_d": [0.0, 0.0], "outside_r": [0.0, 0.0]})
    out = add_money_bands({2024: x})[2024]
    assert out.low_money.tolist() == [1.0, 0.0] and out.low_all.tolist() == [0.0, 1.0]
    assert out.pvi_low.tolist() == [10.0, 0.0] and out.pvi_low_all.tolist() == [0.0, -10.0]
    assert out.pvi_curve.tolist() == pytest.approx([10.0, -10.0])


def test_fcc_lines_read_both_layouts_and_keep_the_slot_across_week_rows():
    invoice = """Line   Start Date   End Date   Description   Start/End Time   MTWTFSS
   51 10/22/24   10/28/24   CBS THIS MORNING   7:00 AM-9:00 AM   78777--
       Weeks:       Start Date End Date     MTWTFSS         Spots/Week        Rate
                    10/22/24    10/28/24    78777--                 36      $80.00
 Spots: #   Ch   Day Air Date   Air Time Description    Start/End Time
       24   WTOL M 10/28/24     7:24 AM CBS THIS MORNING   7:00 AM-9:00 AM
                    10/29/24    11/04/24    78777--                 30      $95.00
"""
    order = """N 1   WILX   09/17/24 09/23/24 News 10 Today @ 5a CM     5:00 AM-5:30 AM MTWTF--     :30       3      $250.00 P-03
    CS 3Q EST - LANSING - ACTUAL Sep/2023 Mo-Fr 5:00a-5:30a TIME PERIOD ON WILX
        Start Date End Date     Weekdays      Spots/Week      Rate    Rating
  Week: 09/17/24     09/23/24   MTWTF--                3 $250.00        0.00
"""
    first = parse_lines(invoice)
    assert [(r["week"], r["spots"], r["rate"], r["slot"]) for r in first] == [("10/22/24", 36, 80.0, "7:00AM-9:00AM"), ("10/29/24", 30, 95.0, "7:00AM-9:00AM")]
    second = parse_lines(order)
    assert [(r["spots"], r["rate"], r["slot"], r["seconds"]) for r in second] == [(3, 250.0, "5:00AM-5:30AM", 30)]
    assert days_mask("MTWTF--") == days_mask("22222--") == "xxxxx--"
    aired = """Line Channel Description    Time    Day  Date   Length Air Time   Ad-ID   Rate  Reconciliation  Ref #
   1   WVIT   5-530A CT NEWS TODAY   5:00 AM-5:30 AM
         All spots on this line are Immediately Preemptible class of time
   08/26/24 to 09/01/24   4x   -1111--
   WVIT   Tu   08/27/24   :30   5:09 AM   SGLCT819H   $250.00   1
   WVIT   W    08/28/24   :30   5:25 AM   SGLCT819H   $250.00   2
   2   WVIT   CT NEWS AT 6   6:00 PM-6:30 PM
   08/26/24 to 09/01/24   1x   1------
   WVIT   M    08/26/24   :15   6:09 PM   SGLCT819H   $900.00   1
"""
    third = parse_lines(aired)
    assert [(r["week"], r["days"], r["spots"], r["rate"], r["slot"], r["seconds"], r["preemptible"]) for r in third] == [
        ("08/26/24", "-1111--", 1, 250.0, "5:00AM-5:30AM", 30, True), ("08/26/24", "-1111--", 1, 250.0, "5:00AM-5:30AM", 30, True), ("08/26/24", "1------", 1, 900.0, "6:00PM-6:30PM", 15, False)]


def test_sponsor_kinds_come_from_how_stations_name_folders():
    assert sponsor_kind("house", "Bresnahan-NRCC") == "coordinated"
    assert sponsor_kind("house", "Hertel for Michigan - DCCC") == "coordinated"
    assert sponsor_kind("house", "Cartwright-D-US House") == "candidate"
    assert sponsor_kind("house", "Valadao for Congress") == "candidate"
    assert sponsor_kind("house", "House Majority PAC") == "outside"
    assert sponsor_kind("non_candidate", "NRCC-IE") == "party_independent"
    assert sponsor_kind("non_candidate", "Congressional Leadership Fund") == "outside"
    assert sponsor_kind("house", "Zzyzx") == "unknown"
    assert sponsor_kind("house", "Tranel - Democrat - Congress") == "candidate"
    assert sponsor_kind("house", "Committee to Elect Jared Golden") == "candidate"
    assert sponsor_kind("house", "Juan Ciscomani") == "candidate"
    assert sponsor_kind("house", "Conley House Victory Fund") == "unknown"
    assert sponsor_kind("house", "OHIO") == "unknown"
    assert sponsor_kind("house", "NRCC") == "unknown"
    assert sponsor_kind("house", "NRCC IE") == "party_independent"
    assert sponsor_kind("non_candidate", "NRCC Robert Bresnahan") == "unknown"
    assert sponsor_kind("non_candidate", "Democratic Congressional Campaign Committee") == "party_independent"


def test_prices_compare_only_the_same_slot_days_week_and_length():
    rows = [("A", "6PM", "MTWTF--", 30, "candidate", 100.0), ("A", "6PM", "MTWTF--", 30, "party_independent", 400.0), ("A", "6PM", "--S----", 30, "party_independent", 900.0),
            ("A", "7PM", "MTWTF--", 60, "candidate", 50.0), ("A", "7PM", "MTWTF--", 60, "party_independent", 900.0), ("B", "6PM", "MTWTF--", None, "candidate", 100.0), ("B", "6PM", "MTWTF--", None, "party_independent", 200.0)]
    lines = pd.DataFrame(rows, columns=["station", "slot", "days", "seconds", "kind", "rate"]).assign(week=pd.Timestamp("2024-10-07"))
    found = price_ratio(slot_cells(lines), "party_independent", "candidate", draws=50)
    assert found["n"] == 2 and found["stations"] == 2 and found["median"] == pytest.approx(3.0)
    assert price_ratio(slot_cells(lines), "coordinated", "candidate", draws=5)["n"] == 0


def test_typical_price_counts_each_station_once_however_busy():
    rows = [("BUSY", f"{h}PM", "MTWTF--", 30, kind, rate) for h in range(1, 10) for kind, rate in (("candidate", 100.0), ("outside", 500.0))]
    rows += [(s, "6PM", "MTWTF--", 30, kind, rate) for s in ("A", "B") for kind, rate in (("candidate", 100.0), ("outside", 200.0))]
    lines = pd.DataFrame(rows, columns=["station", "slot", "days", "seconds", "kind", "rate"]).assign(week=pd.Timestamp("2024-10-07"))
    found = price_ratio(slot_cells(lines), "outside", "candidate", draws=50)
    assert found["median"] == pytest.approx(5.0) and found["typical"] == pytest.approx(2.0)


def test_busiest_station_is_named_and_set_aside_from_the_figure_without_it():
    rows = [("BUSY", f"{h}PM", "MTWTF--", 30, kind, rate) for h in range(1, 10) for kind, rate in (("party_independent", 100.0), ("coordinated", 100.0))]
    rows += [("BUSY", "9AM", "MTWTF--", 30, kind, rate) for kind, rate in (("party_independent", 300.0), ("candidate", 100.0))]
    rows += [(s, "6PM", "MTWTF--", 30, kind, rate) for s in ("A", "B") for kind, rate in (("party_independent", 300.0), ("coordinated", 100.0))]
    lines = pd.DataFrame(rows, columns=["station", "slot", "days", "seconds", "kind", "rate"]).assign(week=pd.Timestamp("2024-10-07"))
    found = busiest_station(slot_cells(lines), "party_independent", "coordinated")
    assert (found["station"], found["n"]) == ("BUSY", 9) and found["median"] == pytest.approx(1.0)
    assert found["against_candidate"] == pytest.approx(3.0) and found["without"]["median"] == pytest.approx(3.0) and found["without"]["n"] == 2
    assert busiest_station(slot_cells(lines), "party_independent", "outside") is None


def test_cluster_benchmarks_keep_only_what_each_is_meant_to():
    base = np.random.default_rng(0).normal(size=500)
    x = np.column_stack([base, base + np.random.default_rng(1).normal(scale=0.1, size=500)])
    mixed = shuffled_columns(x, np.random.default_rng(2))
    assert np.array_equal(np.sort(mixed, axis=0), np.sort(x, axis=0)) and abs(np.corrcoef(mixed.T)[0, 1]) < 0.2
    cloud = one_cloud(x, np.random.default_rng(3))
    assert cloud.shape == x.shape and np.corrcoef(cloud.T)[0, 1] > 0.9
    scores = benchmark_k(x, shuffled_columns, ks=[2], draws=2)
    assert list(scores.index) == [2] and set(scores.columns) == {"silhouette", "stability"}


def test_agreement_ignores_which_side_is_which():
    assert agreement([True, True, False, False], [False, False, True, True]) == 1.0
    assert agreement([True, True, False, False], [True, False, False, False]) == 0.75


def test_cluster_from_seat_finds_a_cluster_seat_type_settles():
    frame = pd.DataFrame({"ici": ["I", "C"] * 30, "cluster": [0, 1] * 30, "total": 1e6 * np.arange(1.0, 61.0)})
    found = cluster_from_seat(frame)
    assert found["seat"] == 1.0 and found["largest"] == 0.5


def test_share_clear_counts_campaigns_above_the_floor():
    x = np.array([[0.0], [0.01], [10.0], [10.01], [5.0]])
    assert share_clear(x, np.array([0, 0, 1, 1, 1])) == 0.8


def test_party_files_keep_a_party_with_a_candidate_apart_from_other_unknowns():
    rows = [("house", "DCCC Pat Smith", "1"), ("non_candidate", "DCCC", "2"), ("non_candidate", "NRCC Pat Smith", "3"), ("house", "Ohio", "4"), ("house", "Pat Smith", "4")]
    manifest = pd.DataFrame(rows, columns=["section", "sponsor", "file_id"]).assign(extension="PDF", entity_id="e")
    files = party_files(manifest).files
    assert files.to_dict() == {"coordinated": 1, "party_independent": 1, "party_unclear": 1, "unknown": 1}


def test_folder_check_finds_repeats_and_drops_candidate_documents_naming_a_party(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "RAW", tmp_path)
    sample = pd.DataFrame({"year": "2024", "station": "WAAA", "file_manager_id": list("abcd"), "extension": "pdf", "file_id": list("abcd"), "kind": ["candidate", "candidate", "outside", "coordinated"]})
    for row, body in zip(sample.itertuples(), (b"own buy", b"paid for with NRCC", b"own buy", b"NRCC")):
        path = ff.file_path(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        path.with_suffix(".pdf.txt").write_bytes(body)
    lines = pd.DataFrame({"station": "WAAA", "slot": "6PM", "days": "MTWTF--", "seconds": 30, "week": pd.Timestamp("2024-10-07"), "file_id": list("abcd"),
                          "kind": ["candidate", "candidate", "outside", "coordinated"], "rate": [100.0, 300.0, 400.0, 100.0]})
    found = folder_check(sample, lines, 2024)
    assert found["repeats"] == 2
    assert found["names_party"]["candidate"] == 0.5 and found["names_party"]["outside"] == 0.0
    assert found["without"]["outside"]["median"] == pytest.approx(4.0)


def test_airtime_multiple_stretches_only_the_media_share_and_the_curve_with_it():
    assert airtime_multiple(0.8, 2.0) == pytest.approx(1.8) and airtime_multiple(0.0, 2.0) == 1.0
    mu, spent = np.array([1.0, 3.0, 12.0, -5.0]), np.array([2e6, 2e6, 4e6, 2e6])
    same = reading_at(mu, 7.0, spent, 1.0, 0.5)
    assert same["gain"] == pytest.approx(reallocation_gain(mu, 7.0, spent, 5e6)[0])
    assert reading_at(mu, 7.0, spent, 2.0, 0.5)["gain"] > same["gain"]


def test_fcc_download_stops_at_a_refusal_and_keeps_what_it_has(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "RAW", tmp_path)
    calls = []

    class Answer:
        status_code, content = 403, b"no"

    def refused(url, **kwargs):
        calls.append(url)
        return Answer()
    monkeypatch.setattr(ff.requests, "get", refused)
    monkeypatch.setattr(ff.time, "sleep", lambda s: None)
    rows = pd.DataFrame({"year": "2024", "station": "WAAA", "folder_id": "f", "file_manager_id": [f"m{i}" for i in range(5)], "extension": "pdf"})
    assert ff.download(rows, workers=1)[2] is True
    assert len(calls) == 1
    assert not list(tmp_path.rglob("*.pdf"))


def test_fcc_markets_match_despite_spelling_and_shared_first_cities():
    assert market_key("WILKES BARRE-SCRANTON-HZTN")[:2] == market_key("Wilkes Barre-Scranton, PA")[:2]
    assert market_key("Sioux Falls(Mitchell)") == market_key("Sioux Falls (Mitchell), SD")
    stations = pd.DataFrame({"id": ["1", "2", "3", "4"], "dma": ["PORTLAND OR", "PORTLAND-AUBURN", "COLUMBIA SC", "WILKES BARRE-SCRANTON-HZTN"], "state": ["OR", "ME", "SC", "PA"]})
    chosen, missed = station_markets(stations, ["Portland, OR", "Wilkes Barre-Scranton, PA", "Nowhere, ZZ"])
    assert sorted(chosen.id) == ["1", "4"] and missed == ["Nowhere, ZZ"]


def test_google_matches_committees_by_fec_id_and_leaves_earlier_years_at_even():
    export = pd.DataFrame({"public_ids": ["FEC ID C00000001", "EIN ID 12-3456789", None], "year": 2022, "spend_before_sep1": [90_000.0, 50_000.0, 50_000.0], "spend_all_year": [99_000.0, 50_000.0, 50_000.0]})
    races = {2018: pd.DataFrame({"cm_d": ["C00000001"], "cm_r": [None]}), 2022: pd.DataFrame({"cm_d": ["C00000001"], "cm_r": [None]})}
    out = add_google(races, export)
    assert out[2022].google_d.iloc[0] == 90_000.0 and out[2022].google_r.iloc[0] == 0.0
    assert out[2022].google_odds.iloc[0] == pytest.approx(np.log(10.0))
    assert out[2018].google_odds.iloc[0] == 0.0


def test_state_blocks_reads_people_and_lines_from_the_redistricting_file(tmp_path):
    geo = ["PLST", "AL", "750", "00", "00", "000", "00", "0000002", "x", "010010201001000"] + [""] * 52 + ["02"] + [""] * 34
    state_level = list(geo)
    state_level[2], state_level[7] = "040", "0000001"
    empty_block = list(geo)
    empty_block[7], empty_block[9] = "0000003", "010010201001001"
    counts = "PLST|AL|000|01|0000001|999\nPLST|AL|000|01|0000002|7\nPLST|AL|000|01|0000003|0\n"
    with zipfile.ZipFile(tmp_path / "al2020.pl.zip", "w") as z:
        z.writestr("algeo2020.pl", "\n".join("|".join(r) for r in (state_level, geo, empty_block)) + "\n")
        z.writestr("al000012020.pl", counts)
    blocks = state_blocks(tmp_path / "al2020.pl.zip")
    assert blocks.to_dict("records") == [{"block": "010010201001000", "people": 7.0, "base": "AL-02"}]


def test_2010_blocks_and_national_lines_and_which_states_changed(tmp_path, monkeypatch):
    def geo(level, record, block, cd):
        line = [" "] * 200
        for start, text in ((8, level), (18, record), (27, "01"), (29, "001"), (54, "021100"), (61, block), (153, cd)):
            line[start:start + len(text)] = text
        return "".join(line)
    with zipfile.ZipFile(tmp_path / "al2010.pl.zip", "w") as z:
        z.writestr("algeo2010.pl", "\n".join([geo("040", "0000001", "    ", "  "), geo("750", "0000002", "2000", "02"), geo("750", "0000003", "2001", "02")]) + "\n")
        z.writestr("al000012010.pl", "PLST,AL,000,01,0000001,99\nPLST,AL,000,01,0000002,7\nPLST,AL,000,01,0000003,0\n")
    assert state_blocks_2010(tmp_path / "al2010.pl.zip").to_dict("records") == [{"block": "010010211002000", "people": 7.0, "base": "AL-02"}]
    (tmp_path / "bef").mkdir()
    with zipfile.ZipFile(tmp_path / "bef" / "cd120.zip", "w") as z:
        z.writestr("CD120_01.txt", "GEOID,CDFP\n010010201001000,99\n")
        z.writestr("NationalCD120.txt", "GEOID,STATEFP,COUNTYFP,TRACTCE,BLOCKCE,CDFP\n010010201001000,01,001,020100,1000,03\n")
    monkeypatch.setattr(rr, "RAW", tmp_path)
    assert rr.block_lines(120).to_dict("records") == [{"block": "010010201001000", "district": "AL-03"}]
    blocks = pd.DataFrame({"old": ["AL-01", "AL-01", "GA-01", "GA-01"], "new": ["AL-01", "AL-02", "GA-01", "GA-02"], "people": [90.0, 10.0, 999.0, 1.0]})
    assert changed_states(blocks) == {"AL"}


def test_new_voters_is_signed_toward_the_democrat_and_zero_without_an_incumbent(monkeypatch):
    kept = pd.DataFrame({"district_id": ["AL-01", "AL-02"], "retained": [0.6, 0.8]})
    monkeypatch.setattr(rr, "incumbent_retention", lambda cycle: kept)
    races = pd.DataFrame({"district_id": ["AL-01", "AL-02", "AL-03", "AL-04"], "status": ["D incumbent", "R incumbent", "Open", "R incumbent"]})
    out = rr.add_new_voters({2022: races})[2022]
    assert out.new_voters.tolist() == pytest.approx([-0.4, 0.2, 0.0, 0.0])
    assert out.new_voters_within.tolist() == pytest.approx([-0.1, -0.1, 0.0, 0.0])


def test_donor_measures_count_people_not_gifts_and_ignore_refunded_donors():
    assert donor_key("SMITH, JOHN Q. MR.", "606011234") == donor_key("Smith, John", "60601")
    assert donor_key("SMITH, JOHN", "60601") != donor_key("SMITH, JANE", "60601")
    line = ["C00000001", "N", "Q2", "P", "1", "15", "IND", "SMITH, JOHN", "X", "IL", "60601", "", "", "06152024", "500", "", "t", "1", "", "", "9"]
    assert parse_line(("|".join(line) + "\n").encode())[:5] == ("C00000001", "15", "IL", "06152024", 500.0)
    assert parse_line(("|".join(line[:18] + ["X"] + line[19:]) + "\n").encode()) is None
    assert parse_line(("|".join(line[:6] + ["ORG"] + line[7:]) + "\n").encode()) is None
    rows = pd.DataFrame({"committee_id": ["a", "a", "a", "a", "b"], "state": ["IL", "IL", "CA", "IL", "IL"], "donor": ["p", "p", "q", "gone", "p"],
                         "amount": [100.0, 100.0, 600.0, 0.0, 50.0], "date": "06152024"})
    earlier = pd.DataFrame({"committee_id": ["a"], "donor": ["q"]})
    donors = committee_donors(rows, earlier)
    measures = race_measures(donors, pd.Series({"IL-01": "a", "IL-02": None, "IL-03": "b"}), pd.Series({"IL-01": "IL", "IL-02": "IL", "IL-03": "IL"}))
    assert measures.loc["IL-01"].to_dict() == pytest.approx({"donors": 2, "local": 1, "top10": 1.0, "repeat": 0.75, "in_state": 0.25})
    assert measures.loc["IL-02"].to_dict() == pytest.approx({"donors": 0, "local": 0, "top10": 0, "repeat": 0, "in_state": 0})
    assert gifts_through_june(pd.DataFrame({"date": ["12312023", "06302024", "07012024", "12312022", "bad"]}), 2024).date.tolist() == ["12312023", "06302024"]
