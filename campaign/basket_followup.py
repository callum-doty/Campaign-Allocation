"""
Checks added after the real run of basket_search.py, and the 2026 record they lead to.
None of this was in BASKET_DESIGN.md's rules, and none of it changes their verdicts. It
changes how the committee test reads.

    python -m campaign.basket_followup

Writes data/basket_followup.json and, once, data/basket_nearline_2026.csv, the seats
behind the predictions in BASKET_DESIGN.md.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import basket_search as b

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = ROOT / "data" / "basket_followup.json"
RECORD = ROOT / "data" / "basket_nearline_2026.csv"
NEAR = 0.95
GROUPS = ("defender", "attacker", "both", "neither")
SHUFFLES = 20000


def who_spent(defend, attack):
    """Which committees spent in a seat, from the favored side's view."""
    return np.select([defend & attack, defend, attack], ["both", "defender", "attacker"], "neither")


def symmetric_seats(races, measures, side):
    """safe_seats on a forecast with the symmetric money term in place of floor_share."""
    original = b.FEATURES
    b.FEATURES = [("money_odds" if f == "floor_share" else f) for f in original]
    try:
        return b.safe_seats(races, measures, side=side)
    finally:
        b.FEATURES = original


def money_forms(races, measures):
    """The miss test under each money term, for each side."""
    out = {}
    for side in "dr":
        for name, seats in (("as_run", b.safe_seats(races, measures, side=side)), ("symmetric", symmetric_seats(races, measures, side))):
            out[f"{name}_{side}"] = {**b.miss_test(seats, b.left_out_scores(seats)), "entries": int(seats.entered.sum())}
    return out


def score_without(seats, flag):
    """left_out_scores with one flag out of the entry model."""
    columns, out = [f for f in b.FLAGS if f != flag] + ["mu_own"], pd.Series(0.0, index=seats.index)
    for y in b.BACKTEST:
        scale, model, _ = b.fit_entry(seats[seats.cycle != y], columns)
        test = seats[seats.cycle == y]
        out[test.index] = scale.transform(test[columns])[:, :-1] @ model.coef_[0][:-1]
    return out


def with_results(seats, races):
    return seats.merge(pd.concat([races[c][["cycle", "district_id", "margin", "dccc", "nrcc"]] for c in b.BACKTEST]), on=["cycle", "district_id"])


def seat_groups(seats, races):
    """Democratic safe seats by committee money and entry score: what the miss test's margins meant for seats."""
    x = with_results(seats.assign(score=b.left_out_scores(seats)), races)
    quiet = ~x.committee_money
    top = quiet & (x.score >= x.score[quiet].quantile(0.9))
    return {name: {"seats": int(mask.sum()), "forecast_margin": float(x.mu_own[mask].mean()), "forecast_win": float(x.p_own[mask].mean()), "lost": int((x.margin[mask] <= 0).sum()),
                   "expected_losses": float((1 - x.p_own[mask]).sum()), "miss": float(x.e_own[mask].mean())}
            for name, mask in (("committee_money", x.committee_money), ("no_money_top_tenth", top), ("no_money_rest", quiet & ~top))}


def forward_check(seats, races, years=(2020, 2022, 2024)):
    """The score and its slope from earlier years, applied to a year's safe seats with no committee money. Few entries come before 2022, so this is rough."""
    x, out = with_results(seats, races), {}
    for y in years:
        train, test = x[x.cycle < y], x[(x.cycle == y) & ~x.committee_money]
        fit = b.fit_entry(train)
        quiet = train[~train.committee_money]
        cell = (quiet.cycle.astype(str) + quiet.status).to_numpy()
        e, s = b.within(quiet.e_own.to_numpy(), cell), b.within(b.flag_score(fit, quiet), cell)
        score = b.flag_score(fit, test)
        shift = (s @ e) / (s @ s) * (score - score.mean())
        model = b.fit_baseline(pd.concat([r for c, r in races.items() if c < y]), list(b.FEATURES))
        sigma_within = np.sqrt(model["sigma"] ** 2 - model["sigma_national"] ** 2)
        mu, miss, won = test.mu_own.to_numpy(), test.margin.to_numpy() - test.mu_own.to_numpy() - test.e_own.to_numpy(), (test.margin.to_numpy() > 0).astype(float)
        loss = lambda m, sd: float(b.log_loss(norm.cdf(m / sd), won).mean())
        error = test.e_own.to_numpy()
        out[y] = {"seats": len(test), "lost": int((1 - won).sum()), "raw": [loss(mu, model["sigma"]), loss(mu + shift, model["sigma"])], "miss_removed": [loss(mu + miss, sigma_within), loss(mu + miss + shift, sigma_within)],
                  "margin_error": [float(np.sqrt((error ** 2).mean())), float(np.sqrt(((error - shift) ** 2).mean()))]}
    return out


def near_line(races, measures):
    """Seats the forecast gave one side 80-95%, 2016-2024, from the favored side's view, with who spent and whether that side won."""
    parts = []
    for side, sign, own, other in (("d", 1, "dccc", "nrcc"), ("r", -1, "nrcc", "dccc")):
        x = with_results(b.safe_seats(races, measures, side=side), races)
        x = x[x.p_own < NEAR]
        parts.append(x.assign(favored=side, won=sign * x.margin > 0, spent=who_spent(x[own] > 0, x[other] > 0)))
    return pd.concat(parts, ignore_index=True)


def near_line_table(near, seed=3):
    """Seats, losses and misses by who spent and side, and how often shuffling wins between seats with and without money gives as large a gap."""
    def row(g):
        return {"seats": len(g), "forecast_win": float(g.p_own.mean()), "lost": int((~g.won).sum()), "expected_losses": float((1 - g.p_own).sum()), "miss": float(g.e_own.mean())}
    money = near.spent != "neither"
    gap = near.won[money].mean() - near.won[~money].mean()
    rng, won, large = np.random.default_rng(seed), near.won.to_numpy(), 0
    for _ in range(SHUFFLES):
        mixed = rng.permutation(won)
        large += abs(mixed[:money.sum()].mean() - mixed[money.sum():].mean()) >= abs(gap)
    return {"by_who_spent": {g: row(near[near.spent == g]) for g in GROUPS}, "by_side": {f"{side}_{'money' if m else 'none'}": row(near[(near.favored == side) & (money == m)]) for side in "dr" for m in (True, False)},
            "money": row(near[money]), "none": row(near[~money]), "win_gap": float(gap), "shuffles_as_large": large / SHUFFLES}


def nearline_record(frozen, both_nominees):
    """One row per 2026 race the frozen baseline gave one side 80-95%, with both nominees. `frozen` is indexed like `both_nominees`."""
    parts = []
    for side, sign, p, own, other in (("D", 1, frozen.p_baseline, "dccc_coordinated", "nrcc_coordinated"), ("R", -1, 1 - frozen.p_baseline, "nrcc_coordinated", "dccc_coordinated")):
        defend, attack = frozen[own].fillna(0.0), frozen[other].fillna(0.0)
        parts.append(pd.DataFrame({"district_id": frozen.district_id, "favored": side, "p_favored": p, "mu_favored": sign * frozen.mu_baseline, "defender_dollars": defend, "attacker_dollars": attack,
                                   "who_spent": who_spent(defend > 0, attack > 0)})[(p >= b.SAFE) & (p < NEAR) & both_nominees])
    return pd.concat(parts, ignore_index=True).round(6)


def run():
    races, _, measures = b.load()
    if races[b.LIVE].margin.notna().any():
        raise SystemExit("2026 results are in the data, so the 2026 record can't be written blind.")
    seats = b.safe_seats(races, measures)
    fit = b.fit_entry(seats)
    results = {"entry_coefficients": dict(zip(fit[2], fit[1].coef_[0].round(3).tolist())), "money_forms": money_forms(races, measures),
               "without_opponents_money": b.miss_test(seats, score_without(seats, "opp_money")), "seat_groups": seat_groups(seats, races), "forward": forward_check(seats, races),
               "near_line": near_line_table(near_line(races, measures))}
    FOLLOWUP.write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))
    if RECORD.exists():
        raise SystemExit(f"{RECORD.name} already exists, and it's the record of what was predicted. Move it aside on purpose if you mean to replace it.")
    now = races[b.LIVE]
    frozen = pd.read_csv(b.FROZEN).set_index("district_id").reindex(now.district_id).reset_index()
    record = nearline_record(frozen, pd.Series(now.cm_d.notna().to_numpy() & now.cm_r.notna().to_numpy()))
    record.to_csv(RECORD, index=False)
    summary = record.assign(lose=1 - record.p_favored).groupby(["favored", "who_spent"]).agg(seats=("lose", "size"), expected_losses=("lose", "sum")).round(2)
    print(summary.to_string())
    print(record[(record.favored == "R") & (record.who_spent == "neither")].district_id.tolist())


if __name__ == "__main__":
    run()
