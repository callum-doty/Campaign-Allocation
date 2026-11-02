"""
The 2026 forecast under each money specification still in question, written
down before any results exist, so the election is a clean test of them. The
2016-2024 backtest can't settle these: the biggest gains and losses in log
loss come in 2020 and 2022, mostly from which way the polls missed.

- baseline: floor_share, the forecast in REPORT.md.
- symmetric: money_odds = log((F_D + k) / (F_R + k)) replaces floor_share,
  with the same F and k. floor_share is the log of the Democratic share, so
  it barely moves with Republican money once Democrats lead.
- surplus: the baseline plus surplus = log((S_D + k) / (S_R + k)), S = raised
  minus spent through June 30, at least 0. It isn't cash on hand: it leaves
  out money carried over from earlier cycles.
- symmetric_surplus: both changes.

Each is fit on 2012-2024 and uses the generic ballot in data/generic_ballot.csv.
freeze() won't overwrite an earlier freeze. The CSV keeps every feature and the
JSON every coefficient, so each forecast can be recomputed from the two files.

score() fixes now how they'll be judged, once results are in: log loss and
Brier over all races, log loss in the races the baseline put within CLOSE
points (the same races for every spec), and margin error in contested races.
Each is also scored with the year's average miss taken out, which is the
reading to trust: one national miss rewards whichever spec leaned that way.

    python -m campaign.freeze_forecast
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .build_races import FLOOR_PSEUDO_DOLLARS, GENERIC_BALLOT, build_races
from .fit_baseline import FEATURES, UNCONTESTED, fit_baseline
from .fit_close import CLOSE
from .predict_margin import predict_margin
from .seat_distribution import seat_distribution
from .seats_needed import seats_needed

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "forecast_freeze_2026"
SYMMETRIC = [f if f != "floor_share" else "money_odds" for f in FEATURES]
SPECS = {"baseline": FEATURES, "symmetric": SYMMETRIC, "surplus": FEATURES + ["surplus"],
         "symmetric_surplus": SYMMETRIC + ["surplus"]}
INPUTS = ["district_id", "status", "tier", "gb", *FEATURES, "money_odds", "surplus", "raised_jun_d", "raised_jun_r",
          "spent_jun_d", "spent_jun_r", "own_jun_d", "own_jun_r", "outside_aug_d", "outside_aug_r"]


def money_terms(races):
    """`races` with money_odds and surplus added."""
    k = FLOOR_PSEUDO_DOLLARS
    floor = {s: (races[f"raised_jun_{s}"] - races[f"own_jun_{s}"]).clip(lower=0) + races[f"outside_aug_{s}"] for s in "dr"}
    left = {s: (races[f"raised_jun_{s}"] - races[f"spent_jun_{s}"]).clip(lower=0) for s in "dr"}
    return races.assign(money_odds=np.log((floor["d"] + k) / (floor["r"] + k)),
                        surplus=np.log((left["d"] + k) / (left["r"] + k)))


def freeze(races, cycle=2026, stem=FROZEN):
    """Write `stem`.csv (one row per race) and `stem`.json (fits and seat totals). `races`: cycle -> build_races(cycle)."""
    csv, summary = stem.with_suffix(".csv"), stem.with_suffix(".json")
    if csv.exists() or summary.exists():
        raise SystemExit(f"{csv.name} is already frozen. Move it aside on purpose if you mean to replace it.")
    races = {c: money_terms(r) for c, r in races.items()}
    train = pd.concat([r for c, r in races.items() if c < cycle])
    fc = races[cycle]
    out = fc[INPUTS].copy()
    need, states = seats_needed(cycle, fc.district_id), fc.district_id.str[:2]
    specs = {}
    for name, features in SPECS.items():
        model = fit_baseline(train, features)
        out[f"mu_{name}"] = predict_margin(fc, model)
        out[f"p_{name}"] = norm.cdf(out[f"mu_{name}"] / model["sigma"])
        seats = seat_distribution(out[f"mu_{name}"], model, need, states)
        specs[name] = {"features": features, **model,
                       "seats": {f: seats[f] for f in ("mean", "p10", "p50", "p90", "need", "p_majority")}}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ballot = pd.read_csv(GENERIC_BALLOT).set_index("cycle").loc[cycle].to_dict()
    out.round(6).to_csv(csv, index=False)
    summary.write_text(json.dumps({"cycle": cycle, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "commit": commit, "generic_ballot": ballot, "specs": specs}, indent=1) + "\n")
    return out, specs


def score(frozen, margin, specs):
    """One row per spec. `frozen`: the CSV; `margin`: Democratic margin by district_id, races without one
    left out; `specs`: the JSON's fits, for each spec's sigma and sigma_national."""
    known = frozen[frozen.district_id.map(margin).notna()]
    actual = known.district_id.map(margin).to_numpy(float)
    won, contested = (actual > 0).astype(float), np.abs(actual) < UNCONTESTED
    near = known.mu_baseline.abs().to_numpy() < CLOSE

    def log_loss(p, rows):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(won[rows] * np.log(p[rows]) + (1 - won[rows]) * np.log(1 - p[rows])))

    rows = {}
    for name, fit in specs.items():
        mu = known[f"mu_{name}"].to_numpy(float)
        miss = float((actual - mu)[contested].mean())
        p = norm.cdf(mu / fit["sigma"])
        p_within = norm.cdf((mu + miss) / np.sqrt(fit["sigma"] ** 2 - fit["sigma_national"] ** 2))
        everything = np.ones(len(known), bool)
        rows[name] = {"log_loss": log_loss(p, everything), "brier": float(np.mean((p - won) ** 2)),
                      "log_loss_close": log_loss(p, near),
                      "margin_error": float(np.sqrt(np.mean((actual - mu)[contested] ** 2))), "national_miss": miss,
                      "log_loss_within": log_loss(p_within, everything), "log_loss_close_within": log_loss(p_within, near),
                      "margin_error_within": float(np.sqrt(np.mean((actual - mu - miss)[contested] ** 2)))}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    _, fits = freeze({c: build_races(c) for c in range(2012, 2027, 2)})
    for spec, fit in fits.items():
        s = fit["seats"]
        print(f"{spec}: {s['mean']:.1f} expected D seats, 80% range {s['p10']}-{s['p90']}, "
              f"{s['p_majority']:.0%} chance of a majority")
