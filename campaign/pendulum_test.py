"""
Does the national House vote swing like a pendulum? Three pictures of how
the Democratic margin y_t (points) moves from one election to the next,
over 1946-2024 (40 elections):

    Brownian motion (random walk):  y_t = y_{t-1} + c                            no pull back to the center
    Spring (mean reversion, AR(1)): y_t = c + p1 y_{t-1}                         pulled back, no momentum
    Pendulum (oscillator, AR(2)):   y_t = c + p1 y_{t-1} + p2 y_{t-2}            pulled back, and can overshoot

Sampled every two years, a damped oscillator x'' + g x' + w^2 x = noise is an
AR(2). Its roots r e^(+-i theta) give the physics: momentum, a swing that
carries past the center, shows up as complex roots (p1^2 + 4 p2 < 0), with
a period of 2 pi / theta elections and r the share of each swing left after
one election. Real roots mean no overshoot: the spring.

Each picture is fit with and without the known forces: s_t (+1 Democratic
president, -1 Republican) and s_t * midterm_t, the president's party losing
ground in midterms, as in national_test.py. A model of the forces alone,
y_t = c + b s_t + m s_t midterm_t, is the benchmark REPORT.md section 8
already uses. The midterm penalty swings the vote back and forth by itself,
so the pendulum has to show momentum on top of it.

Data: 1976-2024 is national_vote.py (MIT). 1946-1974 is each year's
Wikipedia article infobox (Clerk of the House totals), saved in
data/raw/wikipedia/house_vote/. Over 1976-2024 the two agree within 0.2
points in 20 of 25 years (1992 is the worst, 1.1).

Fixed before any model was scored:

- Every model is scored on the same elections: leaving one out at a time
  (1950-2024, the years with two earlier elections) and forward in time
  (each year from 1980 predicted from 1950 up to the year before), by
  root-mean-square error.
- The pendulum has momentum only if, with the forces in both, (a) it beats
  the spring both ways, (b) its full-sample roots are complex, and (c)
  simulating the fitted spring 10,000 times (normal errors at its residual
  spread, started from the real 1946 and 1948 votes) gives a p2 at least
  as negative as the real one less than 10% of the time.
- There is a pull back to the center at all only if, with the forces in
  both, the spring beats Brownian motion both ways and simulating the
  fitted random walk 10,000 times gives a p1 at least as far below 1 less
  than 10% of the time.
- Sensitivity: the same tests with the center allowed to move after the
  1994 realignment (a constant for 1994 on). Forward in time it is
  estimable only from 1996.
- Not tests, readings: the residual spread (sigma, the size of the random
  kicks) before and after 1994; what each model would have said for 2026
  from 2024 and earlier; and what each says for 2028 if 2026 lands at the
  final generic ballot (data/generic_ballot.csv).

Added after the first run, so readings too: sigma split again at 2012
(1950-1992, 1994-2010, 2012-2024, eras picked after seeing the series);
each model's forward error over 2012-2024, the years national_test.py
scores; and the 2026 and 2028 predictions with the center moved in 1994.

The blend (blend_test), fixed before it was scored. Over 2012-2024 the
spring with forces missed the House vote by 2.37 points forward, against
3.04 for the Sept 1 polls, a lead found after looking. Does the spring add
to the polls?

    polls:  y_t = poll_t + a                                (the polls plus their average lean)
    blend:  y_t = poll_t + a + w (spring_t - poll_t)        w = 0 is the polls, w = 1 the spring

- spring_t is the spring with forces fit on 1950 up to the year before,
  the forward prediction above, so it never sees year t.
- Over national_test.py's elections (2002-2024) and its scoring: leaving
  one out, and forward in time (each year from 2012 fit on 2002 up to the
  year before). The polls alone reproduce REPORT.md section 8 (Sept 1:
  3.40 and 3.04).
- The blend passes if it beats the polls both ways and shuffling spring_t
  across the 12 elections 20,000 times gives a leave-one-out error at least
  as low less than 10% of the time.
- Primary: the Sept 1 polls. Secondary: the final polls, which already
  have most of the fall. Sensitivity: the spring with the center moved in
  1994. Readings: w, and an unconstrained y = a + b poll + c spring.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .national_test import national_table
from .national_vote import national_vote
from .plots.plot_spring import plot_spring

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "data" / "raw" / "wikipedia" / "house_vote"
RESULTS = ROOT / "data" / "pendulum_results.json"
FIGURE = ROOT / "figures" / "16_spring.png"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "Campaign-Allocation/1.0 (https://github.com/callum-doty/Campaign-Allocation) "
                         f"python-requests/{requests.__version__}"}

FIRST, MIT_FROM, LAST = 1946, 1976, 2024
SCORED_FROM, FORWARD_FROM, REALIGNED = 1950, 1980, 1994
PRESIDENT = {1946: "D", 1948: "D", 1950: "D", 1952: "D", 1954: "R", 1956: "R", 1958: "R", 1960: "R",
             1962: "D", 1964: "D", 1966: "D", 1968: "D", 1970: "R", 1972: "R", 1974: "R", 1976: "R",
             1978: "D", 1980: "D", 1982: "R", 1984: "R", 1986: "R", 1988: "R", 1990: "R", 1992: "R",
             1994: "D", 1996: "D", 1998: "D", 2000: "D", 2002: "R", 2004: "R", 2006: "R", 2008: "R",
             2010: "D", 2012: "D", 2014: "D", 2016: "D", 2018: "R", 2020: "R", 2022: "D", 2024: "D",
             2026: "R", 2028: "R"}
LAGS = {"brownian": None, "spring": 1, "pendulum": 2}
PASS_P = 0.10
N_SIM = 10_000
SIGMA_ERAS = ((1950, 1992), (1994, 2010), (2012, 2024))   # before the realignment, the wave years, national_test.py's years
BLEND_CYCLES, BLEND_FORWARD_FROM = tuple(range(2002, 2025, 2)), 2012       # national_test.py's years and scoring
N_PERM = 20_000


def wikipedia_margin(year):
    """Democratic minus Republican House vote, as a share of all House
    votes, from the year's Wikipedia infobox (fetched once and saved)."""
    page = PAGES / f"{year}.wikitext"
    if not page.exists():
        PAGES.mkdir(parents=True, exist_ok=True)
        resp = requests.get(API, params={"action": "parse", "page": f"{year} United States House of Representatives elections",
                                         "prop": "wikitext", "format": "json", "formatversion": 2, "redirects": 1},
                            headers=HEADERS, timeout=60)
        resp.raise_for_status()
        page.write_text(resp.json()["parse"]["wikitext"])
        time.sleep(0.5)
    text = page.read_text()
    number = lambda field, i: float(re.sub(r"[^0-9.]", "", re.sub(r"<ref.*", "", re.search(
        rf"\|\s*{field}{i}\s*=\s*([^\n]*)", text).group(1))))
    party = {}
    for i in "1234":
        name = re.search(rf"\|\s*party{i}\s*=\s*([^\n|]*)", text)
        key = name and ("D" if "Democratic" in name.group(1) else "R" if "Republican" in name.group(1) else None)
        if key and key not in party:
            party[key] = (number("popular_vote", i), number("percentage", i))
    (d_votes, d_pct), (r_votes, _) = party["D"], party["R"]
    return 100 * (d_votes - r_votes) / (d_votes / (d_pct / 100))


def house_vote_history():
    """One row per election, 1946-2024: house_vote (Democratic margin),
    source, s (+1 Democratic president, -1 Republican), midterm (1 or 0)."""
    early = pd.Series({year: wikipedia_margin(year) for year in range(FIRST, MIT_FROM, 2)})
    table = pd.DataFrame({"house_vote": pd.concat([early, national_vote(tuple(range(MIT_FROM, LAST + 1, 2)))])})
    table["source"] = np.where(table.index < MIT_FROM, "Wikipedia", "MIT")
    table["s"] = table.index.map(lambda year: 1.0 if PRESIDENT[year] == "D" else -1.0)
    table["midterm"] = (table.index % 4 == 2).astype(float)
    return table


def design(y, s, midterm, years, lags, forces, realigned):
    """Rows for every year from SCORED_FROM: (target, columns). Brownian
    motion's target is the change from the last election, its columns have
    no lag; the others predict the level from their lags."""
    rows = np.flatnonzero(years >= SCORED_FROM)
    columns = [np.ones(len(rows))]
    if lags:
        columns += [y[rows - k] for k in range(1, lags + 1)]
    if forces:
        columns += [s[rows], s[rows] * midterm[rows]]
    if realigned:
        columns.append((years[rows] >= REALIGNED).astype(float))
    target = y[rows] - (y[rows - 1] if lags is None else 0)
    return target, np.column_stack(columns), rows


def fit(X, target, keep):
    return np.linalg.lstsq(X[keep], target[keep], rcond=None)[0]


def scores(y, s, midterm, years, lags, forces, realigned):
    """Root-mean-square error left-one-out ('loo') and forward in time
    ('forward'), plus the full-sample coefficients and residual spread."""
    target, X, rows = design(y, s, midterm, years, lags, forces, realigned)
    n, scored = len(target), years[rows]
    loo = [target[i] - X[i] @ fit(X, target, np.arange(n) != i) for i in range(n)]
    forward = {int(year): target[i] - X[i] @ fit(X, target, scored < year) for i, year in enumerate(scored)
               if year >= FORWARD_FROM}
    coef = fit(X, target, np.ones(n, bool))
    resid = target - X @ coef
    return {"loo": float(np.sqrt(np.mean(np.square(loo)))), "forward": float(np.sqrt(np.mean(np.square(list(forward.values()))))),
            "coef": coef.tolist(), "sigma": float(np.sqrt(resid @ resid / (n - X.shape[1]))),
            "resid": dict(zip(map(int, scored), resid.tolist())), "forward_errors": forward}


def roots(p1, p2):
    """The pendulum's physics from its AR(2) coefficients."""
    disc = p1 ** 2 + 4 * p2
    if disc < 0:
        r, theta = np.sqrt(-p2), np.arccos(p1 / (2 * np.sqrt(-p2)))
        return {"complex": True, "r": float(r), "period_years": float(2 * 2 * np.pi / theta),
                "half_life_years": float(2 * np.log(0.5) / np.log(r))}
    z = sorted([(p1 + np.sqrt(disc)) / 2, (p1 - np.sqrt(disc)) / 2], key=abs, reverse=True)
    return {"complex": False, "roots": [float(v) for v in z]}


def simulate(y, s, midterm, years, lags, coef, sigma, forces, realigned, rng):
    """One series from a fitted Brownian motion (lags None) or spring
    (lags 1), started from the real 1946 and 1948 votes."""
    sim = y.copy()
    for i in np.flatnonzero(years >= SCORED_FROM):
        x = [1.0] + ([sim[i - 1]] if lags else []) + ([s[i], s[i] * midterm[i]] if forces else []) + \
            ([float(years[i] >= REALIGNED)] if realigned else [])
        sim[i] = (0 if lags else sim[i - 1]) + np.dot(coef, x) + rng.normal(0, sigma)
    return sim


def pendulum_test(table, realigned=False, seed=0):
    """Every model's scores; the momentum and pull-back tests (each with
    its simulated p-value and verdict); the pendulum's roots."""
    y, s, midterm = (table[c].to_numpy(float) for c in ("house_vote", "s", "midterm"))
    years = table.index.to_numpy()
    out = {"models": {}}
    for forces in (False, True):
        for name, lags in LAGS.items():
            out["models"][name + ("+forces" if forces else "")] = scores(y, s, midterm, years, lags, forces, realigned)
    out["models"]["forces only"] = scores(y, s, midterm, years, 0, True, realigned)
    m = out["models"]
    rng = np.random.default_rng(seed)

    spring, pendulum = m["spring+forces"], m["pendulum+forces"]
    p2 = pendulum["coef"][2]
    sims = []
    for _ in range(N_SIM):
        sim = simulate(y, s, midterm, years, 1, spring["coef"], spring["sigma"], True, realigned, rng)
        target, X, _ = design(sim, s, midterm, years, 2, True, realigned)
        sims.append(fit(X, target, np.ones(len(target), bool))[2])
    out["pendulum_roots"] = roots(*pendulum["coef"][1:3])
    out["momentum"] = {"p2": p2, "p": float(np.mean(np.array(sims) <= p2)),
                       "beats_spring": pendulum["loo"] < spring["loo"] and pendulum["forward"] < spring["forward"]}
    out["momentum"]["passes"] = bool(out["momentum"]["beats_spring"] and out["pendulum_roots"]["complex"]
                                     and out["momentum"]["p"] < PASS_P)

    walk = m["brownian+forces"]
    p1 = spring["coef"][1]
    sims = []
    for _ in range(N_SIM):
        sim = simulate(y, s, midterm, years, None, walk["coef"], walk["sigma"], True, realigned, rng)
        target, X, _ = design(sim, s, midterm, years, 1, True, realigned)
        sims.append(fit(X, target, np.ones(len(target), bool))[1])
    out["pull_back"] = {"p1": p1, "p": float(np.mean(np.array(sims) <= p1)),
                        "beats_brownian": spring["loo"] < walk["loo"] and spring["forward"] < walk["forward"]}
    out["pull_back"]["passes"] = bool(out["pull_back"]["beats_brownian"] and out["pull_back"]["p"] < PASS_P)
    out["pendulum_roots_no_forces"] = roots(*m["pendulum"]["coef"][1:3])
    return out


def predict(table, model, year, extra=None):
    """A model's full-sample prediction for `year` (2026 or 2028), from
    the real votes plus any assumed ones in `extra` {year: margin}."""
    y = table.house_vote.to_dict() | (extra or {})
    s = 1.0 if PRESIDENT[year] == "D" else -1.0
    x = [1.0]
    if model["lags"]:
        x += [y[year - 2 * k] for k in range(1, model["lags"] + 1)]
    if model["forces"]:
        x += [s, s * float(year % 4 == 2)]
    if model["realigned"]:
        x.append(1.0)
    value = float(np.dot(model["coef"], x))
    return value + (y[year - 2] if model["lags"] is None else 0)


def readings(table, result, realigned, gb_2026):
    """sigma, the root-mean-square residual of the spring with forces (with
    and without the moved center) in each of SIGMA_ERAS; each model's
    forward error over 2012-2024 (national_test.py's forward years); and
    each model's 2026 and 2028 predictions (2028 assumes 2026 = gb_2026)."""
    sigma = {}
    for run, res in (("fixed center", result), ("center moved in 1994", realigned)):
        resid = pd.Series(res["models"]["spring+forces"]["resid"])
        sigma[run] = {f"{a}-{b}": float(np.sqrt(np.mean(resid[(resid.index >= a) & (resid.index <= b)] ** 2)))
                      for a, b in SIGMA_ERAS}
    recent = {name: float(np.sqrt(np.mean([e ** 2 for year, e in m["forward_errors"].items() if year >= 2012])))
              for name, m in result["models"].items()}
    preds = {}
    for run, res in (("fixed center", result), ("center moved in 1994", realigned)):
        preds[run] = {}
        for name, m in res["models"].items():
            spec = {"coef": m["coef"], "forces": name.endswith("forces") or name == "forces only",
                    "lags": 0 if name == "forces only" else LAGS[name.replace("+forces", "")],
                    "realigned": run != "fixed center"}
            preds[run][name] = {"2026": predict(table, spec, 2026),
                                "2028 if 2026 = gb": predict(table, spec, 2028, {2026: gb_2026})}
    return {"sigma": sigma, "forward_2012_2024": recent, "predictions": preds, "gb_2026": gb_2026}


def blend_test(result, polls, poll, n_perm=N_PERM, seed=0):
    """Root-mean-square error of the polls plus their lean ('polls'), the
    blend ('blend', with its full-sample a and w, the shuffle p-value and
    whether it passes) and an unconstrained regression ('free'), left one
    out ('loo') and forward in time ('forward'). `poll` names the column
    of `polls` (national_table) to blend with."""
    forward = result["models"]["spring+forces"]["forward_errors"]
    cycles = np.array(BLEND_CYCLES)
    y = polls.house_vote.reindex(cycles).to_numpy(float)
    p = polls[poll].reindex(cycles).to_numpy(float)
    spring = np.array([polls.house_vote[c] - forward[c] for c in cycles])

    def scores(target, X):
        loo = [target[i] - X[i] @ fit(X, target, np.arange(len(cycles)) != i) for i in range(len(cycles))]
        fwd = [target[i] - X[i] @ fit(X, target, cycles < c) for i, c in enumerate(cycles) if c >= BLEND_FORWARD_FROM]
        return {"loo": float(np.sqrt(np.mean(np.square(loo)))), "forward": float(np.sqrt(np.mean(np.square(fwd))))}

    ones = np.ones(len(cycles))
    blend_X = lambda sp: np.column_stack([ones, sp - p])
    out = {"polls": scores(y - p, ones[:, None]), "blend": scores(y - p, blend_X(spring)),
           "free": scores(y, np.column_stack([ones, p, spring])), "spring": dict(zip(map(int, cycles), spring.round(2).tolist()))}
    out["blend"]["a"], out["blend"]["w"] = map(float, fit(blend_X(spring), y - p, ones.astype(bool)))
    out["free"]["coef"] = fit(np.column_stack([ones, p, spring]), y, ones.astype(bool)).tolist()
    rng = np.random.default_rng(seed)
    shuffled = np.array([scores(y - p, blend_X(rng.permutation(spring)))["loo"] for _ in range(n_perm)])
    out["blend"]["p"] = float(np.mean(shuffled <= out["blend"]["loo"]))
    out["blend"]["passes"] = bool(out["blend"]["loo"] < out["polls"]["loo"] and
                                  out["blend"]["forward"] < out["polls"]["forward"] and out["blend"]["p"] < PASS_P)
    return out


def main():
    table = house_vote_history()
    gb_2026 = float(pd.read_csv(ROOT / "data" / "generic_ballot.csv").set_index("cycle").generic_ballot[2026])
    out = {"main": pendulum_test(table), "realigned": pendulum_test(table, realigned=True)}
    out["readings"] = readings(table, out["main"], out["realigned"], gb_2026)
    polls = national_table(BLEND_CYCLES)
    out["blend"] = {f"{poll}, {run}": blend_test(out[run], polls, poll)
                    for poll in ("gb_sept", "gb") for run in ("main", "realigned")}
    out["series"] = table.house_vote.round(2).to_dict()
    for run in ("main", "realigned"):
        for m in out[run]["models"].values():
            m.pop("resid")
            m["forward_errors"] = {year: round(e, 2) for year, e in m["forward_errors"].items()}
    out["president"] = PRESIDENT
    RESULTS.write_text(json.dumps(out, indent=1, default=str))
    plot_spring(json.loads(RESULTS.read_text()), FIGURE)
    print(f"wrote {RESULTS.relative_to(ROOT)} and {FIGURE.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    main()
