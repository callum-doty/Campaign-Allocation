"""
How the generic ballot moves between a date in the campaign and the final
pre-election average, learned from earlier elections:

    drift_t = gb_final_t - gb_(date, t)

with the final averages in data/generic_ballot.csv and the dated ones in
data/generic_ballot_dated.csv. A forecast made on that date doesn't know
where the polls will end up. It takes gb_date + mean(drift) as its best guess
of the final average, and adds the drift's spread to the national error,
since every race moves with it. The race-level model is still fit on the
final averages, the best measure of each past cycle's national environment;
only the input changes.
"""

import pandas as pd

from .build_races import GENERIC_BALLOT, GENERIC_BALLOT_DATED


def fall_drift(cycle, date="09-01"):
    """{'mean', 'sd', 'n'}: the drift from `date` ("MM-DD") to the final
    average (points, D-positive), over every election before `cycle` with
    both."""
    final = pd.read_csv(GENERIC_BALLOT).set_index("cycle").generic_ballot
    dated = pd.read_csv(GENERIC_BALLOT_DATED, dtype={"date": str})
    on_date = dated[dated.date == date].set_index("cycle").generic_ballot
    drift = (final - on_date)[lambda d: d.index < cycle].dropna()
    return {"mean": float(drift.mean()), "sd": float(drift.std()), "n": len(drift)}
