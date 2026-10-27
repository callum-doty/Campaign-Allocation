"""
How many of the modeled races Democrats need for a House majority (218).

Districts left out of the model are uncontested or nearly so (under $100K
of spending) plus Alaska; each is credited to the party its PVI favors. A
district with no PVI (27 of the 2016 districts redrawn mid-decade in FL, NC
and VA have no earlier presidential result on their new lines) is credited
to whichever party won it two years before, so nothing from the election
being forecast is used.
"""

from .load_presidential import load_presidential
from .load_pvi import LINES, load_pvi
from .load_results import load_results

MAJORITY = 218


def seats_needed(cycle, modeled_districts):
    districts = set(load_presidential(LINES[cycle]).district_id) - set(modeled_districts)
    lean = load_pvi(cycle).reindex(sorted(districts))
    missing = lean.index[lean.isna()]
    if len(missing):
        prior = load_results(cycle - 2).set_index("district_id").margin
        lean[missing] = prior.reindex(missing)
    return MAJORITY - int((lean > 0).sum())
