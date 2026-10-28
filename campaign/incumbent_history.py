"""
Two features from each district's recent House history.

- fresh_d / fresh_r: the Democratic (Republican) nominee is a first-term
  incumbent: an incumbent whose surname isn't among the state's House
  winners four years earlier. Matching within the state, not the district,
  keeps veterans on redrawn lines from counting as new. First-term members,
  many of them elected in a wave, run well behind the incumbency effect
  pooled over all members.
- over_inc / over_open: how far the party that won the seat two years ago
  ran ahead of the district's lean then,

      over = margin_(t-2) - GB_(t-2) - 1.6 * PVI_(t-2)

  (1.6 is about the PVI slope the baseline fits in every cycle), carried
  only when the state's lines haven't changed since. over_inc is set when that
  party's incumbent is running again, over_open when the seat is open; each
  is zero otherwise, as is everything after a redraw or an uncontested race.
  A member who won by half a point in a wave year carries that forward.
"""

import unicodedata

import numpy as np
import pandas as pd

from .compare_maps import compare_maps
from .load_candidates import SUFFIXES, WORD
from .load_presidential import load_presidential
from .load_pvi import LINES, load_pvi
from .load_results import load_results

OVER_PVI_SLOPE = 1.6
UNCONTESTED = 80


def incumbent_history(cycle, races, prior_gb):
    """fresh_d, fresh_r, over_inc, over_open for each row of `races`, which
    needs district_id, d_incumbent, open_seat, status and the nominees' FEC
    names (name_d, name_r, "SURNAME, FIRST"). `prior_gb`: the generic ballot
    two years earlier."""
    def words(name):  # ascii, upper-case words of a name, suffixes dropped
        name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii").upper()
        return [w for w in WORD.findall(name) if w not in SUFFIXES]

    earlier = load_results(cycle - 4)
    won = earlier.d_names.where(earlier.margin > 0, earlier.r_names)
    winners = {(d[:2], words(n)[-1]) for d, names in zip(earlier.district_id, won)
               if isinstance(names, list) for n in names if words(n)}
    state = races.district_id.str[:2]
    out = pd.DataFrame(index=races.index)
    for side, incumbent in (("d", races.d_incumbent == 1), ("r", races.status == "R incumbent")):
        surnames = races[f"name_{side}"].fillna("").map(lambda n: words(n.split(",")[0]))
        returning = [any((s, w) in winners for w in ws) for s, ws in zip(state, surnames)]
        out[f"fresh_{side}"] = (incumbent & ~np.array(returning)).astype(float)

    out["over_inc"], out["over_open"] = 0.0, 0.0
    if cycle - 2 not in LINES:
        return out
    old, new = LINES[cycle - 2], LINES[cycle]
    common = max(set(load_presidential(old).pres_year) & set(load_presidential(new).pres_year))
    # every district in a redrawn state counts as new, even one whose share happens to match
    maps = compare_maps(old_lines=old, new_lines=new, year=common)
    redrawn = maps.changed.groupby(maps.district_id.str[:2]).transform("any")
    unchanged = races.district_id.map(pd.Series(~redrawn.to_numpy(), index=maps.district_id)).eq(True)
    margin = races.district_id.map(load_results(cycle - 2).set_index("district_id").margin)
    over = margin - prior_gb - OVER_PVI_SLOPE * races.district_id.map(load_pvi(cycle - 2))
    over = over.where(unchanged & (margin.abs() < UNCONTESTED)).fillna(0.0)
    held_d = margin > 0
    rerun = ((races.d_incumbent == 1) & held_d) | ((races.status == "R incumbent") & ~held_d)
    out["over_inc"] = over.where(rerun, 0.0)
    out["over_open"] = over.where(races.open_seat == 1, 0.0)
    return out
