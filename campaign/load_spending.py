"""
Independent expenditures in House general elections, totalled by district:
DCCC's and NRCC's own money, and outside money on each side. Cleaning
(amended reports, missing party codes, fraudulent filings) is in
load_expenditures.py.

Coordinated party spending is not in the FEC's independent-expenditure
file. It was small through 2024 (DCCC: $5.4M in 2024), but after NRSC v.
FEC (June 2026) struck down the limits it is how DCCC spends in 2026, so
2026 committee money is missing here.

Given a decision date, outside money is also totalled over what had reached
voters by then (outside_aug_d, outside_aug_r for an Aug 31 date): all a
forecaster deciding on that date could see. Only 2-8% of the outside money
in 2012-2024 had.
"""

from .load_expenditures import load_expenditures

BUCKETS = ["outside_d", "outside_r", "dccc", "nrcc"]


def load_spending(cycle, decision=None):
    """district_id, outside_d, outside_r, dccc, nrcc (dollars), and with a
    `decision` date, outside_aug_d, outside_aug_r."""
    ie = load_expenditures(cycle)
    out = ie.pivot_table(index="district_id", columns="bucket", values="amount", aggfunc="sum", fill_value=0)
    out = out.reindex(columns=BUCKETS, fill_value=0)
    if decision is not None:
        seen = ie[ie.date <= decision].pivot_table(index="district_id", columns="bucket", values="amount",
                                                   aggfunc="sum", fill_value=0)
        seen = seen.reindex(columns=["outside_d", "outside_r"], fill_value=0)
        out = out.join(seen.rename(columns=lambda c: c.replace("outside", "outside_aug"))).fillna(0)
    return out.reset_index()
