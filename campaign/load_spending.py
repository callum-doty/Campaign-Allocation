"""
Independent expenditures in House general elections, totalled by district:
DCCC's and NRCC's own money, and outside money on each side. Cleaning
(amended reports, missing party codes, fraudulent filings) is in
load_expenditures.py.

Coordinated party spending is not in the FEC's independent-expenditure
file. It was small through 2024 (DCCC: $5.4M in 2024), but after NRSC v.
FEC (June 2026) struck down the limits it is how DCCC spends in 2026, so
2026 committee money is missing here.
"""

from .load_expenditures import load_expenditures


def load_spending(cycle):
    """district_id, outside_d, outside_r, dccc, nrcc (dollars)."""
    ie = load_expenditures(cycle)
    out = ie.pivot_table(index="district_id", columns="bucket", values="amount", aggfunc="sum", fill_value=0)
    return out.reindex(columns=["outside_d", "outside_r", "dccc", "nrcc"], fill_value=0).reset_index()
