"""
What a Sept 1 forecaster could read from each campaign's operating expenditures
through June 30, per race, for spending_signal.py to test. Each compares the two
sides and is positive when it favors the Democrat, as money terms are.

- persuasion_odds: money raised less what it cost to raise it, as odds. If direct
  mail and online list churn inflate receipts, this should beat plain receipts.
- contact_odds: early money on TV, digital, mail, field and production, as odds.
- fund_share, admin_share: the difference in the share of spending on fundraising,
  and on payroll and overhead.
- inc_poll: an incumbent's early polling and research, in log dollars, signed by party.
  Safe incumbents have little reason to poll early, so it may reveal private worry.
- pro_vendor: the difference in the share of spending at vendors with at least five
  clients in races within 10 points in earlier cycles. Only earlier cycles count.
- accel: the difference in the share of spending that came in April-June.
- fund_odds, contact_odds, admin_odds together: the direct test of whether splitting
  spending by category says more than the June 30 surplus.

A second set, tried after those, each again the Democrat's value minus the Republican's:

- burn: spent as a share of raised. runway: log months of cash at April-June's pace of
  spending. fund_cost: fundraising cost as a share of raised.
- digital_mail, polling_digital, broadcast_field: both channels took at least 5% of
  spending, so a pair works together if at all.
- poll_first: polled before the first voter contact. contact_lead: how long before
  June 30 voter contact began, in years.
- vendor_hhi: how concentrated spending is among vendors. vendor_retention: the share of
  spending at vendors the same committee used last cycle.
- debt_burden: debts owed on the July report as a share of raised. Candidates' loans to their
  own campaigns count as debts.
"""

import numpy as np
import pandas as pd

from .load_candidates import load_candidates
from .load_early_money import load_debts, load_early_money
from .load_results import load_results

FUND = ["Fundraising", "Processing fees"]
CONTACT = ["Broadcast", "Digital", "Mail/print", "Field/phones", "Production"]
ODDS_DOLLARS = 10_000
PRO_VENDOR_CLIENTS = 5
SINGLE = ["persuasion_odds", "contact_odds", "fund_share", "admin_share", "inc_poll", "pro_vendor", "accel"]
SECOND = ["burn", "runway", "fund_cost", "digital_mail", "polling_digital", "broadcast_field", "poll_first", "contact_lead",
          "vendor_hhi", "vendor_retention", "debt_burden"]
PAIR_SHARE = 0.05
PAIRS = {"digital_mail": ("Digital", "Mail/print"), "polling_digital": ("Polling/research", "Digital"), "broadcast_field": ("Broadcast", "Field/phones")}
SPLIT = ["fund_odds", "contact_odds", "admin_odds"]


def committees(cycle, race_ids):
    """DataFrame by race, cm_d and cm_r: each nominee's committee, from the July report's choice."""
    candidates, early = load_candidates(cycle, load_results(cycle)), load_early_money(cycle)
    out = {}
    for side, party in (("d", "D"), ("r", "R")):
        ids = candidates[candidates.party == party].drop_duplicates("district_id").set_index("district_id").cand_id
        out[f"cm_{side}"] = race_ids.map(ids).map(early.committee_id)
    return pd.DataFrame(out)


def odds(a, b):
    return np.log((a + ODDS_DOLLARS) / (b + ODDS_DOLLARS))


def difference_of_shares(x, top, bottom):
    def share(side):
        return (x[f"{top}_{side}"] / x[f"{bottom}_{side}"].where(x[f"{bottom}_{side}"] > 0)).fillna(0.0)
    return share("d") - share("r")


def committee_measures(cycle, rows, last_cycle_vendors):
    """By committee: the second set's measures that don't need the July report: pairs, poll_first, contact_lead, vendor_hhi, vendor_retention."""
    paid = rows[rows.amount > 0]
    totals = paid.groupby("committee_id").amount.sum()
    share = paid.pivot_table(index="committee_id", columns="category", values="amount", aggfunc="sum", fill_value=0.0).div(totals, axis=0)
    out = pd.DataFrame(index=totals.index)
    for name, (a, b) in PAIRS.items():
        out[name] = ((share.get(a, 0.0) >= PAIR_SHARE) & (share.get(b, 0.0) >= PAIR_SHARE)).astype(float)
    contact = paid[paid.category.isin(CONTACT)].groupby("committee_id").date.min()
    polling = paid[paid.category == "Polling/research"].groupby("committee_id").date.min()
    out["poll_first"] = (polling.reindex(out.index) < contact.reindex(out.index)).astype(float)
    out["contact_lead"] = ((pd.Timestamp(f"{cycle}-06-30") - contact.reindex(out.index)).dt.days / 365).fillna(0.0).clip(lower=0)
    named = paid[paid.vendor != ""]
    by_vendor = named.groupby(["committee_id", "vendor"]).amount.sum()
    out["vendor_hhi"] = (by_vendor / by_vendor.groupby(level=0).transform("sum")).pow(2).groupby(level=0).sum()
    if last_cycle_vendors is not None:
        kept = by_vendor[[pair in last_cycle_vendors for pair in by_vendor.index]]
        out["vendor_retention"] = (kept.groupby(level=0).sum() / by_vendor.groupby(level=0).sum()).reindex(out.index).fillna(0.0)
    else:
        out["vendor_retention"] = 0.0
    return out.fillna(0.0), set(by_vendor.index)


def add_spending(races, pre):
    """`races` (cycle -> money_terms(build_races(cycle))) with the features added. `pre`: cycle -> categorized rows through June 30."""
    out, close_clients, last_vendors = {}, {}, None
    for cycle in sorted(races):
        rows, x = pre[cycle], races[cycle].copy()
        x = x.join(committees(cycle, x.district_id))
        by_category = rows.pivot_table(index="committee_id", columns="category", values="amount", aggfunc="sum", fill_value=0.0)
        late = rows[rows.date >= pd.Timestamp(f"{cycle}-04-01")].groupby("committee_id").amount.sum()
        for side in "dr":
            cm = x[f"cm_{side}"]

            def total(names):
                return cm.map(by_category.reindex(columns=names, fill_value=0.0).sum(axis=1)).fillna(0.0).clip(lower=0)
            x[f"fund_{side}"], x[f"contact_{side}"] = total(FUND), total(CONTACT)
            x[f"poll_{side}"], x[f"admin_{side}"] = total(["Polling/research"]), total(["Payroll/admin"])
            x[f"spend_{side}"] = cm.map(rows.groupby("committee_id").amount.sum()).fillna(0.0).clip(lower=0)
            x[f"late_{side}"] = cm.map(late).fillna(0.0).clip(lower=0)
        close = set(x[x.margin.abs() < 10][["cm_d", "cm_r"]].stack())
        paid = rows[rows.amount > 0]
        close_clients[cycle] = paid[paid.committee_id.isin(close)].groupby("vendor").committee_id.nunique()
        earlier = [close_clients[c] for c in close_clients if c < cycle]
        record = pd.concat(earlier).groupby(level=0).sum() if earlier else pd.Series(dtype=float)
        pros = set(record[record >= PRO_VENDOR_CLIENTS].index)
        pro_share = paid[paid.vendor.isin(pros)].groupby("committee_id").amount.sum() / paid.groupby("committee_id").amount.sum()
        pro = {side: x[f"cm_{side}"].map(pro_share).fillna(0.0) for side in "dr"}
        measures, last_vendors = committee_measures(cycle, rows, last_vendors)
        early = load_early_money(cycle)
        cash_by_committee = early.drop_duplicates("committee_id").set_index("committee_id").cash
        for side in "dr":
            x[f"cash_{side}"] = x[f"cm_{side}"].map(cash_by_committee).fillna(0.0).clip(lower=0)
            x[f"runway_{side}"] = np.log1p(x[f"cash_{side}"] / (x[f"late_{side}"] / 3 + 1000.0))
            x[f"debt_{side}"] = x[f"cm_{side}"].map(load_debts(cycle)).fillna(0.0).clip(lower=0)
        second = {name: x.cm_d.map(measures[name]).fillna(0.0) - x.cm_r.map(measures[name]).fillna(0.0) for name in measures.columns}
        x = x.assign(**second, burn=difference_of_shares(x.assign(raised_d=x.raised_jun_d, raised_r=x.raised_jun_r, spent_d=x.spent_jun_d, spent_r=x.spent_jun_r), "spent", "raised").clip(-3, 3),
                     runway=x.runway_d - x.runway_r,
                     debt_burden=difference_of_shares(x.assign(raised_d=x.raised_jun_d, raised_r=x.raised_jun_r), "debt", "raised").clip(-3, 3),
                     fund_cost=difference_of_shares(x.assign(raised_d=x.raised_jun_d, raised_r=x.raised_jun_r), "fund", "raised").clip(-3, 3))
        out[cycle] = x.assign(
            persuasion_odds=odds((x.raised_jun_d - x.fund_d).clip(lower=0), (x.raised_jun_r - x.fund_r).clip(lower=0)),
            contact_odds=odds(x.contact_d, x.contact_r), fund_odds=odds(x.fund_d, x.fund_r), admin_odds=odds(x.admin_d, x.admin_r),
            fund_share=difference_of_shares(x, "fund", "spend"), admin_share=difference_of_shares(x, "admin", "spend"),
            inc_poll=(x.status == "D incumbent") * np.log1p(x.poll_d / 1000) - (x.status == "R incumbent") * np.log1p(x.poll_r / 1000),
            pro_vendor=pro["d"] - pro["r"], accel=difference_of_shares(x, "late", "spend"))
    return out
