"""
What each House nominee's campaign paid for: FEC operating expenditures
(Schedule B, bulk file oppexp), one row per payment, with what it bought
(classify_purpose.py). About 65MB a cycle zipped; downloaded on first use and
not part of run.py.

- Which committee: the one load_early_money picks for the nominee, so these
  rows and the July report's totals describe the same campaign.
- Memo rows: a memo row restates part of the payment above it. Counted as well,
  campaigns' totals run 10-18% over what they reported; left out, 90% come in at
  or under 99% of it (the rest is refunds and loan repayments, which aren't
  operating expenditures). But a credit card bill's only useful detail is in its
  memo rows, so a payment with memo rows is replaced by them, scaled so they can't
  exceed it, with any remainder left on the payment. The total doesn't change.
- Other forms: a few rows come from presidential or party forms and are dropped.
- Timing: rows carry the date paid. A forecast on Sept 1 uses only rows dated
  through June 30, which the July report had disclosed. Later amendments to those
  reports are in this file, a small leak the July totals don't have.
"""

import zipfile
from pathlib import Path

import pandas as pd
import requests

from .classify_purpose import classify_text, classify_vendors, vendor_categories, vendor_key
from .load_candidates import load_candidates
from .load_early_money import load_early_money
from .load_results import load_results

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "fec" / "oppexp"
FEC_BULK = "https://www.fec.gov/files/bulk-downloads"
COLUMNS = ["CMTE_ID", "AMNDT_IND", "RPT_YR", "RPT_TP", "IMAGE_NUM", "LINE_NUM", "FORM_TP_CD", "SCHED_TP_CD", "NAME", "CITY",
           "STATE", "ZIP_CODE", "TRANSACTION_DT", "TRANSACTION_AMT", "TRANSACTION_PGI", "PURPOSE", "CATEGORY", "CATEGORY_DESC",
           "MEMO_CD", "MEMO_TEXT", "ENTITY_TP", "SUB_ID", "FILE_NUM", "TRAN_ID", "BACK_REF_TRAN_ID"]
KEEP = ["CMTE_ID", "NAME", "TRANSACTION_DT", "TRANSACTION_AMT", "PURPOSE", "CATEGORY", "FILE_NUM", "TRAN_ID", "BACK_REF_TRAN_ID", "MEMO_CD"]


def nominee_committees(cycle):
    """Series, cand_id -> committee_id, for the cycle's nominees with a July report."""
    candidates = load_candidates(cycle, load_results(cycle)).cand_id.dropna()
    early = load_early_money(cycle)
    return early.committee_id[early.index.intersection(candidates)]


def read_rows(cycle):
    """The cycle's rows for nominees' committees, from a cached copy if there is one."""
    cached = RAW / f"rows_{cycle}.csv.gz"
    if cached.exists():
        return pd.read_csv(cached, dtype=str)
    zipped = RAW / f"oppexp{str(cycle)[2:]}.zip"
    if not zipped.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"downloading {zipped.name}")
        resp = requests.get(f"{FEC_BULK}/{cycle}/{zipped.name}", timeout=600)
        resp.raise_for_status()
        zipped.write_bytes(resp.content)
    wanted = set(nominee_committees(cycle))
    with zipfile.ZipFile(zipped) as z, z.open(z.namelist()[0]) as f:
        chunks = [c[c.CMTE_ID.isin(wanted) & (c.FORM_TP_CD == "F3")][KEEP]
                  for c in pd.read_csv(f, sep="|", header=None, names=COLUMNS + ["_"], dtype=str, encoding="latin-1",
                                       chunksize=500_000, quoting=3)]
    rows = pd.concat(chunks, ignore_index=True)
    rows.to_csv(cached, index=False)
    return rows


def swap_memos(rows):
    """Each payment with memo rows replaced by them, scaled to fit, keeping any remainder on the payment. `rows` has
    CMTE_ID, FILE_NUM, TRAN_ID, BACK_REF_TRAN_ID, MEMO_CD, amount. Memo rows without a payment above them are dropped."""
    parent = ["CMTE_ID", "FILE_NUM", "TRAN_ID"]
    memo = rows[(rows.MEMO_CD == "X") & rows.BACK_REF_TRAN_ID.notna()]
    top = rows[rows.MEMO_CD != "X"]
    detail = memo.groupby(["CMTE_ID", "FILE_NUM", "BACK_REF_TRAN_ID"]).amount.sum().rename_axis(parent).rename("detail")
    top = top.join(detail, on=parent)
    has = top.detail.notna() & (top.amount > 0) & (top.detail > 0)
    scale = (top.amount[has] / top.detail[has]).clip(upper=1)
    parents = top.loc[has, parent].assign(scale=scale).drop_duplicates(parent)
    used = memo.merge(parents.rename(columns={"TRAN_ID": "BACK_REF_TRAN_ID"}), on=["CMTE_ID", "FILE_NUM", "BACK_REF_TRAN_ID"])
    used = used.assign(amount=used.amount * used.scale).drop(columns="scale")
    top = top.assign(amount=(top.amount - top.detail).clip(lower=0).where(has, top.amount)).drop(columns="detail")
    both = pd.concat([top, used], ignore_index=True)
    return both[both.amount != 0]


def load_oppexp(cycle):
    """committee_id, vendor (normalized), purpose, amount, date, stage1 (category from the text and FEC's code)."""
    rows = read_rows(cycle)
    rows = rows.assign(amount=pd.to_numeric(rows.TRANSACTION_AMT, errors="coerce"),
                       date=pd.to_datetime(rows.TRANSACTION_DT, format="%m/%d/%Y", errors="coerce")).dropna(subset=["amount", "date"])
    rows = swap_memos(rows)
    return pd.DataFrame({"committee_id": rows.CMTE_ID, "vendor": vendor_key(rows.NAME), "purpose": rows.PURPOSE, "amount": rows.amount,
                         "date": rows.date, "stage1": classify_text(rows.PURPOSE, rows.CATEGORY), "cycle": cycle}).reset_index(drop=True)


def categorize(rows, history):
    """`rows` with category, the vague ones filled in from vendors' clearly labeled money in `history` (rows up to the
    moment being described: never later ones, for a forecast)."""
    known = vendor_categories(history.vendor, history.stage1, history.amount)
    return rows.assign(category=classify_vendors(rows.vendor, rows.stage1, known))
