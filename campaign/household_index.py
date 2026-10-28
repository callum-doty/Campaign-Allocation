"""
Two household-stress composites, fixed before either was compared with an
election result. Both are measured June over June in the election year,
which a September 1 decision can see: June prices come out in mid-July,
and banks' second-quarter delinquency rates in August.

- basket: the household cost basket from the HOUSING AFFORDABILITY TRACKER
  sheet (its ECONOMIC_INDEX tab). Each item's year-over-year price change,
  weighted by the sheet's own 1-3 weights. An item a year lacks (a series
  that starts later, a missing month) drops out and the rest are
  reweighted. Each item comes from FRED, as the series whose values match
  the sheet's quarterly history (its FRED_HISTORY_QUARTERLY tab), after
  allowing for series BLS has since rebased. Left out: Bread (the sheet
  gives it no weight), and Furniture and Cleaning Supplies (no FRED series
  matches the sheet's). Clothing and Personal Care use the CPI series for
  those categories, which track the sheet's closely but not exactly, and
  Doors & Windows uses metal window and door manufacturing, the closest
  match found.
- stress: credit and labor stress. The average year-over-year change, in
  percentage points, of the unemployment rate, U-6, and banks' credit card
  and single-family mortgage delinquency rates.

FRED serves today's revised values, not what was published at the time.
For prices the revisions are small.
"""

import pandas as pd

from .load_fred import load_fred

# (item as the sheet names it, FRED series, the sheet's weight)
BASKET = [
    ("Power (per KWH)", "APU000072610", 3), ("Milk", "APU0000709112", 3), ("Eggs", "APU0000708111", 3),
    ("Chicken (boneless)", "APU0000FF1101", 3), ("Home Insurance", "PCU9241269241262", 3),
    ("Auto Insurance", "PCU9241269241261", 3), ("Medical Care", "CPIMEDSL", 3), ("Food Overall", "CPIUFDSL", 3),
    ("Doors & Windows", "PCU332321332321", 2), ("HVAC System", "PCU333415333415", 1), ("Fridge", "WPU12410329", 2),
    ("Water Heater", "WPU106601", 1), ("Beef", "APU0000703112", 3), ("Coffee", "APU0000717311", 3),
    ("Car Parts", "CUSR0000SETC", 2), ("Childcare / Education", "CUSR0000SEEB", 3), ("Clothing", "CPIAPPSL", 2),
    ("Used Car", "CUSR0000SETA02", 3), ("Health Insurance (PPI)", "WPU411103", 3), ("Rent", "CUSR0000SEHA", 3),
    ("Natural Gas", "APU000072620", 3), ("Water/Sewer/Trash (CPI)", "CUSR0000SEHG", 3),
    ("Public Transit Fares", "CUUR0000SETG", 1), ("Cable / Streaming TV", "CUSR0000SERA02", 1),
    ("Internet / IT", "CUSR0000SEEE", 1), ("Food Away From Home", "CUSR0000SEFV", 2),
    ("Personal Care", "CUUR0000SAG1", 2), ("Rice per lb", "APU0000701312", 3), ("White Flour per lb", "APU0000701111", 3),
    ("White Sugar per lb", "APU0000715211", 3), ("Tomatoes per lb", "APU0000712311", 3),
    ("Bananas per lb", "APU0000711211", 3), ("Potatoes per lb", "APU0000712112", 3), ("Lettuce per lb", "APU0000FL2101", 3),
    ("Cheddar Cheese per lb", "APU0000710212", 3), ("Bacon per lb", "APU0000704111", 3), ("Potato Chips", "APU0000718311", 2),
    ("Disposable Diapers", "PCU3222913222912", 2), ("Oranges per lb", "APU0000711311", 3),
    ("Tissue Paper / Toilet Paper", "WPU091501235", 3), ("Trash Collection (PPI)", "PCU562111562111", 2),
]
# monthly series read in June; quarterly ones (FRED dates a quarter by its first day) in the second quarter
STRESS = [("Unemployment rate", "UNRATE"), ("U-6", "U6RATE"),
          ("Credit card delinquency", "DRCCLACBS"), ("Mortgage delinquency", "DRSFRMACBS")]
FRED_IDS = [sid for _, sid, _ in BASKET] + [sid for _, sid in STRESS]


def household_index(years):
    """DataFrame indexed by year: basket (weighted % change in prices, June
    over June), basket_items (how many items it covers), stress (average
    change in points)."""
    def june(s, year):          # June, or the second quarter for a quarterly series
        month = 4 if s.index.month.isin([1, 4, 7, 10]).all() else 6
        return s.get(pd.Timestamp(f"{year}-{month:02d}-01"))

    basket = [(load_fred(sid), weight) for _, sid, weight in BASKET]
    stress = [load_fred(sid) for _, sid in STRESS]
    rows = []
    for year in years:
        items = [(100 * (june(s, year) / june(s, year - 1) - 1), weight) for s, weight in basket
                 if june(s, year) is not None and june(s, year - 1) is not None]
        points = [june(s, year) - june(s, year - 1) for s in stress
                  if june(s, year) is not None and june(s, year - 1) is not None]
        rows.append({"year": year, "basket": sum(c * w for c, w in items) / sum(w for _, w in items),
                     "basket_items": len(items), "stress": sum(points) / len(points)})
    return pd.DataFrame(rows).set_index("year")
