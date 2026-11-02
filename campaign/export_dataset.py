"""
Everything the project collected, in one table: one row per Democratic or
Republican candidate on a House general-election ballot in the 50 states,
2012-2024, plus the 2026 nominees as they stand. Race, state and national
values repeat on each candidate's row. COLUMNS below says what each column
is, and is written out beside the table.

    python -m campaign.export_dataset

- Candidates: every D or R name on the ballot, not only the one per party the
  model uses, so a race's winner always has a row (the model can keep the
  loser of a D-vs-D race, if the loser spent more).
- FEC records come from load_candidates.py, for every name on the ballot.
  FEC's summary file has no row for some candidates in some years (Self and
  Ivey in 2024), so their ID comes from another year's file: June 30 money
  but blank cycle totals.
- won: the winner load_results names, with three repairs. MIT has no votes
  for two unopposed Florida members (FL-24 2016, FL-25 2020) and Alaska's 2022
  winner came from ranked-choice rounds it doesn't have. Party totals then
  match the official House every year, 2012-2024.
- Independent expenditures are filed by district and side here, not by
  candidate, so ie_for and ie_against sit on one row per party in a race: the
  nominee the model uses, or the party's top vote-getter if it has none.
- Left out, since they don't fit one row per candidate: third-party and
  primary candidates, which the project doesn't use; each transaction and
  each day of polling (totalled by race and by checkpoint here); FRED's
  monthly series (composites only) and EAVS's jurisdictions (state shares);
  and spending filed under a district that doesn't exist (MT-00 in 2024,
  $4.6M), which the model doesn't see either.
"""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest
from .build_races import build_races
from .forecast import forecast
from .household_index import household_index
from .load_candidates import RAW as FEC, load_candidates
from .load_early_money import load_early_money
from .load_early_vote import early_vote_share
from .load_expenditures import load_expenditures
from .load_media_markets import load_media_markets
from .load_presidential import load_presidential
from .load_pvi import LINES, load_pvi
from .load_results import RAW as MIT, clean_name, load_results
from .load_spending import load_spending
from .lookup_quality import SUFFIXES, fold, split_name
from .national_vote import national_vote

DATA = Path(__file__).resolve().parents[1] / "data"
QUALITY = DATA / "raw" / "wikipedia"
CYCLES = (2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026)
BACKTEST = (2016, 2018, 2020, 2022, 2024)
NOT_STATES = {"DC", "AS", "GU", "MP", "PR", "VI"}
MARKET_DATA = (2022, 2024, 2026)
PRESS = {2018: "news_bq_2018.csv", 2022: "news_bq.csv", 2024: "news_bq_2024.csv"}
RANKED_CHOICE = {(2022, "AK-00"): "MARY SATTLER PELTOLA"}
WEBALL = {0: "cand_id", 1: "fec_name", 2: "ici", 4: "party_code", 5: "total_receipts", 7: "total_disbursements",
          10: "cash_on_hand_end", 11: "candidate_contributions", 12: "candidate_loans", 16: "debts_owed",
          17: "individual_contributions", 18: "state", 19: "district", 25: "pac_contributions",
          26: "party_contributions", 27: "fec_coverage_end"}
PARTY = {"DEM": "D", "DFL": "D", "REP": "R"}
SIDE = {"dccc": "d", "outside_d": "d", "nrcc": "r", "outside_r": "r"}
CHANNELS = {"TV": "tv", "Digital": "digital", "Mail": "mail", "Radio": "radio", "Field/phones": "field_phones",
            "Media, unspecified": "media_unspecified", "Production": "production", "Other": "other"}
DATES = {"09-01": "sep01", "09-15": "sep15", "10-01": "oct01", "10-15": "oct15", "10-25": "oct25"}
PRES_YEARS = (2008, 2012, 2016, 2020, 2024)

COLUMNS = [  # name, level, source, meaning
    ("cycle", "race", "", "Election year"),
    ("state", "race", "", "Postal code"),
    ("district_id", "race", "", "State and district; at-large seats are 00. Lines are the ones that election used, so a number can mean different places in different years"),
    ("candidate", "candidate", "MIT Election Lab; FEC for 2026", "Name on the ballot, in capitals, sometimes misspelled in the source (JEFF DENHEM). 2026 has no ballot data: the hand-checked name, or FEC's legal name turned around"),
    ("party", "candidate", "MIT Election Lab; FEC for 2026", "D or R. Each candidate belongs to their biggest major-party line"),
    ("fec_candidate_id", "candidate", "FEC all-candidates file (weball)", "Blank when no FEC record matched. May come from another year's file; see total_disbursements"),
    ("fec_name", "candidate", "FEC weball", "Name as FEC writes it, sometimes misspelled"),
    ("nominee_basis", "candidate", "", "ballot = from the certified results. 2026 only: hand-checked = data/nominees_2026.csv; top spender = biggest spender of that party so far, not confirmed as the nominee"),
    ("used_in_model", "candidate", "build_races", "1 if this is the candidate the forecast uses for that party in a modeled race: its incumbent if it has one, then its biggest spender. Races under $100K of total spending and Alaska aren't modeled"),
    ("incumbent_status", "candidate", "FEC weball, corrected", "I incumbent, C challenger, O open seat, blank if unknown. Corrected where FEC lists a sitting member otherwise (load_candidates.py). In races between two sitting members, race_status counts only the Democrat"),
    ("votes", "candidate", "MIT Election Lab", "General-election votes, summed over fusion lines. Blank before results or where the state prints none for an unopposed member"),
    ("vote_share", "candidate", "MIT Election Lab", "Votes as a percentage of all votes cast in the race, third parties included"),
    ("won", "candidate", "MIT Election Lab, repaired", "1 won, 0 lost, blank for 2026 (no results yet)"),
    ("total_disbursements", "candidate", "FEC weball", "Money spent: the campaign's disbursements over the whole cycle, dollars. For 2026, reported so far. Blank if FEC's summary file has no row for them that year, even with an FEC ID; this and the columns through fec_coverage_end all come from that file"),
    ("total_receipts", "candidate", "FEC weball", "Money raised over the whole cycle, dollars"),
    ("individual_contributions", "candidate", "FEC weball", "From individuals, dollars"),
    ("pac_contributions", "candidate", "FEC weball", "From other political committees, dollars"),
    ("party_contributions", "candidate", "FEC weball", "From party committees, dollars"),
    ("candidate_contributions", "candidate", "FEC weball", "From the candidate, dollars"),
    ("candidate_loans", "candidate", "FEC weball", "Loans from the candidate, dollars"),
    ("debts_owed", "candidate", "FEC weball", "Debts owed by the campaign at the end of coverage, dollars"),
    ("cash_on_hand_end", "candidate", "FEC weball", "Cash at the end of coverage, dollars"),
    ("fec_coverage_end", "candidate", "FEC weball", "Date the totals above run through"),
    ("raised_by_jun30", "candidate", "FEC report summaries", "Receipts through June 30 of the election year, as filed by Aug 31, dollars"),
    ("spent_by_jun30", "candidate", "FEC report summaries", "Disbursements through June 30, as filed by Aug 31, dollars"),
    ("own_money_by_jun30", "candidate", "FEC report summaries", "The candidate's own contributions and loans within raised_by_jun30, dollars"),
    ("ie_for", "candidate", "FEC independent expenditures", "Independent spending on this candidate's side (supporting them or opposing the other party), party committee included, dollars. On one row per party in a race (the model's nominee, or else the top vote-getter), so summing the column counts each dollar once"),
    ("ie_against", "candidate", "FEC independent expenditures", "Independent spending on the other party's side, dollars, on the same rows as ie_for"),
    ("wiki_article_title", "candidate", "Wikipedia revision history", "Article the challenger-quality lookup matched. Non-incumbents the model uses only"),
    ("wiki_article_as_of", "candidate", "Wikipedia revision history", "Timestamp of the revision that stood on Jan 1 of the election year"),
    ("had_wiki_article_jan1", "candidate", "Wikipedia revision history", "1 if the nominee had an article on Jan 1, 0 if not, blank for incumbents, who aren't looked up. The model's challenger-quality feature"),
    ("wiki_views_jan_aug", "candidate", "Wikimedia page views (data/wikipedia_views.csv)", "Page views Jan 1-Aug 31. Competitive races in 2018, 2022 and 2024 only; blank if no article was found. From an earlier, looser name match than the quality lookup, so some articles may be namesakes"),
    ("gdelt_articles", "candidate", "GDELT via BigQuery (data/news_bq*.csv)", "Articles Jan 1-Aug 31 naming the nominee and their state with a political theme. Competitive races in 2018, 2022 and 2024 only. GDELT misses names that are ordinary words (Golden, Wild, Lamb) and common names pick up other people"),
    ("gdelt_mean_tone", "candidate", "GDELT via BigQuery", "Average article tone; negative is unfavorable"),
    ("gdelt_negative_articles", "candidate", "GDELT via BigQuery", "Articles with tone of -5 or lower"),
    ("race_d_votes", "race", "MIT Election Lab", "Votes for every Democrat in the race"),
    ("race_r_votes", "race", "MIT Election Lab", "Votes for every Republican in the race"),
    ("race_total_votes", "race", "MIT Election Lab", "All votes cast in the race"),
    ("race_margin", "race", "MIT Election Lab", "Democratic minus Republican, points of the two-party vote. +/-100 means one party had no candidate, including D-vs-D races"),
    ("race_winner", "race", "MIT Election Lab, repaired", "Name of the race's winner"),
    ("race_dccc_spent", "race", "FEC independent expenditures", "DCCC's independent spending, dollars. 2026 misses its coordinated spending, which is how it spends after NRSC v. FEC"),
    ("race_nrcc_spent", "race", "FEC independent expenditures", "NRCC's independent spending, dollars. Same 2026 gap"),
    ("race_outside_d", "race", "FEC independent expenditures", "Non-party-committee spending on the Democratic side, dollars"),
    ("race_outside_r", "race", "FEC independent expenditures", "Non-party-committee spending on the Republican side, dollars"),
    ("race_outside_d_by_aug31", "race", "FEC independent expenditures", "race_outside_d that had reached voters by Aug 31, dollars"),
    ("race_outside_r_by_aug31", "race", "FEC independent expenditures", "race_outside_r that had reached voters by Aug 31, dollars"),
    *[(f"race_ie_{side}_{slug}", "race", "FEC independent expenditures",
       f"{'Democratic' if side == 'd' else 'Republican'}-side independent spending, party committee included: {name}, from the filer's purpose description, dollars")
      for side in "dr" for name, slug in CHANNELS.items()],
    ("pvi", "race", "The Downballot presidential results by district", "Partisan lean: district's Democratic two-party share minus the nation's, in the last presidential election before the race, points"),
    *[(f"pres_d_share_{year}", "race", "The Downballot presidential results by district",
       f"Democratic share of the {year} two-party presidential vote on this race's lines, blank where not published") for year in PRES_YEARS],
    ("race_status", "race", "build_races", "D incumbent, R incumbent or Open. Blank for races the model doesn't cover, as are all the model columns below"),
    ("race_tier", "race", "build_races", "Rating from lean and incumbency, for reporting only"),
    ("d_incumbent", "race", "build_races", "Model feature: 1 if the Democrat is the incumbent"),
    ("open_seat", "race", "build_races", "Model feature: 1 if neither nominee is the incumbent"),
    ("floor_share", "race", "build_races", "Model feature: log of the Democratic share of money neither national committee controls, visible by Sept 1"),
    ("fresh_d", "race", "build_races", "Model feature: the Democrat is a first-term incumbent"),
    ("fresh_r", "race", "build_races", "Model feature: the Republican is a first-term incumbent"),
    ("over_inc", "race", "build_races", "Model feature: how far the seat's party ran ahead of its lean last time, when its incumbent is running again"),
    ("over_open", "race", "build_races", "Model feature: the same, when the seat is open"),
    ("redrawn", "race", "build_races", "1 if the state's lines changed since the last election"),
    ("inc_redrawn", "race", "build_races", "redrawn, signed by the incumbent's party (+1 D, -1 R)"),
    ("quality", "race", "build_races", "Model feature: Democrat's had_wiki_article_jan1 minus the Republican's, incumbents counting as 0"),
    ("forecast_margin", "race", "backtest, forecast", "Forecast Democratic margin, points, from the final generic ballot. 2016-2024 fit only on earlier elections; 2026 on all of them; 2012 and 2014 have no forecast"),
    ("forecast_p_dem", "race", "backtest, forecast", "Forecast chance the Democrat wins, from the final generic ballot"),
    *[(f"forecast_p_dem_{slug}", "race", "dated_forecast", f"The same as the forecast stood on {date}, from polls and outside money seen by then. 2016-2024 only") for date, slug in DATES.items()],
    ("model_dccc_dollars", "race", "backtest", "The model's best split of DCCC's actual budget that year at the default spending assumption, dollars. 2016-2024 only"),
    ("tv_market", "race", "Cervas, Nielsen markets by district", "District's main TV market. 2022, 2024 and 2026 lines; not Alaska or Hawaii"),
    ("tv_market_population", "race", "Cervas", "People in that market"),
    ("district_population", "race", "Cervas", "People in the district"),
    ("tv_main_market_share", "race", "Cervas", "Share of the district in its main market"),
    ("tv_market_count", "race", "Cervas", "Markets the district touches"),
    ("tv_reach", "race", "Cervas", "People in the district's markets per person in the district"),
    ("early_vote_share", "state", "EAVS", "State's share of ballots cast before Election Day. Years without a complete survey use the nearest one"),
    ("generic_ballot_final", "national", "data/generic_ballot.csv", "Final generic-ballot average heading into the election, Democratic lead in points"),
    *[(f"generic_ballot_{slug}", "national", "data/generic_ballot_dated.csv", f"Generic-ballot average on {date}, through 2024") for date, slug in DATES.items()],
    ("national_house_margin", "national", "MIT Election Lab", "National House popular vote, Democratic minus Republican, points. Blank for 2026"),
    ("household_basket_change", "national", "FRED, HOUSING AFFORDABILITY TRACKER weights", "Household cost basket, percent change June over June of the election year"),
    ("household_stress_change", "national", "FRED", "Credit-and-labor stress index, change in points, June over June"),
]


def candidate_votes(cycle):
    """Each general-election candidate's votes over all their lines and the race's total, as load_results filters them."""
    raw = pd.read_csv(MIT, dtype={"district": str}, low_memory=False)
    keep = (raw.year == cycle) & (raw.stage == "GEN") & (raw["mode"] == "TOTAL")
    for col in ("runoff", "special", "writein"):
        keep &= ~raw[col].astype(str).str.upper().eq("TRUE")
    g = raw[keep].assign(district_id=lambda r: r.state_po + "-" + r.district.str.zfill(2),
                         ballot_name=lambda r: r.candidate.astype(str).map(clean_name),
                         votes=lambda r: pd.to_numeric(r.candidatevotes, errors="coerce").fillna(0))
    votes = g.groupby(["district_id", "ballot_name"]).votes.sum().reset_index()
    totals = g.groupby("district_id").totalvotes.max().rename("race_total_votes").reset_index()
    return votes, totals


@lru_cache(maxsize=None)
def weball(cycle):
    fec = pd.read_csv(FEC / f"weball{str(cycle)[2:]}.txt", sep="|", header=None, dtype=str,
                      usecols=list(WEBALL)).rename(columns=WEBALL)
    fec = fec[fec.cand_id.str.startswith("H")].drop_duplicates("cand_id")
    money = ["total_receipts", "total_disbursements", "cash_on_hand_end", "candidate_contributions", "candidate_loans",
             "debts_owed", "individual_contributions", "pac_contributions", "party_contributions"]
    fec[money] = fec[money].apply(pd.to_numeric, errors="coerce")
    return fec.assign(fec_party=fec.party_code.map(PARTY), fec_district=fec.state + "-" + fec.district.str.zfill(2))


def display_name(fec_name):
    """FEC's 'MCGUIRE, JOHN J. MR. III' as 'JOHN J. MCGUIRE': no ballot data says what a 2026 candidate goes by."""
    surname, given = split_name(fec_name)
    return " ".join([w for w in given.split() if fold(w) not in SUFFIXES] + [surname])


def candidate_rows(cycle):
    """One row per D or R candidate with name, FEC ID, incumbency, votes, won, FEC totals."""
    results = load_results(cycle)
    nominees = load_candidates(cycle, results).dropna(subset=["district_id"]).assign(side_nominee=True)
    nominees = nominees.rename(columns={"name": "nominee_fec_name"})
    fec = weball(cycle)
    if results.empty:
        rows = nominees[nominees.district_id.isin(load_pvi(cycle).index)]      # not FEC's TX-00 and the like
        rows = rows.assign(state=rows.district_id.str[:2])
        listed = pd.read_csv(DATA / f"nominees_{cycle}.csv", dtype=str)
        hand = rows.set_index(["district_id", "party"]).index.isin(listed.set_index(["district_id", "party"]).index)
        rows["nominee_basis"] = np.where(hand, "hand-checked", "top spender")
        turned = rows.nominee_fec_name.map(display_name)
        rows["candidate"] = rows.ballot_name.where(rows.ballot_name.notna(), turned)
        rows["votes"], rows["race_total_votes"] = np.nan, np.nan
    else:
        ballot = pd.concat([results[["district_id", col]].explode(col).rename(columns={col: "ballot_name"}).assign(party=party)
                            for party, col in (("D", "d_names"), ("R", "r_names"))]).dropna(subset=["ballot_name"])
        everyone = load_candidates(cycle, results, every_name=True).rename(columns={"name": "nominee_fec_name"})
        rows = ballot.merge(everyone.drop(columns="disbursements"), on=["district_id", "party", "ballot_name"], how="left").merge(
            nominees[["district_id", "party", "ballot_name", "side_nominee"]], on=["district_id", "party", "ballot_name"], how="left")
        rows["state"] = rows.district_id.str[:2]
        votes, totals = candidate_votes(cycle)
        rows = rows.merge(votes, on=["district_id", "ballot_name"], how="left").merge(totals, on="district_id", how="left")
        rows["votes"] = rows.votes.where(rows.votes > 0)
        rows["race_total_votes"] = rows.race_total_votes.where(rows.race_total_votes > 0)
        rows["nominee_basis"] = "ballot"
        rows["candidate"] = rows.ballot_name
    rows = rows[~rows.state.isin(NOT_STATES)].copy()
    rows["side_nominee"] = rows.side_nominee.eq(True)
    rows = rows.merge(fec.drop(columns=["party_code", "state", "district", "fec_party", "fec_district", "ici"]), on="cand_id", how="left")
    rows["incumbent_status"] = rows.ici
    rows["fec_name"] = rows.fec_name.where(rows.fec_name.notna(), rows.nominee_fec_name)
    rows["candidate"] = rows.candidate.str.replace("\\", "", regex=False).str.upper().str.strip()
    rows["vote_share"] = 100 * rows.votes / rows.race_total_votes
    if results.empty:
        rows["won"], rows["race_winner"] = np.nan, None
        return rows, results
    winners = results.set_index("district_id").winner.copy()
    only = rows.groupby("district_id").ballot_name.agg(lambda names: names.iloc[0] if len(names) == 1 else None)
    winners = winners.where(winners.notna(), only.reindex(winners.index))
    for (year, district), name in RANKED_CHOICE.items():
        if year == cycle:
            winners[district] = name
    rows["won"] = (rows.ballot_name == rows.district_id.map(winners)).astype(float)
    rows["race_winner"] = rows.district_id.map(winners).str.replace("\\", "", regex=False)
    return rows, results


def party_leads(rows):
    """One row per party in a race, to carry what's filed by district and side: the nominee the model uses, or the top vote-getter if it has none."""
    has_nominee = rows.groupby(["district_id", "party"]).side_nominee.transform("any")
    top = rows.votes.fillna(0) == rows.groupby(["district_id", "party"]).votes.transform("max").fillna(0)
    lead = rows.side_nominee | (~has_nominee & top)
    return lead & ~(lead & rows.assign(lead=lead)[lead].duplicated(["district_id", "party"]).reindex(rows.index, fill_value=False))


def press(cycle):
    """gdelt_* by district and party, counting what the tone test counted: a political theme and the nominee's state."""
    if cycle not in PRESS:
        return pd.DataFrame(columns=["district_id", "party", "gdelt_articles", "gdelt_mean_tone", "gdelt_negative_articles"])
    b = pd.read_csv(DATA / PRESS[cycle])
    b[["district_id", "side"]] = b.key.str.split(":", expand=True)
    queried = pd.MultiIndex.from_product([b.district_id.unique(), ["d", "r"]], names=["district_id", "side"])
    b = b[b.political.astype(str).str.lower().eq("true") & b.in_state.astype(str).str.lower().eq("true")]
    g = b.groupby(["district_id", "side"])[["articles", "tone_sum", "neg"]].sum().reindex(queried, fill_value=0).reset_index()
    return pd.DataFrame({"district_id": g.district_id, "party": g.side.str.upper(), "gdelt_articles": g.articles,
                         "gdelt_mean_tone": (g.tone_sum / g.articles.where(g.articles > 0)),
                         "gdelt_negative_articles": g.neg})


def race_spending(cycle):
    """Independent spending by district: DCCC, NRCC, outside by side, by Aug 31, and by side and channel."""
    spent = load_spending(cycle, pd.Timestamp(f"{cycle}-08-31")).rename(columns={
        "dccc": "race_dccc_spent", "nrcc": "race_nrcc_spent", "outside_d": "race_outside_d", "outside_r": "race_outside_r",
        "outside_aug_d": "race_outside_d_by_aug31", "outside_aug_r": "race_outside_r_by_aug31"})
    ie = load_expenditures(cycle)
    ie = ie.assign(side=ie.bucket.map(SIDE), slug=ie.channel.map(CHANNELS))
    wide = ie.pivot_table(index="district_id", columns=["side", "slug"], values="amount", aggfunc="sum", fill_value=0)
    wide.columns = [f"race_ie_{side}_{slug}" for side, slug in wide.columns]
    return spent.merge(wide.reset_index(), on="district_id", how="left")


def export_dataset(races, backtests, fc, dest=DATA / "all_candidates.csv"):
    """`races` maps cycle -> build_races(cycle); `backtests` the backtest() results; `fc` the forecast() result.
    Writes `dest` and, beside it, `{stem}_columns.csv` saying what each column is."""
    forecasts = {b["cycle"]: b["races"] for b in backtests}
    forecasts[fc["cycle"]] = fc["races"]
    views = pd.read_csv(DATA / "wikipedia_views.csv")
    ballot = pd.read_csv(DATA / "generic_ballot.csv").set_index("cycle").generic_ballot
    dated = pd.read_csv(DATA / "generic_ballot_dated.csv").pivot(index="cycle", columns="date", values="generic_ballot")
    stress = household_index(CYCLES)
    national_margin = national_vote(tuple(c for c in CYCLES if c != fc["cycle"]))
    frames = []
    for cycle in CYCLES:
        rows, results = candidate_rows(cycle)
        modeled = races[cycle]
        rows["used_in_model"] = (rows.side_nominee & rows.district_id.isin(modeled.district_id)).astype(int)

        early = load_early_money(cycle).rename(columns={"raised": "raised_by_jun30", "spent": "spent_by_jun30", "own": "own_money_by_jun30"})
        rows = rows.merge(early[["raised_by_jun30", "spent_by_jun30", "own_money_by_jun30"]], left_on="cand_id", right_index=True, how="left")

        quality = pd.read_csv(QUALITY / f"quality_{cycle}.csv", dtype={"cand_id": str}).drop_duplicates("cand_id")
        rows = rows.merge(quality[["cand_id", "title", "as_of", "quality"]].rename(columns={
            "title": "wiki_article_title", "as_of": "wiki_article_as_of", "quality": "had_wiki_article_jan1"}), on="cand_id", how="left")
        incumbent = rows.incumbent_status == "I"
        for col in ("wiki_article_title", "wiki_article_as_of", "had_wiki_article_jan1"):
            rows.loc[incumbent | ~rows.side_nominee, col] = np.nan

        seen = views[views.cycle == cycle][["district_id", "party", "views_jan_aug"]].rename(columns={"views_jan_aug": "wiki_views_jan_aug"})
        rows = rows.merge(seen, on=["district_id", "party"], how="left").merge(press(cycle), on=["district_id", "party"], how="left")
        spent = race_spending(cycle)
        rows = rows.merge(spent, on="district_id", how="left")
        channels = [c for c in spent.columns if c.startswith("race_ie_")]
        for c in spent.columns.drop("district_id"):
            rows[c] = rows[c].fillna(0.0)
        on_d = rows[[c for c in channels if c.startswith("race_ie_d_")]].sum(axis=1)
        on_r = rows[[c for c in channels if c.startswith("race_ie_r_")]].sum(axis=1)
        rows["ie_for"] = np.where(rows.party == "D", on_d, on_r)
        rows["ie_against"] = np.where(rows.party == "D", on_r, on_d)
        lead = party_leads(rows)
        for col in ("wiki_views_jan_aug", "gdelt_articles", "gdelt_mean_tone", "gdelt_negative_articles", "ie_for", "ie_against"):
            rows.loc[~lead, col] = np.nan

        if not results.empty:
            race = results[["district_id", "d_votes", "r_votes", "margin"]].rename(columns={
                "d_votes": "race_d_votes", "r_votes": "race_r_votes", "margin": "race_margin"})
            race[["race_d_votes", "race_r_votes"]] = race[["race_d_votes", "race_r_votes"]].clip(lower=0)
            rows = rows.merge(race, on="district_id", how="left")
        rows["pvi"] = rows.district_id.map(load_pvi(cycle))
        pres = load_presidential(LINES[cycle]).pivot_table(index="district_id", columns="pres_year", values="d_share")
        pres.columns = [f"pres_d_share_{int(y)}" for y in pres.columns]
        rows = rows.merge(pres, left_on="district_id", right_index=True, how="left")

        features = modeled[["district_id", "status", "tier", "d_incumbent", "open_seat", "floor_share", "fresh_d", "fresh_r",
                            "over_inc", "over_open", "redrawn", "inc_redrawn", "quality"]].rename(
            columns={"status": "race_status", "tier": "race_tier"})
        rows = rows.merge(features, on="district_id", how="left")

        if cycle in forecasts:
            f = forecasts[cycle]
            scored = pd.DataFrame({"district_id": f.district_id, "forecast_margin": f.mu, "forecast_p_dem": f.p})
            for date, slug in DATES.items():
                if f"p_{date}" in f:
                    scored[f"forecast_p_dem_{slug}"] = f[f"p_{date}"]
            if "model_dccc" in f:
                scored["model_dccc_dollars"] = f.model_dccc
            rows = rows.merge(scored, on="district_id", how="left")

        if cycle in MARKET_DATA:
            tv = load_media_markets(cycle)[["district_id", "market_name", "market_pop", "pop", "main_market_share", "n_markets", "reach"]].rename(columns={
                "market_name": "tv_market", "market_pop": "tv_market_population", "pop": "district_population",
                "main_market_share": "tv_main_market_share", "n_markets": "tv_market_count", "reach": "tv_reach"})
            rows = rows.merge(tv, on="district_id", how="left")

        rows["early_vote_share"] = rows.state.map(early_vote_share(cycle))
        rows["cycle"] = cycle
        rows["generic_ballot_final"] = ballot.get(cycle)
        for date, slug in DATES.items():
            rows[f"generic_ballot_{slug}"] = dated[date].get(cycle)
        rows["national_house_margin"] = national_margin.get(cycle, np.nan)
        rows["household_basket_change"] = stress.basket.get(cycle)
        rows["household_stress_change"] = stress.stress.get(cycle)
        frames.append(rows.rename(columns={"cand_id": "fec_candidate_id"}).dropna(axis=1, how="all"))

    table = pd.concat(frames, ignore_index=True).reindex(columns=[c[0] for c in COLUMNS])
    table = table.sort_values(["cycle", "district_id", "party", "candidate"], kind="stable")
    for col in ("won", "used_in_model", "had_wiki_article_jan1", "votes", "race_d_votes", "race_r_votes", "race_total_votes",
                "wiki_views_jan_aug", "gdelt_articles", "gdelt_negative_articles", "d_incumbent", "open_seat", "fresh_d", "fresh_r",
                "redrawn", "inc_redrawn", "quality", "tv_market_population", "district_population", "tv_market_count"):
        table[col] = pd.to_numeric(table[col], errors="coerce").round().astype("Int64")
    decimals = {c: 2 for c in table.select_dtypes("float").columns}
    decimals.update({c: 4 for c in table.columns if c.startswith(("forecast_p_", "pres_d_share_", "early_vote", "tv_main", "gdelt_mean", "tv_reach", "floor_share"))})
    table = table.round(decimals)
    table.to_csv(dest, index=False)
    with open(dest.with_name(f"{dest.stem}_columns.csv"), "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["column", "level", "source", "meaning"])
        writer.writerows(COLUMNS)
    return table


if __name__ == "__main__":
    all_races = {cycle: build_races(cycle) for cycle in CYCLES}
    done = export_dataset(all_races, [backtest(cycle, all_races) for cycle in BACKTEST], forecast(all_races, CYCLES[-1]))
    print(f"wrote {len(done)} rows, {done.shape[1]} columns")
