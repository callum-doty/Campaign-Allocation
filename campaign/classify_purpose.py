"""
What a campaign bought, from the purpose it wrote on each operating
expenditure. There are about 130,000 distinct purposes on 2.7 million payments
through June 30 of the election year, so this takes two stages:

1. Rules on the purpose text, first match wins. "DIRECT MAIL FUNDRAISING" is
   fundraising, "EMAIL" is digital rather than mail, "PAYROLL PROCESSING" is
   payroll. Where the text is vague, FEC's category code breaks the tie.
2. Vendors, for what's still vague ("MEDIA", "CONSULTING"). A vendor with at
   least 80% of its clearly labeled dollars in one category gets its vague
   payments counted there too. Only payments made so far count toward a vendor's
   record, so a forecast never learns from a later election.

Fundraising and voter contact can't be fully separated. "POSTAGE" or "DIGITAL
CONSULTING" can be either, and stays with its channel unless the text or
the vendor says otherwise.
"""

import re

import numpy as np
import pandas as pd

VAGUE = ("Consulting/unspecified", "Pass-through", "Other")
SPECIFIC_SHARE = 0.8
SPECIFIC_MIN_DOLLARS = 25_000
RULES = [  # (category, pattern on the upper-cased purpose), first match wins
    ("Fundraising", r"FUND ?RAIS|\bF/R\b|FINANC(E|IAL) (CONSULT|DIRECT|COMMITTEE)|DONOR|LIST (RENT|ACQUI|PURCHASE|MANAGE|EXCHANGE)|"
                    r"RENTAL LIST|ACQUISITION|AQUISITION|PROSPECT|TELEMARKET|TELEFUND|SOLICIT|COMMISSION|DIRECT RESPONSE|"
                    r"CALL ?TIME|DIRECT MARKETING"),
    ("Payroll/admin", r"PAYROLL|SALAR|WAGE|\bSTAFF|EMPLOYEE|\bTAX(ES)?\b|\bW-?2\b|HEALTH (INSURANCE|CARE)|BENEFITS?\b|STIPEND"),
    ("Processing fees", r"MERCHANT|PROCESSING|CARD (TRANSACTION |PROCESSING )?FEE|\bCC (TRANSACTION |PROCESSING )?FEE|TRANSACTION FEE|"
                        r"ONLINE CONTRIBUTION FEE|CONTRIBUTION FEE|EARMARK FEE|PLATFORM FEE|ACTBLUE|WINRED|ANEDOT|NGP.*FEE"),
    ("Polling/research", r"\bPOLL|SURVEY|FOCUS GROUP|RESEARCH|\bOPPO\b|OPPOSITION|BENCHMARK"),
    ("Pass-through", r"SEE (ALL )?(MEMO|BELOW|ATTACH|ITEMIZ|SCHED|ENTR|SUB)|MEMOS? (ITEMS? )?BELOW|ITEMIZ|CREDIT CARD( PAYMENT| BILL)?\b|"
                     r"CARD PAYMENT|PAYMENT: *SEE|^(EXPENSE )?REIMBURSEMENT$"),
    ("Payroll/admin", r"^INTERNET$|INTERNET (SERVICE|ACCESS|BILL)|CABLE.{0,3}INTERNET|WEB ?HOST|DOMAIN|E-?MAIL (HOST|ACCOUNT|SERVICE FEE)|"
                      r"CELL ?PHONE|TELEPHONE|PHONE SERVICE|WIRELESS|AIRLINE|AIRFARE|AIR TRAVEL|FLIGHT|AMTRAK"),
    ("Digital", r"DIGITAL|ONLINE|\bINTERNET|\bWEB|FACEBOOK|\bMETA\b|GOOGLE|YOUTUBE|SOCIAL MEDIA|STREAM|\bOTT\b|\bCTV\b|E-?MAIL|\bTEXT|"
                r"\bSMS\b|\bP2P\b|PEER.TO.PEER|BOOSTED"),
    ("Field/phones", r"CANVASS|\bGOTV|GET OUT THE VOTE|ORGANIZING"),
    ("Mail/print", r"MAIL|POSTAGE|PRINT|LITERATURE|POSTCARD|\bSIGNS?\b|PALM CARD|BUMPER|BANNER|T-?SHIRT|CAMPAIGN MATERIAL|COLLATERAL|DOOR HANGER"),
    ("Broadcast", r"MEDIA BUY|MEDIA PLACEMENT|PLACED MEDIA|PAID MEDIA|AIR ?TIME|\bAD BUY|MEDIA PURCHASE"),
    ("Production", r"PRODUCTION|VIDEO|CREATIVE|DESIGN|PHOTO|GRAPHIC"),
    ("Broadcast", r"\bTV\b|TELEVISION|BROADCAST|\bCABLE|\bRADIO|\bSPOTS?\b"),
    ("Field/phones", r"\bFIELD|PHONE BANK|PHONE CALL|ROBOCALL|AUTOMATED CALL|VOTER (FILE|DATA|CONTACT|ID)|\bDOOR|PETITION|BALLOT ACCESS|"
                     r"SIGNATURE|VOLUNTEER|DATABASE|\bVAN\b|DATA (SERVICE|MANAGE)|\bDATA\b"),
    ("Contributions", r"SPONSOR|\bDUES\b"),
    ("Events", r"CATERING|\bEVENT|TICKET|VENUE|RECEPTION|DINNER|FOOD|BEVERAGE|ROOM RENTAL|FACILITY|\bMEETING"),
    ("Payroll/admin", r"COMPLIANCE|ACCOUNT|BOOKKEEP|TREASURER|LEGAL|ATTORNEY|\bLAW\b|\bRENT\b|RENTAL|OFFICE|SOFTWARE|TRAVEL|TRANSPORT|LODGING|HOTEL|"
                      r"MILEAGE|\bGAS\b|FUEL|PARKING|TAXI|\bUBER|LYFT|RIDESHARE|\bMEALS?\b|INSURANCE|BANK|SERVICE CHARGE|UTILIT|PHONE|"
                      r"SUPPLIES|COMPUTER|EQUIPMENT|FILING FEE|QUALIFYING FEE|SUBSCRIPTION|MEMBERSHIP|STORAGE|SHIPPING|DELIVERY|LICENSE|SECURITY"),
    ("Contributions", r"CONTRIBUTION|DONATION|TRANSFER|\bREFUND"),
    ("Consulting/unspecified", r"CONSULT|STRATEG|\bMEDIA\b|ADVERT|COMMUNICATION|CAMPAIGN MANAG|PUBLIC RELATIONS|MARKETING|MANAGEMENT|SERVICES?\b"),
]
FEC_CATEGORY = {"001": "Payroll/admin", "002": "Payroll/admin", "003": "Fundraising", "005": "Polling/research",
                "006": "Mail/print", "007": "Events", "011": "Contributions", "012": "Contributions"}
COMPILED = [(category, re.compile(pattern)) for category, pattern in RULES]
LEGAL_WORDS = re.compile(r"\b(LLC|L L C|INC|INCORPORATED|CO|CORP|CORPORATION|COMPANY|LTD|LP|LLP|PLLC|THE)\b")


def by_rules(purpose):
    """The category for one upper-cased purpose."""
    return next((category for category, pattern in COMPILED if pattern.search(purpose)), "Other")


def classify_text(purposes, codes=None):
    """Stage 1: Series of categories from purposes, and FEC's category codes where the text is vague."""
    text = purposes.fillna("").str.upper().str.strip()
    found = text.map({p: by_rules(p) for p in text.unique()})
    if codes is not None:
        coded = codes.map(FEC_CATEGORY)
        found = found.where(~((found == "Other") & coded.notna()), coded)    # filers code consultants as overhead, so not those
    return found


def vendor_key(names):
    """Vendor names with case, punctuation and legal endings removed, so 'THE FROST GROUP, LLC' is 'FROST GROUP'."""
    return (names.fillna("").str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.replace(LEGAL_WORDS, " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def vendor_categories(vendors, categories, amounts):
    """Series, vendor -> category, for vendors with SPECIFIC_SHARE of at least SPECIFIC_MIN_DOLLARS of clearly
    labeled money in one category."""
    known = pd.DataFrame({"vendor": vendors, "category": categories, "amount": amounts})
    known = known[~known.category.isin(VAGUE) & (known.vendor != "") & (known.amount > 0)]
    totals = known.groupby(["vendor", "category"]).amount.sum().unstack(fill_value=0.0)
    keep = (totals.sum(axis=1) >= SPECIFIC_MIN_DOLLARS) & (totals.max(axis=1) >= SPECIFIC_SHARE * totals.sum(axis=1))
    return totals[keep].idxmax(axis=1)


def classify_vendors(vendors, categories, known):
    """Stage 2: vague categories replaced by `known` (vendor_categories) where the vendor is in it."""
    vague = categories.isin(VAGUE)
    filled = vendors[vague].map(known)
    out = categories.copy()
    out.loc[filled.dropna().index] = filled.dropna()
    return out
