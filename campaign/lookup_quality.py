"""
Challenger quality from Wikipedia: did a nominee who isn't the incumbent
have an article on Jan 1 of the election year? Officeholders (state
legislators, mayors, former members) and people already known in public
life mostly do; first-time candidates mostly don't, and an article created
during the campaign doesn't count, since the check reads the article as it
stood on Jan 1 (its revision history), not as it is now.

For each nominee, two searches: first name, surname and state; then
surname, state and "congressional". A nominee is matched by the name on the
ballot when there is one (the name the public and Wikipedia use: Ami Bera,
not FEC's Amerish), and by FEC's legal name otherwise, as for most nominees
before an election's results are published. Searches keep the surname as
written, accents and apostrophes included (Wikipedia's search doesn't find
O'Donnell as odonnell). A hit has to be the same
person by name (same_person) and its Jan 1 text has to name the state and
something political (an office, a party, a campaign); a redirect doesn't
count. The
name rule is strict because loose matching finds relatives and namesakes:
on first initial alone, about 1 match in 5 was someone else (Mac Collins
for his son Mike, a Wyoming Mike Sullivan for Nebraska's Mark Sullivan).
The article's first name must equal the nominee's, be a prefix of it
(Dan/Daniel) or a common nickname (Bill/William); middle names, when both
have one, must share an initial; one more surname may follow (Kamlager-Dove,
Watson Coleman); a different generation rules it out (Joseph P. Kennedy Sr.
for Joseph P. Kennedy III), as does a disambiguation naming another state.
Name and state can't tell apart two politicians of the same name in one
state (Pennsylvania's Bob Brooks, a Republican state representative, and
Bob Brooks, the Democratic nominee in PA-07 in 2026), so an article whose
infobox gives only the other party is ruled out too.
Checked by hand in October 2026: 40 of 40 random matches were the right
person, and 98% of 302 sampled incumbents (who nearly all have articles)
were found, so a few real articles are missed.

Results go to data/raw/wikipedia/quality_{cycle}.csv, one row per nominee
by FEC candidate ID; a later run looks up only nominees not there yet.
Requests go one at a time, GAP seconds apart: Wikimedia answers faster
anonymous clients with HTTP 429, and its wait is honored.
"""

import re
import time
import unicodedata

import pandas as pd
import requests

API = "https://en.wikipedia.org/w/api.php"
GAP = 0.35                          # seconds between requests
POLITICAL = re.compile(r"congress|representative|politician|candidate|nominee|mayor|senat|legislat|assembly|delegate|"
                       r"commissioner|sheriff|council|governor|attorney general|elected|campaign|judge", re.I)
QUALIFIER_OK = re.compile(r"politician|congress|representative|legislat|senator|delegate|mayor", re.I)
GENERATIONS = {"jr", "sr", "ii", "iii", "iv"}
SUFFIXES = GENERATIONS | {"mr", "mrs", "ms", "dr", "hon", "the", "md", "esq", "phd"}
NICKNAMES = [group.split() for group in (
    "william bill will billy willie wm", "robert bob rob bobby", "richard rick dick rich ricky", "james jim jimmy jamie",
    "thomas tom tommy", "michael mike mick mickey", "joseph joe joey", "john jack johnny jon", "charles chuck charlie chip",
    "edward ed eddie ted ned", "daniel dan danny", "david dave", "steven stephen steve", "christopher chris kit",
    "anthony tony", "andrew andy drew", "gregory greg", "matthew matt", "kenneth ken kenny", "ronald ron ronnie",
    "donald don donnie", "timothy tim timmy", "patrick pat", "peter pete", "gerald gerry jerry", "lawrence larry",
    "frederick fred freddie", "benjamin ben benny", "samuel sam sammy", "nicholas nick nicky", "jonathan jon",
    "katherine catherine kathleen kate katie kathy cathy kat", "elizabeth liz beth betsy betty lisa",
    "margaret peggy maggie meg", "patricia pat patty trish", "jennifer jen jenny", "susan sue suzanne suzy",
    "deborah debra deb debbie", "rebecca becky", "alexander alexandra alex", "douglas doug", "bradley brad",
    "jeffrey jeff", "raymond ray", "theodore ted teddy", "victoria vicky tori", "abigail abby", "kimberly kim",
    "cynthia cindy", "judith judy", "valerie val", "harold hal harry", "henry hank harry", "francis frank",
    "franklin frank", "leonard len lenny", "eugene gene", "phillip philip phil", "russell russ", "walter walt",
    "albert al", "alan allan allen al", "dennis denny", "terrence terence terry", "vincent vince", "zachary zach",
    "joshua josh", "jacob jake", "nathaniel nathan nate", "carolyn caroline carol", "barbara barb", "pamela pam",
    "sandra sandy", "dorothy dot", "virginia ginny", "madeleine madeline maddy", "christine christina chris",
    "angela angie", "mary mj", "martin marty", "jerrold jerry", "jerome jerry", "louis lou", "salvatore sal",
    "vernon vern", "melvin mel")]
NICK = re.compile(r"\"([^\"]+)\"|'([^']+)'|\(([^)]+)\)")
INFOBOX_PARTY = re.compile(r"^\s*\|\s*party\s*=(.*)$", re.I | re.M)
PROSE_PARTY = re.compile(r"(democrat|republican)[^.]{0,60}?(?:member|politician|representative|senator|nominee|"
                         r"candidate|legislator|mayor)")


def fold(text):
    """Lower case, accents and punctuation off, hyphens as spaces: 'Luján' -> 'lujan'."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[.'\"]", "", text).replace("-", " ")


def split_name(name):
    """(surname, given names) as written, from an FEC name ('KEAN, THOMAS H. JR.', "TROUT, W. MICHAEL
    \"MIKE\"") or a plain one ('Bruce Westerman', 'THOMAS H KEAN JR')."""
    name = str(name).replace("\\", "")
    if "," in name and not set(fold(name.split(",", 1)[1]).split()) <= SUFFIXES:     # not 'OLSZEWSKI, JR.'
        surname, given = name.split(",", 1)
        return surname.strip(), given
    words = [w for w in name.replace(",", " ").split() if fold(w) not in SUFFIXES]    # the surname is before a suffix
    return words[-1], " ".join(words[:-1])


def name_parts(name):
    """(first names to try, middle names, surname words), folded."""
    surname, given = split_name(name)
    nick = NICK.search(given)
    given_words = [w for w in fold(NICK.sub(" ", given)).split() if w not in SUFFIXES]
    surname_words = [w for w in fold(surname).split() if w not in SUFFIXES]
    firsts = ([fold(next(g for g in nick.groups() if g)).split()[0]] if nick else []) + given_words[:1]
    if given_words and len(given_words[0]) == 1 and len(given_words) > 1:    # A. DONALD MCEACHIN goes by Donald
        firsts.append(given_words[1])
    return firsts, given_words[1:], surname_words


def same_first(a, b):
    a, b = fold(a), fold(b)
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))) or \
        any(a in group and b in group for group in NICKNAMES)


def same_person(title, name, state, states):
    """Whether article `title` can be nominee `name` of `state`, by name alone. `states` lists every state name."""
    qualifier = re.search(r"\((.*)\)", title)
    if qualifier:
        q = qualifier.group(1)
        if not (QUALIFIER_OK.search(q) or state.lower() in q.lower()):
            return False
        named = [s for s in states if re.search(rf"\b{s}\b", q, re.I)]
        named = [s for s in named if not any(s != o and s in o for o in named)]   # West Virginia, not also Virginia
        if any(s.lower() != state.lower() for s in named):
            return False
    raw = fold(re.sub(r"\(.*\)", "", title)).split()
    words = [w for w in raw if w not in SUFFIXES]
    title_gen, name_gen = GENERATIONS & set(raw), GENERATIONS & set(fold(NICK.sub(" ", str(name)).replace(",", " ")).split())
    if (title_gen and name_gen and title_gen != name_gen) or ("sr" in title_gen and "sr" not in name_gen):
        return False                                                    # a father or a grandson
    firsts, middles, surname = name_parts(name)
    if len(words) < 2 or not surname or not firsts:
        return False
    last = surname[-1]
    if last not in words[1:] or words.index(last) < len(words) - 2:      # one more surname may follow, no more
        return False
    initial_only = len(firsts[0]) == 1 and len(firsts) > 1 and words[0][:1] == firsts[0] and firsts[-1] in words
    if not (any(same_first(f, words[0]) for f in firsts) or initial_only):  # J PATRICK MURRAY: Jay Patrick Murray
        return False
    title_middles = [w for w in words[1:words.index(last)] if w not in surname and w not in firsts]
    return not (title_middles and middles) or any(a[0] == b[0] for a in title_middles for b in middles)


def party_ok(lead, party):
    """False if the article puts its subject only in the other major party than the nominee's ('D' or 'R'):
    by the infobox, or without one, by a party word just before an office in the opening sentences
    ('a former Republican member of the Pennsylvania House')."""
    parties = " ".join(INFOBOX_PARTY.findall(lead)).lower()
    if not parties:
        opening = " ".join(re.split(r"(?<=\.)\s", lead[max(lead.find("'''"), 0):])[:2]).lower()
        parties = " ".join(PROSE_PARTY.findall(opening))
    is_d, is_r = "democrat" in parties, "republican" in parties
    return not ((party == "D" and is_r and not is_d) or (party == "R" and is_d and not is_r))


def wiki(session, params):
    errors = 0
    while True:
        time.sleep(max(0.0, getattr(session, "last", 0.0) + GAP - time.monotonic()))
        session.last = time.monotonic()
        try:
            resp = session.get(API, params={**params, "format": "json", "formatversion": 2}, timeout=60)
            if resp.status_code == 429:                                  # throttled: wait as asked, then again
                time.sleep(float(resp.headers.get("retry-after", 15)) + 1)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            errors += 1
            if errors > 3:
                raise
            time.sleep(5 * 4 ** errors)


def lead_on(session, title, cycle):
    """(the article's lead as it stood on Jan 1 of `cycle`, that revision's time); ('', None) if it
    didn't exist yet or was a redirect."""
    r = wiki(session, {"action": "query", "prop": "revisions", "titles": title, "redirects": 1, "rvlimit": 1,
                       "rvdir": "older", "rvstart": f"{cycle}-01-01T00:00:00Z", "rvprop": "timestamp|content",
                       "rvslots": "main"})
    pages = r.get("query", {}).get("pages", [])
    revisions = pages[0].get("revisions", []) if pages else []
    if not revisions:
        return "", None
    text = revisions[0]["slots"]["main"].get("content", "")
    return ("" if text.lstrip().upper().startswith("#REDIRECT") else text.split("\n==")[0]), revisions[0]["timestamp"]


def lookup_quality(cycle, nominees, states, dest, headers):
    """Append a row to `dest` for each nominee who isn't the incumbent and isn't there yet: cand_id,
    district_id, party, name, title (the article, if any), as_of (its Jan 1 revision) and quality (1 if
    it existed then). `nominees` is load_candidates(cycle, ...), with FEC's name and the ballot's
    (ballot_name, when known); `states` maps postal code -> state name."""
    work = dest if dest.exists() else dest.with_name(f"{dest.stem}.partial.csv")   # a first run lands whole or resumes
    done = set(pd.read_csv(work, dtype=str).cand_id) if work.exists() else set()
    in_states = nominees.district_id.str[:2].isin(list(states))              # not territories or unfiled districts
    todo = nominees[(nominees.ici != "I") & nominees.cand_id.notna() & in_states & ~nominees.cand_id.isin(done)]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not todo.empty:
        print(f"looking up {len(todo)} {cycle} nominees on Wikipedia")
    session = requests.Session()
    session.headers.update(headers)
    names = list(states.values())
    for i, n in enumerate(todo.itertuples()):
        state = states[n.district_id[:2]]
        ballot = getattr(n, "ballot_name", None)
        known_as = ballot if isinstance(ballot, str) else n.name
        spellings = [known_as]
        if fold(split_name(n.name)[0]) != fold(split_name(known_as)[0]):          # either file can misspell a surname
            spellings.append(n.name)
        title, as_of = None, None
        for known_as in spellings:
            first = next((f for f in name_parts(known_as)[0] if len(f) > 1), "")      # not an initial
            surname = re.sub(r'["().]', "", split_name(known_as)[0]).strip()
            for query in (f'{first} "{surname}" {state}', f'"{surname.split()[-1]}" {state} congressional'):
                hits = wiki(session, {"action": "query", "list": "search", "srsearch": query, "srlimit": 10})
                for hit in hits.get("query", {}).get("search", []):
                    if not same_person(hit["title"], known_as, state, names):
                        continue
                    lead, when = lead_on(session, hit["title"], cycle)
                    if lead and state.lower() in lead.lower() and POLITICAL.search(lead) and party_ok(lead, n.party):
                        title, as_of = hit["title"], when
                        break
                if title:
                    break
            if title:
                break
        row = pd.DataFrame([{"cand_id": n.cand_id, "district_id": n.district_id, "party": n.party, "name": n.name,
                             "title": title, "as_of": as_of, "quality": int(title is not None)}])
        row.to_csv(work, mode="a", header=not work.exists(), index=False)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)
    if work != dest:
        work.rename(dest)
