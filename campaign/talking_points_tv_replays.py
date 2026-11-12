"""Post-hoc check for the TV talking-points test: how much of each channel's day is the same words
replayed across its programs, and is that political material or ads?

The 8-word mask in talking_points_tv.py removes runs that air in two or more programs on the same
channel-day (ads, promos, teases, replayed clips). This splits the masked words by span: political
if the span has two or more distinct words from POL and fewer than two from AD, ad-like if it has
two or more from AD, otherwise other. Keyword lists are crude and were set after seeing the
results; treat the split as a reading, not a test.

usage: python3 -m campaign.talking_points_tv_replays
"""
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from . import talking_points_tv as tv

POL = set("""trump trump's biden harris obama clinton democrat democrats democratic republican republicans gop congress
senate senator administration border immigration immigrants illegal election vote voters ballot
maga liberal liberals conservative conservatives radical impeachment pelosi schumer mcconnell vance
newsom mamdani epstein tariffs""".split())
AD = set("""call visit dot com offer free order today ask doctor side effects insurance save mortgage
medicare coverage plan prescription""".split())

def spans(path):
    d = pd.read_csv(path, sep="\t", dtype={"text": str}, keep_default_na=False)
    toks = [tv.tokens(t) for t in d.text]
    words = defaultdict(list)
    for p, ts in zip(d.identifier.values, toks):
        words[p].extend(ts)
    N = tv.MASK_N
    grams = {p: [hash(tuple(w[i:i + N])) for i in range(len(w) - N + 1)] for p, w in words.items()}
    earlier, reruns = set(), set()
    for p in sorted(words, key=lambda i: i.split("_")[2]):
        g = set(grams[p])
        if g and len(g & earlier) / len(g) >= tv.RERUN_SHARE:
            reruns.add(p); continue
        earlier |= g
    owners = defaultdict(set)
    for p, g in grams.items():
        if p not in reruns:
            for h in g: owners[h].add(p)
    out = Counter(); runs = Counter()
    for p, w in words.items():
        if p in reruns: continue
        m = np.zeros(len(w), bool)
        for i, h in enumerate(grams[p]):
            if len(owners[h]) > 1: m[i:i + N] = True
        out["all"] += len(w)
        i = 0
        while i < len(w):
            if not m[i]: i += 1; continue
            j = i
            while j < len(w) and m[j]: j += 1
            span = set(w[i:j]); n = j - i
            pol, ad = len(span & POL) >= 2, len(span & AD) >= 2
            kind = "political" if pol and not ad else ("ad-like" if ad else "other")
            out[kind] += n
            if kind == "political" and n >= 25: runs[" ".join(w[i:min(j, i + 25)])] += 1
            i = j
    return path.name, out, runs

if __name__ == "__main__":
    files = sorted(tv.CAPS.glob("*.tsv.gz"))
    rows, top = [], defaultdict(Counter)
    with ProcessPoolExecutor(12) as pool:
        for name, out, runs in pool.map(spans, files, chunksize=2):
            day, chan = name.split(".")[0].split("_")
            rows.append(dict(year=day[:4], channel=chan, **{k: out[k] / out["all"] * 100 for k in ("political", "ad-like", "other")}))
            if day >= "20250101": top[chan].update(runs)
    t = pd.DataFrame(rows).groupby(["year", "channel"]).mean().unstack().round(1)
    print(t.to_string())
    for c in top:
        print(c, [r[:110] for r, _ in top[c].most_common(6)])
