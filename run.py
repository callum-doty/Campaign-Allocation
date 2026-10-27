"""
Run the whole pipeline: fetch data, build races, backtest 2016-2024 forward
in time, solve the two-player game, price in TV-market cost, forecast 2026,
and write REPORT.md with its figures.

    python run.py              # download what's missing, then run
    python run.py --refresh    # also re-download 2026 FEC files (still being filed)
"""

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from campaign.backtest import backtest  # noqa: E402
from campaign.build_races import build_races  # noqa: E402
from campaign.channel_mix import channel_mix  # noqa: E402
from campaign.fetch import fetch  # noqa: E402
from campaign.forecast import forecast  # noqa: E402
from campaign.load_media_markets import load_media_markets  # noqa: E402
from campaign.media_cost import media_cost  # noqa: E402
from campaign.report import report  # noqa: E402
from campaign.solve_game import solve_game  # noqa: E402

CYCLES = (2012, 2014, 2016, 2018, 2020, 2022, 2024, 2026)
BACKTEST = (2016, 2018, 2020, 2022, 2024)
MARKET_DATA = (2022, 2024)        # completed cycles with TV-market data
LIVE = 2026


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help=f"re-download {LIVE} FEC files")
    args = parser.parse_args()

    fetch(CYCLES, refresh=(LIVE,) if args.refresh else ())
    races = {cycle: build_races(cycle) for cycle in CYCLES}
    backtests = [backtest(cycle, races) for cycle in BACKTEST]
    games = {b["cycle"]: solve_game(b["races"], b["model"]["sigma"]) for b in backtests}
    mixes = [channel_mix(cycle, load_media_markets(cycle)) for cycle in MARKET_DATA]
    media = []
    for b in backtests:
        if b["cycle"] in MARKET_DATA:
            dccc = next(m for m in mixes if m.cycle.iloc[0] == b["cycle"]).query("group == 'DCCC'")
            tv_share = dccc.amount[dccc.channel == "TV"].sum() / dccc.amount.sum()
            media.append(media_cost(b, load_media_markets(b["cycle"]), tv_share))
    fc = forecast(races, LIVE)
    report(races, backtests, fc, games, media, mixes)

    for b in backtests:
        print(f"{b['cycle']}: Brier {b['brier']:.3f}, gain from reallocating DCCC's money {b['gain']:+.2f} seats")
    print(f"{LIVE}: {fc['seats']['mean']:.0f} expected Democratic seats at D{fc['gb']:+.1f}, "
          f"{fc['seats']['p_majority']:.0%} chance of a majority")
    print("wrote REPORT.md and figures/")


if __name__ == "__main__":
    main()
