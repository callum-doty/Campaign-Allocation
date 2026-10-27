"""
The allocation question as a two-player game: DCCC and NRCC each re-spend
their own budget, DCCC maximizing expected Democratic seats and NRCC
minimizing them, under the same spending curve:

    seats(xD, xR) = sum_i Phi((mu_i + f(xD_i) - f(obsD_i) - f(xR_i) + f(obsR_i)) / sigma)

anchored so the observed allocations reproduce the baseline forecast. The
payoff is concave in DCCC's spending and convex in NRCC's (outside a ~40
point blowout), so a stable equilibrium exists. It is found by fictitious
play: each side best-responds to the average of the other's past choices,
which converges in zero-sum games. "Exploitability" is how many expected
seats the two sides together could still gain by deviating: zero at an
equilibrium.

Question it answers: does the threat of an NRCC raid on a seat DCCC leaves
undefended make it rational to "garrison" seats that look safe?
"""

from scipy.stats import norm

from .allocate import allocate
from .spending_effect import spending_effect as f

ROUNDS = 40
STEP = 250_000
CAP_SHARE = 0.15


def solve_game(races, sigma, rounds=ROUNDS, step=STEP):
    """`races` needs mu, dccc and nrcc. Returns both sides' equilibrium
    allocations, the seats at the observed and equilibrium allocations, and
    the exploitability of each."""
    mu, obs_d, obs_r = (races[c].to_numpy(float) for c in ("mu", "dccc", "nrcc"))
    budget_d, budget_r = obs_d.sum(), obs_r.sum()
    fixed_d, fixed_r = f(obs_d), f(obs_r)

    exploit = {}
    x_d, x_r = obs_d.copy(), obs_r.copy()
    for k in range(rounds + 1):
        br_d = allocate(mu - f(x_r) + fixed_r, sigma, obs_d, budget_d, CAP_SHARE * budget_d, step=step)
        br_r = allocate(-(mu + f(x_d) - fixed_d), sigma, obs_r, budget_r, CAP_SHARE * budget_r, step=step)
        if k in (0, rounds):  # at the observed allocation, and at the end
            value = norm.cdf((mu + f(x_d) - fixed_d - f(x_r) + fixed_r) / sigma).sum()
            d_best = norm.cdf((mu + f(br_d) - fixed_d - f(x_r) + fixed_r) / sigma).sum()
            r_best = norm.cdf((mu + f(x_d) - fixed_d - f(br_r) + fixed_r) / sigma).sum()
            exploit["observed" if k == 0 else "equilibrium"] = (float(value), float(d_best - value + value - r_best))
        if k < rounds:
            x_d += (br_d - x_d) / (k + 2)
            x_r += (br_r - x_r) / (k + 2)
    return {"x_d": x_d, "x_r": x_r, "seats_observed": exploit["observed"][0],
            "seats_equilibrium": exploit["equilibrium"][0],
            "exploit_observed": exploit["observed"][1], "exploit_equilibrium": exploit["equilibrium"][1]}
