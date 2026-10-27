"""Expected Democratic margin (points) for each race under a fitted baseline.
Win probability is then Phi(margin / sigma)."""


def predict_margin(races, model):
    c = model["coef"]
    return (c["intercept"] + sum(c[f] * races[f] for f in c if f != "intercept") + races.gb).to_numpy(float)
