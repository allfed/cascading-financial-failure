from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cascading_trade_network import AGDP
from src.reading import expected_gdp


def get_params(
    Model: Callable, upper_q=0.75, lower_q=0.25, tol=1e-3, timespan=2
) -> list[float]:
    # 2007-2009 recession fit
    # SETUP
    G = Model(
        trade_file="./data/trade/imf_cif_2007_import.xlsx",
        gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
        gdp_years=[2007],
    )
    gdps = expected_gdp(
        pd.read_csv("./data/country_list.csv"),
        extrapolate_from=list(range(2007 - timespan + 1, 2007 + 1)),
        extrapolate_to=2009,
    )
    # make sure country lists are identical
    gdps = {n: gdps[n] for n in gdps if n in G}
    G.remove_nodes_from([n for n in G if n not in gdps])
    assert len(G) == len(gdps)

    # INITIAL CONDITION
    USA_loss = 1 - gdps["USA"][1] / gdps["USA"][0]

    # Y_TRUE (compute (log-)losses)
    loss = dict(
        zip(
            [n for n in gdps if gdps[n][0] > gdps[n][1]],
            [gdps[n][0] - gdps[n][1] for n in gdps if gdps[n][0] > gdps[n][1]],
        )
    )
    log_loss = dict(zip(loss.keys(), np.log(list(loss.values()))))

    # FIND MINIMA (optimal control parameter values)
    minima = G.fit(log_loss, [lower_q, 0.5, upper_q], {"USA": USA_loss}, tol, log=True)
    return [a[0] for _, a in minima.items()]


def main(timespan=4, model=AGDP):
    res = []
    dq = 0.5
    # (industrial loss as fraction of GDP + fatalities x gdp per capita) / GDP
    # industrial loss % and # of fatalities are estimates of Blouin et al.
    india_loss = (3.55e12 * 1.5 / 100 * 25.0 / 100 + 33e6 * 2484.8) / 3.55e12
    pak_loss = (
        338.37e9 * 8 / 100 * 20.8 / 100 + 24e6 * 338.37e9 / 241_499_431
    ) / 338.37e9
    # other case, similar to Taiwan scenario
    # india_loss = 0.375
    # pak_loss = 0.375
    print("India init loss [%]:", india_loss * 100)
    print("Pakistan init loss [%]:", pak_loss * 100)
    # MODEL SETUP
    G = model(
        trade_file="./data/trade/imf_cif_2023_import.xlsx",
        gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
        gdp_years=[2023],
    )
    params = get_params(model, 1 - (1 - dq) / 2, (1 - dq) / 2, timespan=timespan)

    for a in tqdm(params):
        pred = G.predict(list(G.nodes()), a, {"IND": india_loss, "PAK": pak_loss})
        res.append(
            [
                model.__name__,
                a,
                sum(pred.values()),
            ]
        )

    res = pd.DataFrame(np.array(res), columns=["model", "alpha", "output"])
    print(res)


if __name__ == "__main__":
    main()
