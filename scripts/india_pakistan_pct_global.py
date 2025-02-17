import sys

sys.path.append("./src")
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from tqdm import tqdm
import networkx as nx

from cascading_trade_network import AGDP
from reading import expected_gdp


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
    for reduction in [0, 0.5, 1.0]:
        # MODEL SETUP
        G = model(
            trade_file="./data/trade/imf_cif_2023_import.xlsx",
            gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
            gdp_years=[2023],
        )
        if reduction > 0:
            # redistribute India and Pakistan trade to other countries
            for e in G.edges():
                if "IND" in e or "PAK" in e:
                    try:
                        other_c = [c for c in e if c != "IND" and c != "PAK"][0]
                    except IndexError:
                        continue
                    w = G.edges[e]["weight"]
                    G.edges[e]["weight"] -= reduction * w
                    w = w / len(G.edges(other_c))
                    for other_e in G.edges(other_c):
                        if "IND" in other_e or "PAK" in other_e:
                            continue
                        G.edges[other_e]["weight"] += w

        params = get_params(model, 1 - (1 - dq) / 2, (1 - dq) / 2, timespan=timespan)

        for a in tqdm(params):
            for init_loss in np.linspace(0.0, 0.99, 50):
                pred = G.predict(
                    list(G.nodes()), a, {"IND": init_loss, "PAK": init_loss}
                )
                res.append(
                    [
                        init_loss
                        * (G.nodes["IND"]["GDP"] + G.nodes["PAK"]["GDP"])
                        / sum(nx.get_node_attributes(G, "GDP").values())
                        * 100,
                        str(int(reduction * 100)),
                        a,
                        (
                            sum(pred.values())
                            / sum(nx.get_node_attributes(G, "GDP").values())
                        )
                        * 100,
                    ]
                )

    res = pd.DataFrame(
        np.array(res), columns=["init_loss", "trade reduction [%]", "alpha", "output"]
    )
    res[["init_loss", "output"]] = res[["init_loss", "output"]].apply(pd.to_numeric)
    _, ax = plt.subplots(tight_layout=True)
    sb.lineplot(
        res,
        x="init_loss",
        y="output",
        hue="trade reduction [%]",
        errorbar=("pi", 100),
        estimator="median",
        ax=ax,
    )
    plt.xlabel("India, Pakistan initial loss [% global GDP]")
    plt.ylabel("Global loss [% global GDP]")
    plt.savefig("./results/india_pakistan_pct_global.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
