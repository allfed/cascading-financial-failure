from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from tqdm import tqdm

from src.cascading_trade_network import AGDP, CascadingTradeNetwork
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
        gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
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


def get_setup(
    model: Callable, year_pre_boom: int, init_country: str, timespan: int
) -> tuple[CascadingTradeNetwork, dict[str, float], float]:
    # MODEL SETUP
    G = model(
        trade_file=f"./data/trade/imf_cif_{year_pre_boom}_import.xlsx",
        gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
        gdp_years=[year_pre_boom],
    )
    if "THA" in init_country:
        extrapolation_base = [2006, 2007, 2010]
    else:
        extrapolation_base = list(
            range(year_pre_boom - timespan + 1, year_pre_boom + 1)
        )
    gdps = expected_gdp(
        pd.read_csv("./data/country_list.csv"),
        extrapolate_from=extrapolation_base,
        extrapolate_to=year_pre_boom + 1,
        gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
    )
    # make sure country lists are identical
    gdps = {n: gdps[n] for n in gdps if n in G}
    G.remove_nodes_from([n for n in G if n not in gdps])
    assert len(G) == len(gdps)
    is_min = False
    if init_country == "THAmin":
        is_min = True
        init_country = "THA"
    if init_country == "KOR":
        local_rel_loss = 2.5 / 100
    elif init_country == "KORwar":
        local_rel_loss = 37.5 / 100
        init_country = "KOR"
    else:
        local_rel_loss = 1 - gdps[init_country][1] / gdps[init_country][0]
    init_dict = {init_country: local_rel_loss}
    if init_country == "UKR":
        rus_loss = 1 - gdps["RUS"][1] / gdps["RUS"][0]
        global_loss = 0.007 * sum([v[1] for v in gdps.values()])
        # 0.7% is from World Bank analysis: https://hdl.handle.net/10986/37359
        init_dict["RUS"] = rus_loss
    elif init_country == "THA":
        if is_min:
            # 18.4%, source: data/shares-of-gdp-by-economic-sector.csv
            # 2.5%, source: https://openknowledge.worldbank.org/entities/publication/a1282547-3df7-5d52-98b0-dfeb15a30443
            global_loss = 2.5 / 100 * 18.4 / 100 * sum([v[1] for v in gdps.values()])
        else:
            # 43.35%, source: data/shares-of-gdp-by-economic-sector.csv
            global_loss = 2.5 / 100 * 43.35 / 100 * sum([v[1] for v in gdps.values()])
            pass
    elif init_country == "KOR":
        # https://www.bloomberg.com/graphics/2024-korea-war-threatens-trillions-for-global-economy/
        if local_rel_loss > 0.3:  # war
            global_loss = 3.9 / 100 * sum([v[1] for v in gdps.values()])
        else:  # regime collapse
            global_loss = 0.5 / 100 * sum([v[1] for v in gdps.values()])
    else:
        global_loss = sum([gdps[n][0] - gdps[n][1] for n in gdps])
    return G, init_dict, global_loss


def calculate(timespan=4, model: Callable = AGDP):
    res = []
    dq = 0.5
    scenarios = {
        "2011 Thailand Floods_min": [2010, "THAmin"],
        "2011 Thailand Floods": [2010, "THA"],
        "Invasion of Ukraine": [2021, "UKR"],
        "Korean regime collapse": [2018, "KOR"],
        "Korean war": [2018, "KORwar"],
    }
    for scenario, (year, init_node) in tqdm(scenarios.items()):
        if "_min" in scenario:
            scenario = scenario.replace("_min", "")
        G, local_rel_impact, global_impact = get_setup(model, year, init_node, timespan)

        local_rel_impact = local_rel_impact
        params = get_params(model, 1 - (1 - dq) / 2, (1 - dq) / 2, timespan=timespan)

        for a in params:
            pred = G.predict(list(G.nodes()), a, local_rel_impact)
            for country in local_rel_impact:
                pred[scenario] = (
                    G.nodes[country]["capacity"] * local_rel_impact[country]
                )
            res.append(
                [
                    scenario,
                    "model",
                    a,
                    sum(pred.values()),
                ]
            )
        res.append(
            [
                scenario,
                "data",
                "N/A",
                global_impact,
            ]
        )

    res = pd.DataFrame(np.array(res), columns=["scenario", "source", "alpha", "output"])
    res["output"] = res["output"].apply(pd.to_numeric)
    _, ax = plt.subplots(tight_layout=True)
    sb.pointplot(
        res,
        x="scenario",
        y="output",
        hue="source",
        errorbar=("pi", 100),
        dodge=0.2,
        capsize=0.2,
        linestyle="none",
        marker="D",
        estimator="median",
        ax=ax,
    )
    plt.ylabel("Global loss [$]")
    ax.get_legend().set_title("")
    plt.savefig("./results/model_fit_other.png", bbox_inches="tight")


def main(timespan=4, model: Callable = AGDP):
    calculate(timespan, model)


if __name__ == "__main__":
    main()
