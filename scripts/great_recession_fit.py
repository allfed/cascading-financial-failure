from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from tqdm import tqdm
from sklearn.metrics import r2_score
from src.cascading_trade_network import AGDP, CascadingTradeNetwork
from src.reading import expected_gdp


def plot_country_by_country(
    y_true: dict[str, float],
    alpha_list: list[float],
    G: CascadingTradeNetwork,
    init_dict: dict[str, float],
) -> None:
    y_true = {k: v for k, v in y_true.items() if k not in init_dict and k in G}
    res = []
    for a in tqdm(alpha_list):
        y_pred = G.predict(
            [n for n in G if n not in init_dict and n in y_true], a, init_dict
        )
        res.extend(list(y_pred.items()))

    res = pd.DataFrame(np.array(res), columns=["country", "loss"])
    res["source"] = "model"
    res["loss"] = pd.to_numeric(res["loss"])

    y_true_df = pd.DataFrame(
        np.array(list(y_true.items())), columns=["country", "loss"]
    )
    y_true_df["source"] = "data"
    y_true_df["loss"] = pd.to_numeric(y_true_df["loss"])

    res = pd.concat([res, y_true_df]).reset_index(drop=True)
    assert isinstance(res, pd.DataFrame)
    _, ax = plt.subplots(tight_layout=True)
    sb.pointplot(
        res,
        x="country",
        y="loss",
        hue="source",
        order=res.loc[res["source"] == "data", :].sort_values("loss", ascending=False)[
            "country"
        ],
        log_scale=True,
        errorbar=("pi", 100),
        capsize=0.2,
        linestyle="none",
        marker="D",
        estimator="median",
        ax=ax,
    )
    ax.tick_params("x", labelrotation=60)
    y_upper = res[res["source"] == "model"].groupby("country").max()["loss"]
    y_lower = res[res["source"] == "model"].groupby("country").min()["loss"]
    y_true_df = res[res["source"] == "data"].groupby("country").first()["loss"]
    hit_rate = ((y_true_df >= y_lower) * (y_true_df <= y_upper)).mean()
    print("Great Recession hit rate: ", hit_rate)
    plt.tight_layout()
    plt.ylabel("Financial loss [$]")


def plot_true_v_pred(
    y_true: dict[str, float],
    alpha: float,
    G: CascadingTradeNetwork,
    init_dict: dict[str, float],
) -> None:
    _, ax = plt.subplots(tight_layout=True)
    y_true = {k: v for k, v in y_true.items() if k not in init_dict and k in G}
    y_pred = G.predict(
        [n for n in G if n not in init_dict and n in y_true], alpha, init_dict
    )
    xy = np.array([[y_true[c], y_pred[c]] for c in y_true])
    ax.loglog(xy[:, 0], xy[:, 1], "o")
    ax.loglog(xy[:, 0], xy[:, 0], "-")
    ax.set_xlabel("True loss [$]")
    ax.set_ylabel("Predicted loss [$]")
    print("Great Recession R^2:", r2_score(xy[:, 0], xy[:, 1]))


def main(timespan=4, model: Callable = AGDP, c_by_c=False):
    # 2007-2009 recession fit
    # SETUP
    G = model(
        trade_file="./data/trade/imf_cif_2007_import.xlsx",
        gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
        gdp_years=[2007],
    )
    gdps = expected_gdp(
        country_list=pd.read_csv("./data/country_list.csv"),
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

    # HYPERPARAMETERS
    tol = 1e-3
    upper_q = 0.75
    lower_q = 1 - upper_q

    minima = G.fit(
        log_loss,
        [lower_q, 0.5, upper_q],
        {"USA": USA_loss},
        tol,
        log=True,
        ret_score=True,
    )
    for q, a in minima.items():
        print(
            "q=",
            q,
            "->",
            "alpha=",
            np.round(a, -int(np.log10(tol))),
            f"+/- {tol}",
        )

    if c_by_c:
        plot_country_by_country(
            loss,
            [m[0] for m in minima.values()],
            G,
            {"USA": USA_loss},
        )
        plt.savefig("./results/2007_fit_c_by_c.png", bbox_inches="tight")
    else:
        plot_true_v_pred(
            loss,
            minima[0.5][0],
            G,
            {"USA": USA_loss},
        )
        plt.savefig("./results/2007_fit.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
