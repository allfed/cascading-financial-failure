import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cascading_trade_network import AGDP, ATV, DTV, DGDP
from src.loss_transfer import beta, linear, quadratic
from src.reading import expected_gdp


def calculate_scores() -> list:
    scores = []
    for Model in tqdm([AGDP, ATV, DTV, DGDP], desc="model"):
        for loss_func in tqdm([beta, linear, quadratic], desc="loss func"):
            for timespan in tqdm([2, 4, 8, 16], desc="timespan"):
                # Bloomberg Korean scenario
                G_korea = Model(
                    trade_file="./data/trade/imf_cif_2018_import.xlsx",
                    gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
                    gdp_years=[2018],
                    loss_transfer=loss_func(),
                )
                # INITIAL CONDITIONS
                kor_local_rel_impact = [2.5 / 100, 37.5 / 100]
                # Y_TRUE (global cumulative impacts)
                kor_global_rel_impact = [0.5 / 100, 3.9 / 100]

                # 2007-2009 recession
                G_2007 = Model(
                    trade_file="./data/trade/imf_cif_2007_import.xlsx",
                    gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
                    gdp_years=[2007],
                    loss_transfer=loss_func(),
                )
                gdps = expected_gdp(
                    pd.read_csv("./data/country_list.csv"),
                    extrapolate_from=list(range(2007 - timespan + 1, 2007 + 1)),
                    extrapolate_to=2009,
                    gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
                )
                # make sure country lists are identical
                gdps = {n: gdps[n] for n in gdps if n in G_2007}
                G_2007.remove_nodes_from([n for n in G_2007 if n not in gdps])
                assert len(G_2007) == len(gdps)

                # INITIAL CONDITION
                USA_loss = 1 - gdps["USA"][1] / gdps["USA"][0]
                init_dict = {"USA": USA_loss}

                # Y_TRUE (compute (log-)losses)
                loss = dict(
                    zip(
                        [
                            n
                            for n in gdps
                            if gdps[n][0] > gdps[n][1] and n not in init_dict
                        ],
                        [
                            gdps[n][0] - gdps[n][1]
                            for n in gdps
                            if gdps[n][0] > gdps[n][1] and n not in init_dict
                        ],
                    )
                )
                log_loss = dict(zip(loss.keys(), np.log(list(loss.values()))))

                # HYPERPARAMETERS
                tol = 1e-3

                for dq in tqdm(np.linspace(0.1, 0.9, 50), leave=False):
                    res = []
                    lower_q = (1 - dq) / 2
                    upper_q = 1 - lower_q
                    minima = G_2007.fit(
                        log_loss,
                        [lower_q, upper_q],
                        init_dict,
                        tol,
                        log=True,
                    )
                    y_true = pd.DataFrame(
                        np.array(list(loss.items())), columns=["country", "loss"]
                    )
                    y_true["loss"] = pd.to_numeric(y_true["loss"])
                    y_true = y_true.set_index("country", drop=True).squeeze()
                    y_pred = []
                    for a in minima.values():
                        y_pred_d = G_2007.predict(
                            [n for n in G_2007 if n not in init_dict and n in y_true],
                            a[0],
                            init_dict,
                        )
                        y_pred.append(list(y_pred_d.items()))
                    y_pred = pd.DataFrame(
                        np.array(y_pred).reshape(-1, 2), columns=["country", "loss"]
                    )
                    y_pred["loss"] = pd.to_numeric(y_pred["loss"])
                    y_upper = y_pred.groupby("country").max()["loss"]
                    y_lower = y_pred.groupby("country").min()["loss"]
                    hit_rate = ((y_true >= y_lower) * (y_true <= y_upper)).mean()
                    interval_score = (
                        (y_upper - y_lower)
                        + (2 / dq)
                        * (
                            (y_lower - y_true) * (y_true < y_lower)
                            + (y_true - y_upper) * (y_true > y_upper)
                        )
                    ).mean()
                    res.extend(
                        [
                            Model.__name__,
                            loss_func.__name__,
                            timespan,
                            dq,
                            hit_rate,
                            interval_score,
                        ]
                    )

                    hit_rate = 0
                    interval_score = 0
                    for init, output in zip(
                        kor_local_rel_impact, kor_global_rel_impact
                    ):
                        output *= sum(
                            nx.get_node_attributes(G_korea, "capacity").values()
                        )
                        kor_y_pred = []
                        for _, a in minima.items():
                            a = a[0]
                            kor_pred = G_korea.predict(
                                list(G_korea.nodes()), a, {"KOR": init}
                            )
                            kor_pred["KOR"] = G_korea.nodes["KOR"]["capacity"] * (init)
                            kor_y_pred.append(sum(kor_pred.values()))
                        y_upper = max(kor_y_pred)
                        y_lower = min(kor_y_pred)
                        hit_rate += (y_lower <= output) * (y_upper >= output)
                        interval_score += (y_upper - y_lower) + (2 / dq) * (
                            (y_lower - output) * (output < y_lower)
                            + (output - y_upper) * (output > y_upper)
                        )
                    hit_rate /= len(kor_global_rel_impact)
                    interval_score /= len(kor_global_rel_impact)
                    res.extend([hit_rate, interval_score])
                    scores.append(res)
    return scores


def main():
    scores = calculate_scores()
    scores = pd.DataFrame(
        scores,
        columns=[
            "model",
            "loss_func",
            "timespan",
            "dq",
            "hit_rate_train",
            "interval_score_train",
            "hit_rate_test",
            "interval_score_test",
        ],
    )
    scores.to_csv("scores.csv", index=False)


if __name__ == "__main__":
    main()
