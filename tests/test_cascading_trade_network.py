import sys

sys.path.append("./src")
import re
from random import sample

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cascading_trade_network import AGDP, ATV, DGDP, DTV, CascadingTradeNetwork
from reading import expected_gdp, load_data
from loss_transfer import beta


@pytest.fixture
def default_models() -> tuple[CascadingTradeNetwork, ...]:
    return AGDP(), ATV(), DGDP(), DTV()


@pytest.fixture
def default_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    trade, gdp = load_data()
    trade = trade.T  # convert A<-B to A->B
    gdp = gdp.set_index("iso3", drop=True)
    return trade, gdp


@pytest.fixture
def custom_models(default_models) -> tuple[CascadingTradeNetwork, ...]:
    r"""
       0
      / \
     1   2
      \ /
       3
       |
       4
    """
    # override defult models with a custom dataset
    for model in default_models:
        nodes = list(model.nodes)
        model.remove_nodes_from(nodes)
        model.add_weighted_edges_from(
            [
                (0, 1, 0.5),
                (0, 2, 0.5),
                (1, 3, 0.5),
                (2, 3, 0.5),
                (3, 4, 1 / 3),
                (1, 0, 0.5),
                (2, 0, 0.5),
                (3, 1, 1 / 3),
                (3, 2, 1 / 3),
                (4, 3, 1.0),
            ],
            weight="weight",
        )
        model.add_weighted_edges_from(
            [
                (0, 1, 0.5),
                (0, 2, 0.5),
                (1, 3, 0.5),
                (2, 3, 0.5),
                (3, 4, 1 / 3),
                (1, 0, 0.5),
                (2, 0, 0.5),
                (3, 1, 1 / 3),
                (3, 2, 1 / 3),
                (4, 3, 1.0),
            ],
            weight="_reduced_weight",
        )
        nx.set_node_attributes(model, {n: 1 for n in model}, name="capacity")
        nx.set_node_attributes(model, {n: 1 for n in model}, name="_reduced_capacity")
        nx.set_node_attributes(model, {n: False for n in model}, name="is_hit")
    return default_models


def test_init(default_models, default_data):
    trade_data, gdp_data = default_data
    for model in default_models:
        assert len(model) > 0
        assert np.sum(nx.adjacency_matrix(model)) > 0
        gdp = nx.get_node_attributes(model, "GDP")
        assert len(gdp) == len(model)
        for country, gdp in gdp.items():
            assert isinstance(gdp, float)
            assert gdp > 0
            assert gdp < np.inf
            assert gdp == pytest.approx(gdp_data.loc[country, "value"])
        is_hit = nx.get_node_attributes(model, "is_hit")
        assert len(is_hit) == len(model)
        assert sum(is_hit.values()) == 0
        for edge, value in nx.get_edge_attributes(model, "weight").items():
            assert value == pytest.approx(trade_data.loc[*edge])
        cap = nx.get_node_attributes(model, "capacity")
        assert len(cap) == len(model)
        for country, capacity in cap.items():
            assert isinstance(capacity, float)
            assert capacity > 0
            assert capacity < np.inf
            assert capacity == pytest.approx(
                gdp_data.loc[country, "value"]
                + np.abs(
                    trade_data.loc[:, country].sum() - trade_data.loc[country, :].sum()
                )
            )


def test_total_trade(default_models, default_data):
    trade_data, _ = default_data
    for model in default_models:
        country, other_country = sample(sorted(model), 2)
        assert model._total_trade(country, other_country) == pytest.approx(
            trade_data.loc[country, other_country]
            + trade_data.loc[other_country, country]
        )


def test_initialise_cascade(default_models):
    for model in default_models:
        init_d = dict(zip(sample(sorted(model), 2), np.random.random(2)))
        model._initialise_cascade(init_d)
        is_hit = nx.get_node_attributes(model, "is_hit")
        assert sum(is_hit.values()) == 2
        for country in model:
            v = init_d[country] if country in init_d else 0
            assert model.nodes[country]["_reduced_capacity"] == pytest.approx(
                model.nodes[country]["capacity"] * (1 - v)
            )
    for model in default_models:
        with pytest.raises(
            AssertionError,
            match=re.escape("at least one initial node must be provided"),
        ):
            model._initialise_cascade({})
        with pytest.raises(
            AssertionError, match=re.escape("initial impact must be in [0, 1]")
        ):
            model._initialise_cascade({"USA": 1.5})


@pytest.mark.parametrize(
    ["alpha", "init_loss"], [(1.5, 0.1), (1 + np.random.random(), np.random.random())]
)
def test_predict(custom_models, alpha, init_loss):
    transfer_func = beta()[0]
    node_1_reduced_cap = 1 - transfer_func(alpha, init_loss)
    impact_1_3 = transfer_func(alpha, 1 - node_1_reduced_cap / 1)
    # 2*impact_1_3 because 1 and 2 are hitting 3
    node_3_reduced_cap = 1 - 2 * impact_1_3 * (0.5 + 1 / 3)
    impact_3_1 = transfer_func(alpha, 1 - node_3_reduced_cap / 1)
    reduced_1_3_trade = (0.5 + 1 / 3) - (0.5 + 1 / 3) * impact_1_3
    node_1_reduced_cap = node_1_reduced_cap - impact_3_1 * reduced_1_3_trade
    for model in custom_models:
        prediction = list(model.predict([1], alpha, {0: init_loss}).values())[0]
        assert prediction == pytest.approx(1 - node_1_reduced_cap)


def test_fit():
    for Model in [AGDP, DGDP, DTV, ATV]:
        G = Model(
            trade_file="./data/trade/imf_cif_2007_import.xlsx",
            gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
            gdp_years=[2007],
        )
        gdps = expected_gdp(
            pd.read_csv("./data/country_list.csv"),
            extrapolate_from=[2006, 2007],
            extrapolate_to=2009,
        )
        # make sure country lists are identical
        gdps = {n: gdps[n] for n in gdps if n in G}
        G.remove_nodes_from([n for n in G if n not in gdps])
        assert len(G) == len(gdps)

        USA_loss = 1 - gdps["USA"][1] / gdps["USA"][0]

        loss = dict(
            zip(
                [n for n in gdps if gdps[n][0] > gdps[n][1]],
                [gdps[n][0] - gdps[n][1] for n in gdps if gdps[n][0] > gdps[n][1]],
            )
        )
        log_loss = dict(zip(loss.keys(), np.log(list(loss.values()))))

        tol = 1e-3
        upper_q = 0.75
        lower_q = 0.25

        minima = G.fit(
            log_loss, [lower_q, 0.5, upper_q], {"USA": USA_loss}, tol, log=True
        )
        assert isinstance(minima, dict)
        assert len(minima) == 3
        assert all([isinstance(v[0], float) for v in minima.values()])
        assert all(
            [v[0] > 1 and v[0] < np.inf and not np.isnan(v[0]) for v in minima.values()]
        )
        minima = G.fit(
            log_loss,
            [lower_q, 0.5, upper_q],
            {"USA": USA_loss},
            tol,
            log=True,
            ret_score=True,
        )
        assert isinstance(minima, dict)
        assert len(minima) == 3
        assert all([isinstance(v[0], float) for v in minima.values()])
        assert all(
            [v[0] > 1 and v[0] < np.inf and not np.isnan(v[0]) for v in minima.values()]
        )
        assert all([isinstance(v[1], float) for v in minima.values()])
        assert all(
            [
                v[1] > -np.inf and v[1] < np.inf and not np.isnan(v[1])
                for v in minima.values()
            ]
        )


def test_order_neighbours(custom_models):
    for model in custom_models:
        model.nodes[1]["capacity"] += 10
        trade_volume = {n: model._total_trade(3, n) for n in model.neighbours(3)}
        order = model._order_neighbours(trade_volume)
        match model.__class__.__name__:
            case "DTV":
                assert order[0] == 4
            case "ATV":
                assert order[-1] == 4
            case "DGDP":
                assert order[0] == 1
            case "AGDP":
                assert order[-1] == 1
