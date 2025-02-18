from abc import ABC, abstractmethod
from collections import deque
from operator import itemgetter

import networkx as nx
import numpy as np
from scipy.optimize import minimize_scalar

from src.reading import load_data
from src.loss_transfer import beta
from typing import Callable


class CascadingTradeNetwork(nx.DiGraph, ABC):
    """Directed trade graph with cascading financial failure dynamics.
    This class inherits from networkx.DiGraph and AbstractBaseClass.
    Therefore, it is meant as a template for model variants and not to
    be run by itself.

    Attributes:
        Nodes:
            capacity (float): country's GDP.
            is_hit (bool): whether the country was hit/visited or not.
        Edges:
            weight (float): trade volume (in the same units as capacity)
                from one country to another.

    The model used here presumes that one country's
    financial crisis propagates throughout the global trade network
    like a cascading failure with reverberation.
    That means that each country experiencing failure transfers that loss
    onto its neighbours, which causes them to experience a failure, which
    they transfer onto their neighbours and so on.
    The reverberation means that a country transferring its failure
    onto others gets hit back by the failure echo from its neighbours.
    The transfer of the loss from one country to another is provided
    as an argument of the class. This can be any function [0,1]->[0,1]
    with a single control parameter.
    The traversal of the cascade follows the breadth-first search approach,
    with the order of visiting neighbours dependent on the particular
    implementation of the model.
    """

    def __init__(
        self,
        country_list_file="./data/country_list.csv",
        trade_file="./data/trade/imf_cif_2008_import.xlsx",
        gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
        inflation_file="./data/inflation.csv",
        gdp_years=[2008],
        USD_value_year=2015,
        loss_transfer: tuple[Callable[[float, float], float], list[float]] = beta(),
    ):
        """
        Arguments:
            country_list_file (str): path to file containing the list
                of countries.
            trade_file (str): path to file containing trading data.
            gdp_file (str): path to file containing yearly GDP.
            inflation_file (str): path to file containing yearly
                inflation rates.
            gdp_years (list[int]): list of years to get the GDP data for.
                When multiple years are provided an *average* GDP value
                shall be used for the nodes' capacities.
            USD_value_year (int): year from which the USD value
                is to be assumed for the GDP data.
            loss_transfer (tuple): the loss transfer function and the
                appropriate parameter bounds for the optimisation.
                The function must be [0,1]->[0,1] and take a single
                control parameter.
                Examples are in `src/loss_transfer.py`.
        """
        trade, gdp = load_data(
            country_list_file=country_list_file,
            trade_file=trade_file,
            gdp_file=gdp_file,
            inflation_file=inflation_file,
            gdp_years=gdp_years,
            USD_value_year=USD_value_year,
        )
        gdp = (
            gdp.drop(columns=["year"])
            .set_index("iso3", drop=True)
            .groupby("iso3")
            .mean()
            .to_dict()["value"]
        )
        super().__init__(nx.from_pandas_adjacency(trade.T, create_using=nx.DiGraph))
        nx.set_node_attributes(self, gdp, name="GDP")
        total_imports = {
            node: sum([e[2] for e in self.in_edges(node, data="weight")])
            for node in self
        }
        total_exports = {
            node: sum([e[2] for e in self.out_edges(node, data="weight")])
            for node in self
        }
        abs_net_exports = {
            node: np.abs(total_exports[node] - total_imports[node]) for node in self
        }
        nx.set_node_attributes(
            self,
            {node: gdp[node] + abs_net_exports[node] for node in self},
            name="capacity",
        )
        nx.set_node_attributes(self, gdp, name="_reduced_capacity")
        nx.set_edge_attributes(
            self, nx.get_edge_attributes(self, "weight"), "_reduced_weight"
        )
        nx.set_node_attributes(self, {n: False for n in self}, name="is_hit")
        self.loss_transfer_func, self.loss_transfer_bounds = loss_transfer
        self.neighbours = self.neighbors

    def _total_trade(self, node: str, neighbour: str) -> float:
        """
        Compute total trade between a country and its neighbour.

        Arguments:
            node (str): country name.
            neighbour (str): neighbour country name.

        Returns:
            float: total (import+export) trade volume
                between the countries
        """
        try:
            imports = self.edges[node, neighbour]["_reduced_weight"]
        except KeyError:
            imports = 0
        try:
            exports = self.edges[neighbour, node]["_reduced_weight"]
        except KeyError:
            exports = 0
        return exports + imports

    def _reduce_trade(
        self, node: str, neighbour: str, relative_reduction: float
    ) -> None:
        """
        Reduce imports and exports between two countries.

        Arguments:
            node, neighbour (str): the two countries' names
            relative_reduction (float): fraction by which to reduce trade.

        Returns:
            None.
        """
        try:
            self.edges[node, neighbour]["_reduced_weight"] *= 1 - relative_reduction
        except KeyError:
            pass
        try:
            self.edges[neighbour, node]["_reduced_weight"] *= 1 - relative_reduction
        except KeyError:
            pass

    def _initialise_cascade(self, start_nodes: dict[str, float]) -> None:
        """
        Initialise cascading dynamics:
        `starting_nodes` get `is_hit` attribute set to True, others to False.
        `capacity`, `_reduced_capacity` and `_reduced_weight` attributes
        get reset.

        Arguments:
            start_nodes (dict[str]): mapping of nodes (country names) from which
                the cascades shall begin to their initial losses as fractions in [0,1].

        Returns:
            None.
        """
        assert len(start_nodes) > 0, "at least one initial node must be provided"
        assert all([0 <= si <= 1 for si in start_nodes.values()]), (
            "initial impact must be in [0, 1]"
        )
        nx.set_node_attributes(
            self,
            {node: True if node in start_nodes else False for node in self},
            name="is_hit",
        )
        _reduced_capacity = {
            node: cap - start_nodes[node] * cap if node in start_nodes else cap
            for node, cap in nx.get_node_attributes(self, "capacity").items()
        }
        nx.set_node_attributes(
            self,
            _reduced_capacity,
            name="_reduced_capacity",
        )
        nx.set_edge_attributes(
            self, nx.get_edge_attributes(self, "weight"), "_reduced_weight"
        )

    @abstractmethod
    def _order_neighbours(self, neighbour_trade_volume: dict[str, float]) -> list[str]:
        """
        Provide the ordering in which to hit neighbours.

        Arguments:
            neighbour_trade_volume (dict[str, float]): mapping of neighbours to
                total trade volume.

        Returns:
            list[str]: list of sorted neighbours
        """
        raise NotImplementedError

    def _run_cascade(self, alpha: float) -> None:
        """
        Simulate the cascading dynamics.

        Arguments:
            alpha (float): the control parameter.

        Returns:
            None.
        """
        assert isinstance(alpha, float)
        propagation_queue = deque(
            {
                node
                for node, is_hit in nx.get_node_attributes(self, name="is_hit").items()
                if is_hit
            }
        )
        assert len(propagation_queue) > 0
        failsafe_iter = len(self) ** 2
        while propagation_queue and failsafe_iter >= 0:
            failsafe_iter -= 1
            hit_node = propagation_queue.pop()
            impact = (
                1
                - self.nodes[hit_node]["_reduced_capacity"]
                / self.nodes[hit_node]["capacity"]
            )
            trade_volume = {
                n: self._total_trade(hit_node, n) for n in self.neighbours(hit_node)
            }
            hit_node_degree = self.degree(hit_node)
            assert isinstance(hit_node_degree, int)
            for neighbour, relative_reduction in zip(
                self._order_neighbours(trade_volume),
                [self.loss_transfer_func(alpha, impact)] * hit_node_degree,
            ):
                reduction = relative_reduction * trade_volume[neighbour]
                self.nodes[neighbour]["_reduced_capacity"] -= reduction
                self._reduce_trade(hit_node, neighbour, relative_reduction)
                if not self.nodes[neighbour]["is_hit"]:
                    self.nodes[neighbour]["is_hit"] = True
                    propagation_queue.appendleft(neighbour)

    def _simulate(self, start_nodes: dict[str, float], alpha: float) -> None:
        """
        Initialise and run the cascade model.

        Arguments:
            start_nodes (dict[str]): mapping of nodes (country names) from which
                the cascades shall begin to their initial losses.
            alpha (float): the control parameter.

        Returns:
            None.
        """
        self._initialise_cascade(start_nodes)
        self._run_cascade(alpha)

    def _loss_function(self, x: float, quantile: float) -> float:
        """
        Quantile loss function a.k.a pinball loss function.

        Arguments:
            x (float): the difference between the target value and prediction.
            quantile (float): the target quantile. Must be in [0, 1].

        Returns:
            float: the "loss" of the prediction. The lower the better.
        """
        return quantile * max(x, 0) + (1 - quantile) * max(-x, 0)

    def _fit_score(
        self,
        alpha: float,
        y_true: dict[str, float],
        start_nodes: dict[str, float],
        quantile=0.5,
        log=True,
    ) -> float:
        """
        Calculate the mean pinball loss value of the model prediction
        based on the given true values.

        Arguments:
            alpha (float): the model's control parameter.
            y_true (dict[str, float]): mapping of country to their real value loss.
            start_nodes (dict[str, float]): mapping of country to their initial value loss.
            quantile (float): target quantile to predict (must be in [0, 1]).
            log (bool): whether to consider the logarithm of the value loss (True)
                or not (False).

        Returns:
            float: model's prediction loss score, the lower the better.
        """
        self._simulate(start_nodes, alpha)
        y_pred_values = [
            self.nodes[n]["capacity"] - self.nodes[n]["_reduced_capacity"]
            for n in self
            if n not in start_nodes
        ]
        if log:
            y_pred_values = np.log(y_pred_values)
        y_pred = dict(
            zip(
                [n for n in self if n not in start_nodes],
                y_pred_values,
            )
        )
        countries = y_true.keys() & y_pred.keys()
        quantile_score = np.mean(
            [self._loss_function(y_true[n] - y_pred[n], quantile) for n in countries]
        )
        return float(quantile_score)

    def fit(
        self,
        y_true: dict[str, float],
        quantiles: list[float],
        start_nodes: dict[str, float],
        tol=1e-3,
        log=True,
        ret_score=False,
    ) -> dict[float, tuple[float, ...]]:
        """
        Fit the model to data, i.e., find the value of the control parameter
        that minimises prediction loss function for the specified quantile.

        Arguments:
            y_true (dict[str, float]): mapping of country to true value loss.
            quantiles (list[float]): which quantiles to find parameters for.
            start_nodes (dict[str, float]): initial conditions, mapping of country
                to their initial value loss.
            tol (float, optional): absolute tolerance level in the parameter
                estimation, default=1e-3.
            log (bool, optional): whether to consider the logarithm of value loss or not.
            ret_score (bool, optional): whether to return the prediction loss function value
                as well or not.

        Returns:
            dict[float, tuple[float, ...]]: mapping of quantile to its optimal parameter,
                or the optimal parameter and prediction loss function value.
                Either time as a tuple t, such that t[0] is the parameter value.
        """
        out = {
            q: minimize_scalar(
                self._fit_score,
                args=(y_true, start_nodes, q, log),
                method="bounded",
                bounds=self.loss_transfer_bounds,
                options={"xatol": tol},
            )
            for q in quantiles
        }
        if ret_score:
            return {q: (float(res.x), float(res.fun)) for q, res in out.items()}
        return {q: (float(res.x),) for q, res in out.items()}

    def predict(
        self, nodes: list[str], alpha: float, start_nodes: dict[str, float]
    ) -> dict[str, float]:
        """
        Predict value loss for the specified countries.

        Arguments:
            nodes (list[str]): list of countries for which to find the prediction.
            alpha (float): the control parameter.
            start_nodes (dict[str, float]): initial condition, mapping of country
                to initial value loss.

        Returns:
            dict[str, float]: mapping of country to the prediction of value loss.
        """
        assert len(nodes) > 0
        self._simulate(start_nodes, alpha)
        capacity = itemgetter(*nodes)(nx.get_node_attributes(self, "capacity"))
        _reduced_capacity = itemgetter(*nodes)(
            nx.get_node_attributes(self, "_reduced_capacity")
        )
        if len(nodes) == 1:
            capacity = (capacity,)
            _reduced_capacity = (_reduced_capacity,)
        return dict(
            zip(
                nodes,
                [float(x) for x in np.array(capacity) - np.array(_reduced_capacity)],
            )
        )


class DTV(CascadingTradeNetwork):
    """Descending trade volume"""

    def _order_neighbours(self, neighbour_trade_volume: dict[str, float]) -> list[str]:
        return sorted(
            neighbour_trade_volume,
            key=lambda k: neighbour_trade_volume[k],
            reverse=True,
        )


class ATV(CascadingTradeNetwork):
    """Ascending trade volume"""

    def _order_neighbours(self, neighbour_trade_volume: dict[str, float]) -> list[str]:
        return sorted(
            neighbour_trade_volume,
            key=lambda k: neighbour_trade_volume[k],
            reverse=False,
        )


class DGDP(CascadingTradeNetwork):
    """Descending GDP"""

    def _order_neighbours(self, neighbour_trade_volume: dict[str, float]) -> list[str]:
        return sorted(
            neighbour_trade_volume,
            key=lambda k: self.nodes[k]["capacity"],
            reverse=True,
        )


class AGDP(CascadingTradeNetwork):
    """Ascending GDP"""

    def _order_neighbours(self, neighbour_trade_volume: dict[str, float]) -> list[str]:
        return sorted(
            neighbour_trade_volume,
            key=lambda k: self.nodes[k]["capacity"],
            reverse=False,
        )
