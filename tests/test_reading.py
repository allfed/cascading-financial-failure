import sys

sys.path.append("./src")

from functools import reduce

import cpi
import numpy as np
import pandas as pd
import pytest

from reading import (
    _adjust_values_for_inflation,
    adjust_for_inflation,
    expected_gdp,
    load_data,
    read_gdp,
    read_trade,
)


@pytest.fixture
def country_list() -> pd.DataFrame:
    return pd.read_csv("./data/country_list.csv")


def test_read_trade(country_list):
    tr = read_trade(country_list)
    assert isinstance(tr, pd.DataFrame)
    assert tr.shape[0] == tr.shape[1]
    assert tr.index.equals(tr.columns)
    assert (tr.dtypes == np.dtype("float64")).all()
    assert not tr.isnull().values.any()
    assert not (tr < 0).any(axis=None)
    assert tr.loc["ALB", "AFG"] == pytest.approx(24)


def test_read_gdp(country_list, file="./data/gdp/gdp_1960_2023_in_2015USD.csv"):
    gdp = read_gdp(country_list, file=file, years=[2008])
    assert isinstance(gdp, pd.DataFrame)
    assert gdp.shape[1] == 3
    for c in gdp["iso3"]:
        assert c in country_list["iso3"].values
    assert (gdp["year"] == 2008).all()
    assert gdp["value"].dtype == np.dtype("float64")
    assert gdp["year"].dtype == np.dtype("int64")
    assert gdp.loc[gdp["iso3"] == "BEL", "value"].values == pytest.approx(
        432_478_428.73e3
    )


def test_adjust_values_for_inflation():
    inflation_vector = np.random.random(10)
    values = np.random.random(10) * 1e3
    back = [
        reduce(lambda v, next_v: v / (1 + next_v), inflation_vector, v) for v in values
    ]
    forw = [
        reduce(lambda v, next_v: v * (1 + next_v), inflation_vector, v) for v in values
    ]
    assert _adjust_values_for_inflation(values, inflation_vector) == pytest.approx(forw)
    assert _adjust_values_for_inflation(
        values, inflation_vector, backwards=True
    ) == pytest.approx(back)


@pytest.mark.parametrize("test_country", ("USA", "BEL", "CHN"))
@pytest.mark.parametrize("target_year", (2008, 2020))
def test_adjust_for_inflation(country_list, test_country, target_year):
    gdp = read_gdp(
        country_list,
        years=[target_year],
        file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
    )
    v = gdp.loc[gdp["iso3"] == test_country, "value"].values[0]
    adj_v = cpi.inflate(v, 2015, target_year)
    gdp = adjust_for_inflation(gdp, source_year=2015, target_year=target_year)
    v = gdp.loc[gdp["iso3"] == test_country, "value"].values[0]
    assert v == pytest.approx(adj_v, rel=1e-5)


def test_load_data(country_list):
    data = load_data()
    assert isinstance(data, tuple)
    assert len(data) == 2
    for d in data:
        assert isinstance(d, pd.DataFrame)
    trade, gdp = data
    assert trade.shape[0] == trade.shape[1]
    assert trade.index.equals(trade.columns)
    assert gdp.shape[1] == 3
    assert sorted(gdp["iso3"].unique()) == sorted(trade.index)
    for c in gdp["iso3"].values:
        assert c in country_list["iso3"].values
    for c in trade.index:
        assert c in country_list["iso3"].values


@pytest.mark.parametrize("test_country", ("POL", "JPN", "ARG"))
def test_expected_gdp(country_list, test_country):
    e_gdp = expected_gdp(country_list, extrapolate_to=2008)
    assert isinstance(e_gdp, dict)
    for k, t_v in e_gdp.items():
        assert k in country_list["iso3"].values
        assert isinstance(t_v, tuple)
        assert all([isinstance(v, float) for v in t_v])
        assert all([v > 0 and v < np.inf and not np.isnan(v) for v in t_v])
    gdp = read_gdp(
        country_list,
        years=[2008],
        file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
    )
    assert e_gdp[test_country][1] == pytest.approx(
        cpi.inflate(gdp.loc[gdp["iso3"] == test_country, "value"], 2015, 2007), rel=1e-5
    )
    gdp = read_gdp(
        country_list,
        years=[2006, 2007],
        file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
    )
    gdp = adjust_for_inflation(gdp, "./data/inflation.csv", target_year=2007)
    vv = gdp.loc[gdp["iso3"] == test_country, "value"]
    next_year_value = 2 * vv.max() - vv.min()
    assert e_gdp[test_country][0] == pytest.approx(next_year_value)
