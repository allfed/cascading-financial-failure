import logging

import country_converter as coco
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import linregress

coco.logging.getLogger().setLevel(logging.CRITICAL)


def read_trade(
    country_list: pd.DataFrame,
    file="./data/trade/imf_cif_2008_import.xlsx",
    kwargs={"skiprows": 1, "index_col": 0},
) -> pd.DataFrame:
    """
    Read trading data and format them into a pandas.DataFrame.
    NOTE:South Sudan and Taiwan are not included in the IMF data set.

    Arguments:
        country_list (pandas.DataFrame): DataFrame containing country names and iso3 codes.
            Must have "iso3" column.
        file (str): trading data file path.
        kwargs (dict): key-word arguments for the pandas.read_excel function.

    Returns:
        pandas.Dataframe: square matrix containing trading data, index/cols are
            labelled by iso3 country codes.
    """
    trade = pd.read_excel(file, **kwargs)
    trade = trade.loc[
        trade.index.intersection(trade.columns), trade.index.intersection(trade.columns)
    ]
    trade.index = coco.convert(trade.index, to="iso3")
    trade.columns = coco.convert(trade.columns, to="iso3")
    trade = (
        trade.loc[
            trade.index.isin(country_list["iso3"]),
            trade.columns.isin(country_list["iso3"]),
        ].fillna(0)
        * 1e6
    )  # raw data is in millions
    return trade


def _melt_world_bank_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a typical World Bank sourced data into long form table.

    Arguments:
        raw_data (pandas.DataFrame): a DataFrame containing the data.
            Should contain a single data series.
            E.g., only `GDP`, or only `% access to electricity`, not both.

    Returns:
        pandas.DataFrame: long format DataFrame with `iso3`, `year`, `value` columns.
    """
    data = raw_data.drop(columns=["Series Name", "Series Code", "Country Name"])
    data = (
        data.melt(id_vars=["Country Code"])
        .replace(r".\[(.*?)\]", "", regex=True)
        .replace("..", np.nan)
    )
    data.columns = ["iso3", "year", "value"]
    data[["year", "value"]] = data[["year", "value"]].apply(pd.to_numeric)
    return data


def read_gdp(
    country_list: pd.DataFrame,
    years: list[int] = [],
    file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
) -> pd.DataFrame:
    """
    Read GDP data from the World Bank and format it into a pandas.DataFrame.
    Works on both GDP and GDP growth rate data.
    NOTE:TWN is not listed in World Bank data.
    https://datahelpdesk.worldbank.org/knowledgebase/articles/114933-where-are-your-data-on-taiwan

    Arguments:
        country_list (pandas.DataFrame): DataFrame containing country names and iso3 codes.
            Must have "iso3" column.
        year (list[int]): list of years to include. Defaults to an empty list
            which loads all available years.
        file (str): GDP data file path.

    Returns:
        pandas.DataFrame: a DataFrame containing GDP data in a long table format.
    """
    gdp = pd.read_csv(file)
    gdp = gdp.loc[gdp["Country Code"].isin(country_list["iso3"]), :]
    gdp = _melt_world_bank_data(gdp)
    if years:
        gdp = gdp[gdp["year"].isin(years)]
    gdp = gdp.dropna().reset_index(drop=True)
    assert isinstance(gdp, pd.DataFrame)
    return gdp


def _adjust_values_for_inflation(
    values: NDArray[np.float64], inflation_vector: NDArray[np.float64], backwards=False
) -> NDArray[np.float64]:
    """
    Adjust a vector of values with a vector of inflation rates.

    Arguments:
        values (NDArray[float]): 1-D array of values to be adjusted.
        inflation_vector (NDArray[float]): 1-D array of inflation rates,
            as fractions, not percentages.
        backwards (bool): whether we're adjusting backwards or forwards in time.

    Returns:
        NDArray[float]: a 1-D array of adjusted values.
    """
    assert inflation_vector.ndim == 1
    assert inflation_vector.dtype == float
    assert values.ndim == 1
    assert values.dtype == float
    if backwards:
        return values * np.prod(1 / (1 + inflation_vector))
    return values * np.prod(1 + inflation_vector)


def _adjust_group_for_inflation(
    group: pd.DataFrame,
    inflation_data: pd.Series,
    source_year: int,
    target_year: int | None,
) -> pd.DataFrame:
    """
    Adjust values in a pandas group (DataFrame),
    representing data in a single year, for inflation.

    Arguments:
        group (pandas.DataFrame): a DataFrame (from `df.groupby(by="year"`) with
            columns: `iso3` and `value`.
        inflation_data (pandas.Series): yearly inflation rates.
        source_year (int): the year from which the currency value is assumed in data.
        target_year (int | None): the year from which we take the currency value
            to convert to. If `None`, this year is taken from the group object.

    Returns:
        pandas.DataFrame: group object with the values adjusted.
    """
    assert group.columns.to_list() == ["iso3", "value"]
    assert isinstance(group.name, int)
    target_year = target_year or group.name
    inflation_vector = (
        inflation_data.loc[range(*sorted([source_year + 1, target_year + 1]))] / 100.0
    ).to_numpy()
    group["value"] = _adjust_values_for_inflation(
        group["value"].to_numpy(),
        inflation_vector=inflation_vector,
        backwards=source_year > target_year,
    )
    return group


def adjust_for_inflation(
    data: pd.DataFrame,
    inflation_file="./data/inflation.csv",
    source_year=2015,
    target_year: int | None = None,
) -> pd.DataFrame:
    """
    Adjust data values for inflation.
    Assumes the currency to be USD.

    Arguments:
        data (pandas.DataFrame): data to be adjusted, assumed to be in
            long format with columms `iso3`, `year`, `value`.
        inflation_file (str): path to inflation data file.
        source_year (int): the year from which the USD value is assumed in data.
        target_year (int | None): the year from which we take the USD value
            to convert to. If `None`, this year will be taken from the `year` column.

    Returns:
        pandas.DataFrame: data adjusted for inflation.
    """
    assert source_year != target_year, "The years provided are the same year."
    inflation = pd.read_csv(inflation_file)
    inflation = _melt_world_bank_data(inflation).dropna().reset_index(drop=True)
    inflation = (
        inflation[inflation["iso3"] == "USA"]
        .drop(columns="iso3")
        .set_index("year")
        .squeeze()
    )
    data["value"] = data.groupby(by="year", group_keys=False).apply(
        _adjust_group_for_inflation,
        inflation_data=inflation,
        source_year=source_year,
        target_year=target_year,
        include_groups=False,
    )["value"]
    return data


def load_data(
    country_list_file="./data/country_list.csv",
    trade_file="./data/trade/imf_cif_2008_import.xlsx",
    gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
    inflation_file="./data/inflation.csv",
    gdp_years=[2008],
    USD_value_year=2015,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load trade and GDP data into a pandas DataFrame for only the countries
    that are present in all data sets.

    Arguments:
        country_list_file (str): path to file containing the list of countries.
        trade_file (str): path to file containing trading data.
        gdp_file (str): path to file containing yearly GDP.
        inflation_file (str): path to file containing yearly inflation rates.
        gdp_years (list[int]): list of years to get the GDP data for.
        USD_value_year (int): year from which the USD value is to be assumed for
            the GDP data.

    Returns:
        tuple[pandas.Dataframe, pandas.DataFrame]:
            an NxN trade matrix, (N*gdp_years)x3 long format GDP table
    """
    country_list = pd.read_csv(country_list_file)
    trade = read_trade(country_list, trade_file)
    gdp = read_gdp(country_list, gdp_years, gdp_file)
    gdp = adjust_for_inflation(gdp, inflation_file, USD_value_year)
    common_countries = trade.index.intersection(gdp["iso3"]).to_list()
    trade = trade.loc[common_countries, common_countries]
    gdp = gdp.loc[gdp["iso3"].isin(common_countries), :]
    missing_countries = set(country_list["iso3"]).difference(common_countries)
    if missing_countries:
        print(
            """reading.load_data() WARNING:
These countries shan't be included as they are not present in all data sets:"""
        )
        print(missing_countries)
    return trade, gdp


def expected_gdp(
    country_list: pd.DataFrame,
    gdp_file="./data/gdp/gdp_1960_2023_in_2015USD.csv",
    inflation_file="./data/inflation.csv",
    inflation_source_year=2015,
    extrapolate_from: list[int] = [2006, 2007],
    extrapolate_to: int = 2009,
) -> dict[str, tuple[float, float]]:
    """
    Find the expected country GDP using linear extrapolation.

    Arguments:
        country_list (pandas.DataFrame): DataFrame containing country names and iso3 codes.
            Must have "iso3" column.
        gdp_file (str): path to file containing yearly GDP.
        inflation_file (str): path to file containing yearly inflation rates.
            Inflation adjustment is done to the last year year in `extrapolate_from`.
        inflation_source_year (int): year in which the `data` values are.
        extrapolate_from (list[int]): list of years to which fit the linear model.
        extrapolate_to (int): target year to predict with the linear model.

    Returns:
        dict[str, tuple[float, float]]: mapping of country to its expected,
            and actual GDP value.
    """
    assert isinstance(extrapolate_from, list)
    assert len(extrapolate_from) >= 2
    assert isinstance(extrapolate_to, int)
    assert extrapolate_to > min(extrapolate_from)
    assert sorted(extrapolate_from) == extrapolate_from
    gdp = read_gdp(
        country_list,
        extrapolate_from + [extrapolate_to],
        gdp_file,
    )
    gdp = adjust_for_inflation(
        gdp, inflation_file, inflation_source_year, extrapolate_from[-1]
    )
    gdps_dict = {}
    for c, country_data in gdp.groupby("iso3"):
        country_data = country_data.sort_values(by="year")[["year", "value"]].to_numpy()
        if country_data.shape != (len(extrapolate_from) + 1, 2):
            print("""reading.extrapolate_gdp() WARNING:""")
            print(c, "doesn't have sufficient data to extrapolate GDP; skipping.")
            continue
        fit = linregress(country_data[:-1, 0], country_data[:-1, 1])
        expected_gdp = fit.slope * country_data[-1, 0] + fit.intercept
        gdps_dict[c] = (expected_gdp, country_data[-1, 1])
    return gdps_dict
