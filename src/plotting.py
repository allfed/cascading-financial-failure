import geopandas as gpd
from matplotlib.axes import Axes


def _plot_winkel_tripel_map_border(ax: Axes) -> None:
    """
    Plot a Winkel Tripel map border on the specified matplotlib.Axes object.

    Arguments:
        ax (matplotlib.axes.Axes): axis on which to plot.

    Returns:
        None.
    """
    border_geojson = gpd.read_file(
        "https://raw.githubusercontent.com/ALLFED/ALLFED-map-border/main/border.geojson"
    )
    border_geojson.plot(ax=ax, edgecolor="black", linewidth=0.1, facecolor="none")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def _prepare_world(
    world_map_file="./data/map/ne_110m_admin_0_countries.shp",
) -> gpd.GeoDataFrame:
    """
    Prepare the geospatial Natural Earth (NE) data (to be presumebly used in plotting).

    Arguments:
        world_map_file (str): path to file containing NE data

    Returns:
        geopandas.GeoDataFrame: NE data projected to Winkel Tripel.
    """
    # get the world map
    world = gpd.read_file(world_map_file)
    world = world.to_crs("+proj=wintri")
    return world


def plot_metric_map(
    ax: Axes,
    metric: dict[str, float],
    metric_name: str,
    shrink=1.0,
    world_map_file="./data/map/ne_110m_admin_0_countries.shp",
    **kwargs,
) -> Axes:
    """
    Plot world map with countries coloured by the specified metric.

    Arguments:
        ax (matplotlib.axes.Axes): axis on which to plot.
        metric (dict): dictionary containing the mapping: contry iso3 code -> value
        metric_name (str): the name of the metric
        shrink (float, optional): colour bar shrink parameter
        world_map_file (str): path to file containing NE data
        **kwargs (optional): any additional keyworded arguments recognised
            by geopandas plot function.

    Returns:
        matplotlib.axes.Axes: the Axes object containing the plot.
    """
    world = _prepare_world(world_map_file)

    # Join the country_community dictionary to the world dataframe
    world[metric_name] = world["ADM0_A3"].map(metric)

    world.plot(
        ax=ax,
        column=metric_name,
        missing_kwds={"color": "lightgrey"},
        legend=True,
        legend_kwds={
            "label": metric_name,
            "shrink": shrink,
        },
        **kwargs,
    )

    _plot_winkel_tripel_map_border(ax)

    # Add a title with self.scenario_name if applicable
    ax.set_title(f"{metric_name}")
    return ax
