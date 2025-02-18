# Cascading Financial Failure Model
The model's structure is a directed, weighted graph representing the world trade network, where edges represent import/export between countries (vertices), which in turn have an associated “capacity” – their gross domestic product (GDP) plus the absolute value of net exports. 

The model's dynamics are similar to the independent cascade model, although our model is deterministic, and we allow countries to hit back the source of their failure.
A country experiencing a failure, i.e., a fractional reduction of its capacity, transfers that failure onto its neighbours with whom it trades via a transfer function and reduces this trade as well.
Those neighbours then experience their own failure and transfer it onto their neighbours, and so on until all nodes have been affected.
The "echo" effect can only happen once between a pair of nodes and at reduced trade volume.

The purpose of the model is to provide a fast and interpretable estimate for of a cascading global financial failure in catastrophic scenarios, such as a nuclear conflict.

[A paper describing the model in detail can be found here.](https://arxiv.org/)

# Dependencies

A list of dependencies is in the [environment.yml](./environment.yml) and [requirements.txt](./requirements.txt) files.

We recommend using [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/) for setting up the virtual Python environment with the [environment.yml](./environment.yml) file.

To set up the environment with `conda`, run in the root directory (the one where this README.md resides):

```bash
conda create -f environment.yml
```

Then activate it with:
```bash
conda activate cff
```

Finally, install the model with:
```bash
python -m pip install -e .
```

Creating the environment with `pip` and related tools is also possible, using the [requirements.txt](./requirements.txt) file. 

The code was tested using Python 3.12 and 3.13.

# Repository structure and using the model

The primary files are in [./src/](./src/) folder, and the model itself is contained in [./src/cascading_trade_network.py](./src/cascading_trade_network.py).

Examples on how to use the model are in the [./scripts/](./scripts/) folder.
For a short and concrete example of how to use the model see the [jupyter notebook](./scripts/example.ipynb).
NOTE: the `./environment.yml` does *not* contain `jupyter` so in order to use the notebook you need to run
```bash
conda install jupyter
```
in the `cff` environment (see previous section).

All the data used for the model is publicly available and the sources are listed in the [./data/README.md](./data/README.md) file.

To reproduce the results from the paper, activate the appropriate Python environment (see previous section) and run (in this, i.e., root, directory):
```bash
python ./scripts/generate_plots.py
```

The folder structure is as follows:

```bash
.
├── data
│   ├── country_list.csv # country list taken from https://github.com/allfed/allfed-integrated-model
│   ├── gdp # GDP data from the World Bank
│   │   ├── gdp_1960_2023_in_2015USD.csv
│   │   └── gdp_1960_2023_in_2015USD_metadata.csv
│   ├── inflation.csv # CPI from the World Bank
│   ├── inflation_metadata.csv
│   ├── map # geographical data for plotting the World map
│   │   ├── ne_110m_admin_0_countries.cpg
│   │   ├── ne_110m_admin_0_countries.dbf
│   │   ├── ne_110m_admin_0_countries.prj
│   │   ├── ne_110m_admin_0_countries.README.html
│   │   ├── ne_110m_admin_0_countries.shp
│   │   ├── ne_110m_admin_0_countries.shx
│   │   └── ne_110m_admin_0_countries.VERSION.txt
│   ├── README.md # links to all data sources
│   ├── shares-of-gdp-by-economic-sector.csv # from Our World in Data
│   ├── shares-of-gdp-by-economic-sector.metadata.json
│   └── trade # trading data from the International Monetary Fund
│       ├── imf_cif_2007_import.xlsx
│       ├── imf_cif_2008_import.xlsx
│       ├── imf_cif_2010_import.xlsx
│       ├── imf_cif_2018_import.xlsx
│       └── imf_cif_2023_import.xlsx
├── environment.yml
├── LICENSE
├── README.md
├── requirements.txt
├── results # plots used in the research article
│   ├── 2007_fit_c_by_c.png
│   ├── 2007_fit.png
│   ├── 2007_fit_si.png
│   ├── india_pakistan_map_AGDP_pct.png
│   ├── india_pakistan_map_AGDP_usd.png
│   ├── india_pakistan_pct_global.png
│   ├── india_pakistan.png
│   ├── model_comparison_1.png
│   ├── model_comparison_2.png
│   ├── model_fit_other.png
│   ├── model_fit_other_si.png
│   ├── propagation_example.png
│   └── scores.csv
├── scripts # an example notebook and scripts used to generate results
│   ├── example.ipynb
│   ├── compare_models.py
│   ├── generate_plots.py
│   ├── great_recession_fit.py
│   ├── india_pakistan_pct_global.py
│   ├── india_pakistan.py
│   ├── india_pakistan_specific_map.py
│   ├── india_pakistan_specific.py
│   ├── other_scenarios.py
│   ├── plot_propagation.py
│   └── score_models.py
├── src # the main model files
│   ├── cascading_trade_network.py
│   ├── loss_transfer.py
│   ├── plotting.py
│   └── reading.py
└── tests
    ├── test_cascading_trade_network.py
    ├── test_loss_transfer.py
    └── test_reading.py
```
