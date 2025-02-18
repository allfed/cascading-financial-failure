import great_recession_fit
import india_pakistan
import india_pakistan_specific_map
import matplotlib.pyplot as plt
import other_scenarios
import compare_models
import india_pakistan_pct_global
from src.cascading_trade_network import AGDP

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)

timespan = 4
model = AGDP

india_pakistan_specific_map.main(timespan=timespan, model=model, relative=False)
india_pakistan_specific_map.main(timespan=timespan, model=model, relative=True)
plt.rcParams["figure.figsize"] = (12, 4)
great_recession_fit.main(timespan=timespan, model=model, c_by_c=True)
plt.rcParams["figure.figsize"] = (6, 4)
great_recession_fit.main(timespan=timespan, model=model, c_by_c=False)
india_pakistan.main(timespan=timespan, model=model)
india_pakistan_pct_global.main(timespan=timespan, model=model)
plt.rcParams["figure.figsize"] = (10, 6.5)
other_scenarios.main(timespan=timespan, model=model)
plt.rcParams["figure.figsize"] = (12, 12)
compare_models.main()
