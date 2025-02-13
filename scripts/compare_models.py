import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def plot_comparison(data, remove_bad_lin=True):
    if remove_bad_lin:
        bad_lin = data[((data["hit_rate_test"] == 0) & (data["dq"] == 0.9))][
            "neighbour order"
        ].unique()
        print(bad_lin)
        data = data[
            ~(
                (data["loss transfer"] == "linear")
                & (data["neighbour order"].isin(bad_lin))
            )
        ]
    data = data.sort_values(by=["timespan", "loss transfer", "neighbour order"])
    _, axs = plt.subplots(
        4,
        data["timespan"].unique().size,
        # tight_layout=True,
        sharex="col",
        sharey="row",
    )
    axs = axs.flatten()
    idx = 0
    for y, tl, yl in [
        ("interval_score_train", "Great Recession", "interval score"),
        ("hit_rate_train", "Great Recession", "hit rate"),
        ("interval_score_test", "Korean Conflict", "interval score"),
        ("hit_rate_test", "Korean Conflict", "hit rate"),
    ]:
        for ts in data["timespan"].unique():
            sb.lineplot(
                data[data["timespan"] == ts],
                x="dq",
                y=y,
                hue="loss transfer",
                style="neighbour order",
                ax=axs[idx],
                legend="auto" if idx == 0 else False,
            )
            axs[idx].set_xlabel("Δq")
            axs[idx].set_ylabel(f"{yl}")
            if idx in [0, 8]:
                axs[idx].set_title(f"{tl}; timespan={ts}")
            if idx in [1, 2, 3, 9, 10, 11]:
                axs[idx].set_title(f"timespan={ts}")
            idx += 1
    leg = axs[0].legend(
        loc="lower center",
        bbox_to_anchor=(2.2, -0.20),
        ncol=10,
        prop={"size": 9},
    )
    leg.set_zorder(20)


def main():
    data = pd.read_csv("results/scores.csv")
    data["loss transfer"] = data["loss_func"]
    data["neighbour order"] = data["model"]
    data = data.drop(columns=["loss_func", "model"])

    plot_comparison(data, False)
    plt.savefig("./results/model_comparison_1.png", bbox_inches="tight")
    plot_comparison(data, True)
    plt.savefig("./results/model_comparison_2.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
