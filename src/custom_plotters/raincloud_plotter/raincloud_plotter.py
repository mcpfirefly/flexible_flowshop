import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt


def raincloud_plotter(data, filename, n_samples, generate_heuristic_schedules=None):
    fig, ax = plt.subplots(1, len(data.columns))

    filename_rainplot_arrays = "{}/{}_rainplot.csv".format(
        filename, generate_heuristic_schedules
    )
    data.to_csv(filename_rainplot_arrays, index=False)

    for i, column in enumerate(data.columns):
        # Generate the raincloud plot
        pt.RainCloud(data=data[column], ax=ax[i], orient="h", palette="Set2", alpha=0.7)

        ax[i].set_title(column)

    fig.set_size_inches(16, 6)
    fig.subplots_adjust(wspace=0.5)

    font = {"size": 12}
    plt.rc("font", **font)

    if generate_heuristic_schedules is not None:
        plt.suptitle(
            "Raincloud Plot - Experiment with Policy: {}, Samples = {}".format(
                generate_heuristic_schedules, n_samples
            ),
            fontsize=15,
            y=0.945,
        )
    else:
        plt.suptitle(
            "Raincloud Plot, Samples = {}".format(n_samples), fontsize=15, y=0.945
        )

    plt.savefig(
        "{}/{}_Raincloud.svg".format(filename, generate_heuristic_schedules), dpi=400
    )
    plt.close()
