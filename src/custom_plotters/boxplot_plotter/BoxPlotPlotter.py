import random
import string
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D

#### https://stackoverflow.com/questions/40813813/how-to-annotate-boxplot-median-quartiles-and-whiskers
#### https://python-graph-gallery.com/38-show-number-of-observation-on-boxplot/


def get_x_tick_labels(df, grouped_by):
    tmp = df.groupby([grouped_by]).size()
    return ["{0}: {1}".format(k, v) for k, v in tmp.to_dict().items()]


def series_values_as_dict(series_object):
    tmp = series_object.values()
    return [y for y in tmp][0]


def AddBoxPlotValues(bp, ax):
    """This actually adds the numbers to the various points of the boxplots"""
    for element in ["whiskers", "medians", "caps"]:
        for line in bp[element]:
            # Get the position of the element. y is the label you want
            (x_l, y), (x_r, _) = line.get_xydata()
            # Make sure datapoints exist
            # (I've been working with intervals, should not be problem for this case)
            if not np.isnan(y):
                x_line_center = x_l + (x_r - x_l) / 2 + 0.15
                y_line_center = y  # Since it's a line and it's horisontal
                # overlay the value:  on the line, from center to right
                ax.text(
                    x_line_center,
                    y_line_center,  # Position
                    "%.3f" % y,  # Value (3f = 3 decimal float)
                    verticalalignment="center",  # Centered vertically with line
                    backgroundcolor="white",
                )


def BoxPlotPlotter(data, filename, n_samples, generate_heuristic_schedules=None):
    fig, ax = plt.subplots(1, len(data.keys()))

    medianprops = dict(linestyle="-.", linewidth=2.5, color="firebrick")
    filename_boxplot_arrays = "{}/{}_boxplot.csv".format(
        filename, generate_heuristic_schedules
    )
    data.to_csv(filename_boxplot_arrays, index=False)

    for i, key in enumerate(data.keys()):
        BoxPlot = data.boxplot(
            key,
            ax=ax[i],
            return_type="dict",
            showmeans=True,
            meanline=True,
            medianprops=medianprops,
        )
        AddBoxPlotValues(BoxPlot, ax[i])

        # Orange legend line
        lmedian = Line2D([], [], color="#FF5722", label="Median", markersize=14)
        lmean = Line2D([], [], color="green", label="Mean", markersize=14)
        # loutliers = Line2D([], [], color='white',markeredgecolor="black", markerfacecolor="white", marker='o', label='Outliers', markersize=12)
        # Green legend triangle
        # green_triangle = Line2D([], [], color='green', marker='^', linestyle='None', markersize=14, label='Mean')
        # Add legend shapes to legend handle
        ax[i].legend(handles=[lmedian, lmean], loc="lower center", ncol=1, fontsize=12)
        ax[i].legend_.set_bbox_to_anchor([0.8, 0.75])

    fig.set_size_inches(16, 9)
    fig.subplots_adjust(hspace=2)

    font = {"size": 12}
    matplotlib.rc("font", **font)

    if generate_heuristic_schedules != None:
        plt.suptitle(
            "Box Plot - Experiment with Policy: {}, Samples = {}".format(
                generate_heuristic_schedules, n_samples
            ),
            fontsize=15,
            y=0.945,
        )
    else:
        plt.suptitle("Box Plot , Samples = {}", format(n_samples), fontsize=15, y=0.945)

    plt.savefig(
        "{}/{}_Boxplot.svg".format(filename, generate_heuristic_schedules), dpi=400
    )
    plt.close()
