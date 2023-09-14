import random
import string
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import Line2D
plt.style.use(['science','no-latex','grid'])

#### https://stackoverflow.com/questions/40813813/how-to-annotate-boxplot-median-quartiles-and-whiskers
#### https://python-graph-gallery.com/38-show-number-of-observation-on-boxplot/
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


def BoxPlotPlotter(data,n_samples,generate_heuristic_schedules):
    fig, ax = plt.subplots(1, len(data.keys()))

    medianprops = dict(linewidth=2.5, color="green")
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
            showfliers=False
        )
        AddBoxPlotValues(BoxPlot, ax[i])

        if key == "MAKESPAN":
            key = "Makespan [h]"
        elif key == "OCC":
            key = "OCC [$]"
        elif key == "WL":
            key = "Total WL [h]"

        #exp_id = Line2D([], [], label="ID35", color="blue")
        ax[i].set_xlabel(key)
        #ax[i].legend(handles=[exp_id])

        # Orange legend line
        # loutliers = Line2D([], [], color='white',markeredgecolor="black", markerfacecolor="white", marker='o', label='Outliers', markersize=12)
        # Green legend triangle
        #green_triangle = Line2D([], [], color='green', marker='^', linestyle='None', markersize=14, label='Mean')
        #Add legend shapes to legend handle
    lmedian = Line2D([], [], label="Median",color="green")
    lmean = Line2D([], [],label="Mean",color="orange")
    fig.set_size_inches(8, 4)
    plt.legend(handles=[lmedian, lmean], loc="lower right", ncol=1, bbox_to_anchor=[1.65, 0.75])
    plt.tight_layout()
    fig.subplots_adjust(hspace=2)
    #font = {"size": 12}
    #matplotlib.rc("font", **font)

    #if generate_heuristic_schedules != None:
        #plt.suptitle(
        #    "Box Plot - Experiment with Policy: {}, Samples = {}".format(
        #        generate_heuristic_schedules, n_samples
        #    ),
        #    fontsize=15,
        #    y=0.945,
        #)
    #else:
    #    plt.suptitle("Box Plot , Samples = {}", format(n_samples), fontsize=15, y=0.945)

    plt.savefig(
        "{}/{}_Boxplot.png".format(filename, generate_heuristic_schedules), dpi=400
    )
    plt.close()


if __name__ == "__main__":
    filename = r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\plots"
    filename_data = r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\03_HEURISTICS\ID38\SCT_rainplot.csv"
    data = pd.read_csv(filename_data)
    n_samples = len(data)+1
    BoxPlotPlotter(data,n_samples,"SCT")