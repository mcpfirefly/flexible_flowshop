
import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt
plt.style.use(['science','no-latex','grid'])



def raincloud_plotter(filename,data, n_samples, generate_heuristic_schedules=None):
    fig, ax = plt.subplots(len(data.columns),1)

    for i, column in enumerate(data.columns):
        # Generate the raincloud plot
        pt.RainCloud(width_viol =0.9,width_box = 0.1,point_size=2,data=data[column], ax=ax[i], orient="h", palette="Set2", alpha=0.3)

        key = column

        if key == "MAKESPAN":
            key = "Makespan [h]"
        elif key == "OCC":
            key = "OCC [$]"
        elif key == "WL":
            key = "Total WL [h]"



        ax[i].set_ylabel(key)
        #ax[i].set_title(column)

    fig.set_size_inches(8, 6)
    fig.subplots_adjust(wspace=0.2)

    plt.savefig(
        "{}/{}_raincloud.png".format(filename, generate_heuristic_schedules), dpi=400
    )
    plt.close()


if __name__ == "__main__":
    filename = r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\plots"
    filename_data = r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\03_HEURISTICS\ID35\FIFO_rainplot.csv"
    data = pd.read_csv(filename_data)
    n_samples = len(data)+1
    raincloud_plotter(filename,data,n_samples,"FIFO")