import matplotlib.pyplot as plt
import scienceplots

import pandas as pd
import numpy as np
import os
import re
from scipy import stats

from matplotlib import colors as mcolors
colors_css = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

plot_heuristics = True
colors_heuristics = ["mediumvioletred", "indianred", "olivedrab", "teal","springgreen","deepskyblue"]
several = True
plot_best = True
def get_experiment_id(csv_path):
    pattern = r'\\(ID\d+)'
    # Search for the pattern in the path
    match = re.search(pattern, csv_path)
    # Check if a match was found
    if match:
        extracted_string = match.group(1)  # Get the first matching group (ID01 in this case)
        print(extracted_string)
        return extracted_string

base = r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results"
output = base + r"\individual_plots"
output_svg = base + r"\individual_plots_svg"
os.makedirs(output, exist_ok=True)
os.makedirs(output_svg, exist_ok=True)
plt.style.use(['science','no-latex','grid'])
# Define your base source directory
window_size = 12
smoothing = False
output_directory = "plots"

if smoothing:
    smoothing_factor = 0.999
    window_size = None

loc_eval_files = []
loc_train_files = []
# Function to calculate moving average
def calculate_moving_average(data):
    return data.rolling(window=window_size, min_periods=1).mean(), data.rolling(window=window_size, min_periods=1).std()

def smooth_values(values):
    # Apply exponential moving average (EMA) smoothing to y-values
    values = np.array(values)
    smoothed_values = [values[0]]  # Initialize with the first element
    for y in values[1:]:
        smoothed_y = smoothing_factor * smoothed_values[-1] + (1 - smoothing_factor) * y
        smoothed_values.append(smoothed_y)
    return smoothed_values

def calculate_trendline(x,y):
    # calculate equation for trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p

def filter_outliers(df, column):
    df = df[(np.abs(stats.zscore(column)) < 3)]
    episodes = list(range(1, len(df) + 1))
    return df, episodes

# Function to capitalize the first letter of each word
def format_column(column):
    if column == 'oc_costs':
        return 'OCC'
    elif column == 'total_reward':
        return 'Return'
    elif column == "sim_duration":
        return 'Makespan'
    elif column == 'weighted_lateness':
        return 'Weighted Lateness'
    elif column == 'l':
        return 'Episode Length'
    return column

def generate_single_plot(csv_path, column_name, label,idx, is_evaluation):

    df = pd.read_csv(csv_path, header=1)
    df, episodes = filter_outliers(df, df[column_name])
    #episodes = list(range(1, len(df) + 1))
    if smoothing:
        df[column_name] = smooth_values(df[column_name])
        std_eval = np.std(smooth_values(df[column_name]))
    else:
        df[column_name], std_eval = calculate_moving_average(df[column_name])

    mean_eval = df[column_name]

    plt.plot(episodes,mean_eval, label=label,color=colors[idx])
    plt.fill_between(episodes, mean_eval - std_eval, mean_eval + std_eval, color=colors[idx], alpha=0.2)


    if column_name == "sim_duration":
        #plt.gca().set_ylim(bottom=25)
        if plot_heuristics:
            plt.plot(episodes, [35.74] *len(episodes), label="Best Solution PPO (ID23)", color=colors_css[colors_heuristics[4]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [42.18] *len(episodes), label="Best Solution SAC (ID16)", color=colors_css[colors_heuristics[5]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [44.01] *len(episodes), label="Best Solution FIFO", color=colors_css[colors_heuristics[0]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [34.25] *len(episodes), label="Best Solution SPT", color=colors_css[colors_heuristics[1]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [33.55] *len(episodes), label="Best Solution EDD", color=colors_css[colors_heuristics[2]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [33.07] *len(episodes), label="Best Solution SCT", color=colors_css[colors_heuristics[3]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [26.56]*len(episodes), label=f'Best Solution MIP (Kopanos)', color="black", linestyle="dashed",linewidth=1.2)
            plt.plot(episodes, [25.01] * len(episodes), label=f'Best Solution CP (Bleidorn)$^1$', color=colors_css["lightseagreen"],
                     linestyle="dashed", linewidth=1.2)
        else:
            plt.plot(episodes, [26.56] * len(mean_eval), label=f'Best Solution MIP (Kopanos)', color="black",
                     linestyle="dashed", linewidth=1.5)
            plt.plot(episodes, [25.01] * len(mean_eval), label=f'Best Solution CP (Bleidorn)$^1$',
                     color=colors_css["lightseagreen"],
                     linestyle="dashed", linewidth=1.2)

    elif column_name == "oc_costs":
        #plt.gca().set_ylim(bottom=60)
        if plot_heuristics:
            plt.plot(episodes, [82.01] *len(episodes), label="Best Solution PPO (ID23)", color=colors_css[colors_heuristics[4]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [88.89] *len(episodes), label="Best Solution SAC (ID19)", color=colors_css[colors_heuristics[5]], linestyle="dashed", linewidth=1.2)

            plt.plot(episodes, [87.06] *len(episodes), label="Best Solution FIFO", color=colors_css[colors_heuristics[0]],
                     linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [76.56] *len(episodes), label="Best Solution SPT", color=colors_css[colors_heuristics[1]],
                     linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [74.77] *len(episodes), label="Best Solution EDD", color=colors_css[colors_heuristics[2]],
                     linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [69.89] *len(episodes), label="Best Solution SCT", color=colors_css[colors_heuristics[3]],
                     linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [62.91] *len(episodes), label=f'Best Solution MIP (Kopanos)', color="black",
                     linestyle="dashed", linewidth=1.2)
        else:
            plt.plot(episodes, [62.91] * len(mean_eval), label=f'Best Solution MIP (Kopanos)', color="black",
                     linestyle="dashed", linewidth=1.5)
    elif column_name == "weighted_lateness":
        #plt.gca().set_ylim(bottom=0)
        if plot_heuristics:
            plt.plot(episodes, [762.32] *len(episodes), label="Best Solution PPO (ID32)", color=colors_css[colors_heuristics[4]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [1186.81] *len(episodes), label="Best Solution SAC (ID19)", color=colors_css[colors_heuristics[5]], linestyle="dashed", linewidth=1.2)

            plt.plot(episodes, [1180.45] *len(episodes), label="Best Solution FIFO", color=colors_css[colors_heuristics[0]], linestyle="dashed",
                     linewidth=1.2)
            plt.plot(episodes, [627.47] *len(episodes), label="Best Solution SPT", color=colors_css[colors_heuristics[1]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [230.75] *len(episodes), label="Best Solution EDD", color=colors_css[colors_heuristics[2]], linestyle="dashed", linewidth=1.2)
            plt.plot(episodes, [512.87] *len(episodes), label="Best Solution SCT", color=colors_css[colors_heuristics[3]], linestyle="dashed", linewidth=1.2)

            plt.plot(episodes, [19.09]*len(episodes), label=f'Best Solution MIP (Kopanos)', color="black", linestyle="dashed",linewidth=1.2)
        else:
            plt.plot(episodes, [19.09] * len(mean_eval), label=f'Best Solution MIP (Kopanos)', color="black",
                     linestyle="dashed", linewidth=1.5)

    last_y_values = mean_eval[int(len(mean_eval) * 0.9):int(len(mean_eval) * 1)]
    min = np.min(last_y_values)
    max = np.max(last_y_values)

    if plot_best and column_name != "total_reward":
        plt.plot(episodes, [min] * len(mean_eval), color=colors[idx], linestyle="dashed", linewidth=1.5, alpha=0.7,label=f"Best Solution {agent} ({exp_id})")

    plt.tight_layout()
    plt.xlabel('Episodes')
    plt.ylabel(format_column(column_name))
    min_val = np.min(df[column_name])
    if column_name != "total_reward" and min_val > 0:
        max_lim = np.max(mean_eval) + np.max(std_eval)
        plt.ylim(top=max_lim + max_lim * 0.25)
    else:
        min_lim = np.min(mean_eval) + np.min(std_eval)
        plt.ylim(bottom = min_lim + min_lim*0.25, top = 10)
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    #plt.suptitle(experiment_name)
    #plt.title(f'Episode vs. {format_column(column_name)} ({plot_type})')
    return df[column_name]
# List of columns to generate plots for
columns_to_plot = ['sim_duration', 'total_reward',"oc_costs","weighted_lateness","l"]


# Recursive function to process subdirectories and collect file paths
def process_directory(directory_path, loc_evaluation_files, loc_training_files):
    for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            subdir = os.path.join(root, dir)
            eval_csv_path = os.path.join(subdir, 'Logs_evaluation.monitor.csv')
            training_csv_path = os.path.join(subdir, 'Logs_training.monitor.csv')

            if os.path.exists(eval_csv_path):
                loc_evaluation_files.append(eval_csv_path)
            if os.path.exists(training_csv_path):
                loc_training_files.append(training_csv_path)


# Function to generate and save individual plots from file paths
def generate_individual_plots_from_files(agent,file_paths, column_name, title, path, is_evaluation):
    values_eval = [[] for _ in range(len(loc_evaluation_files))]
    values_train = [[] for _ in range(len(loc_evaluation_files))]
    for idx, file_path in enumerate(file_paths):
        experiment_name = get_experiment_id(file_path)
        if len(file_paths)>1:
            label = f"{agent} {idx+1} ({experiment_name})"
        else:
            label = f"{agent} ({experiment_name})"

        figure = plt.gcf()
        figure.set_size_inches(6, 4)
        lens = []
        column = generate_single_plot(file_path, column_name,label,idx, is_evaluation=is_evaluation)

        if is_evaluation:
            generate_plots_vs_custom(idx, file_path, "total_reward", "sim_duration", is_evaluation=True)
        else:
            generate_plots_vs_custom(idx, file_path, "total_reward", "sim_duration")

        if is_evaluation:
            if smoothing:

                values_eval[idx] = smooth_values(column)
            else:
                col , std= calculate_moving_average(column)
                values_eval[idx] = col.tolist()
        else:
            if smoothing:
                values_train[idx] = smooth_values(column)
            else:
                values_train[idx], std = calculate_moving_average(column)


    if is_evaluation:
        mins = []
        for value in values_eval:
            mins.append(len(value))
        minimum = np.min(mins)
        values_eval = [sublist[:minimum] for sublist in values_eval]
        mean_eval = np.mean(values_eval,axis=0)
        std_eval = np.std(values_eval, axis=0)
        episodes = list(range(1, len(mean_eval) + 1))
        if column_name == "total_reward":
            eval_means_for_experiment_reward[experiment_name] = mean_eval
        elif column_name == "sim_duration":
            eval_means_for_experiment_mks[experiment_name] = mean_eval
        elif column_name == "oc_costs":
            eval_means_for_experiment_occ[experiment_name] = mean_eval
        elif column_name == "weighted_lateness":
            eval_means_for_experiment_wl[experiment_name] = mean_eval
        elif column_name == "l":
            eval_means_for_experiment_length[experiment_name] = mean_eval

    else:
        mins = []
        for value in values_train:
            mins.append(len(value))
        if len(mins) > 1:
            minimum = np.min(mins)
            values_train = [sublist[:minimum] for sublist in values_eval]
        mean_train = np.mean(values_train, axis=0)
        std_train = np.std(values_train, axis=0)
        episodes = list(range(1, len(mean_train) + 1))
        plt.plot(episodes, mean_train, label=f"Mean ($\mu$)", color="black", linestyle="dashed")
        plt.fill_between(episodes, mean_train - std_train, mean_train + std_train, color='gray', alpha=0.2, label=f"STD ($\sigma$)")

        if column_name == "total_reward":
            train_means_for_experiment_reward[experiment_name] = mean_train
        elif column_name == "sim_duration":
            train_means_for_experiment_mks[experiment_name] = mean_train
        elif column_name == "oc_costs":
            train_means_for_experiment_occ[experiment_name] = mean_train
        elif column_name == "weighted_lateness":
            train_means_for_experiment_wl[experiment_name] = mean_train
        elif column_name == "l":
            train_means_for_experiment_length[experiment_name] = mean_train


    plt.xlabel('Episodes')
    plt.ylabel(format_column(column_name))
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    #plt.suptitle(experiment_name)
    #plt.title(f'Episode vs. {format_column(column_name)} ({plot_type})')


    if is_evaluation:
        plot_type = 'Evaluation'
    else:
        plot_type = 'Training'
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),ncol=1, bbox_to_anchor=(1.01, 0.8),loc='upper left')
    plt.savefig(os.path.join(output, f'{experiment_name}_{idx}_{column_name}.png'), dpi=400)
    plt.close()

def generate_plots_vs_custom(idx, csv_path, column_x, column_y, is_evaluation=False):
    experiment_name = get_experiment_id(csv_path)
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    df = pd.read_csv(csv_path, header=1)
    if column_x != "episodes":
        df, episodes = filter_outliers(df, df[column_x])
        df, episodes = filter_outliers(df, df[column_y])

        plt.figure()
        figure = plt.gcf()
        figure.set_size_inches(2,3)
        plt.scatter(df[column_x],df[column_y], label=f"{experiment_name}", s=10, alpha=0.7)
        p = calculate_trendline(df[column_x],df[column_y])
        plt.plot(df[column_x], p(df[column_x]), color="red")

        plt.xlabel(format_column(column_x))
        plt.ylabel(format_column(column_y))
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        #plt.suptitle(experiment_name)
        #plt.title(f'Run #{idx+1}: {plot_type} - {format_column(column_x)} vs. {format_column(column_y)}')
        #plt.legend(handlelength=1.0, handleheight=0.8)

        # Create a "Monitor Plots" subfolder within the same directory as the CSV file
        if column_y == "l":
            column_y = "episode_length"

        plt.savefig(os.path.join(output, f'{experiment_name}_{idx}_{column_x}_vs_{column_y}.png'),dpi=400)

        print("Plot!")
        plt.close()

if __name__ == "__main__":

    experiments_1 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID20",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID21",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID22",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID23",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID24",]

    experiments_2 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_6\Set_1\ID25",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_6\Set_1\ID26",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_6\Set_1\ID27",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_6\Set_1\ID28",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_6\Set_1\ID29",]

    experiments_3 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_1\ID30",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_1\ID28"]

    experiments_4 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_2\ID31",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_2\ID28"]
    experiments_5 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_3\ID32",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_3\ID28"]

    experiments_6 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_8\ID33",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_8\ID34"]

    experiments_7 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID39",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID40",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID41"]

    experiments_8 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID42",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID43",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID44",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID45"]

    experiments_9 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID46",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID47",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID48",
                     r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\04_HYBRID\ID49"]


    experiments_list_ = [experiments_1,experiments_2,experiments_3,experiments_4,experiments_5, experiments_7,experiments_8,experiments_9]

    experiments_list = [experiments_9]
    colors = ["blue", "red", "green", "brown", "orange", "pink"]
    for experiments in experiments_list:
        means_all = []
        ids_list = []

        for path in experiments:
            ids_list.append(get_experiment_id(path))

        ids_list_nums = list(np.char.replace(ids_list, 'ID', '').astype(int))

        eval_means_for_experiment_reward = dict.fromkeys(ids_list)
        eval_means_for_experiment_length = dict.fromkeys(ids_list)
        eval_means_for_experiment_mks = dict.fromkeys(ids_list)
        eval_means_for_experiment_occ = dict.fromkeys(ids_list)
        eval_means_for_experiment_wl = dict.fromkeys(ids_list)
        eval_means_for_experiment_l = dict.fromkeys(ids_list)

        train_means_for_experiment_reward = dict.fromkeys(ids_list)
        train_means_for_experiment_length = dict.fromkeys(ids_list)
        train_means_for_experiment_mks = dict.fromkeys(ids_list)
        train_means_for_experiment_occ = dict.fromkeys(ids_list)
        train_means_for_experiment_wl = dict.fromkeys(ids_list)
        train_means_for_experiment_l = dict.fromkeys(ids_list)

        for j, path_to_experiment in enumerate(experiments):
            # Start processing from the base source directory
            loc_evaluation_files = []
            loc_training_files = []
            process_directory(path_to_experiment, loc_evaluation_files, loc_training_files)
            ids_list = []
            exp_id = get_experiment_id(path_to_experiment)
            if exp_id == "ID16" or exp_id == "ID17" or exp_id == "ID19":
                agent = "SAC"
            elif exp_id == "ID35":
                agent = "FIFO"
            elif exp_id == "ID36":
                agent = "SPT"
            elif exp_id == "ID37":
                agent = "EDD"
            elif exp_id == "ID38":
                agent = "SCT"
            elif exp_id == "ID39" or exp_id == "ID40" or exp_id == "ID41":
                agent = "PPO & KOPANOS"
            elif exp_id == "ID42" or exp_id == "ID46":
                agent = "PPO & FIFO"
            elif exp_id == "ID43" or exp_id == "ID47":
                agent = "PPO & SPT"
            elif exp_id == "ID44" or exp_id == "ID48":
                agent = "PPO & EDD"
            elif exp_id == "ID45" or exp_id == "ID49":
                agent = "PPO & SCT"
            else:
                agent = "PPO"

            for i in range(len(columns_to_plot)):
                generate_individual_plots_from_files(agent,loc_evaluation_files, columns_to_plot[i], (f'{format_column(columns_to_plot[i])} vs. Episodes'), path_to_experiment, is_evaluation=True)
                #generate_individual_plots_from_files(loc_training_files, columns_to_plot[i], (f'{format_column(columns_to_plot[i])} vs. Episodes'), is_evaluation=False)

        eval_list = [eval_means_for_experiment_reward,
                     eval_means_for_experiment_mks,
                     eval_means_for_experiment_occ,
                     eval_means_for_experiment_wl]

        train_list = [train_means_for_experiment_reward,
                     train_means_for_experiment_mks,
                     train_means_for_experiment_occ,
                     train_means_for_experiment_wl]


        order  = ["Return", "Makespan", "Operating & Changeover Costs","Total Weighted Lateness"]
        lens = []
        for y, element in enumerate(eval_list):
            for i in range(len(element)):
                experiment_name = ids_list_nums[i]
                name = f"ID{experiment_name}"
                exp_id = name
                list_means = [[] for i in range(len(element))]
                y_values = element[name]
                if exp_id == "ID16" or exp_id == "ID17" or exp_id == "ID19":
                    agent = "SAC"
                elif exp_id == "ID35":
                    agent = "FIFO"
                elif exp_id == "ID36":
                    agent = "SPT"
                elif exp_id == "ID37":
                    agent = "EDD"
                elif exp_id == "ID38":
                    agent = "SCT"
                elif exp_id == "ID39" or exp_id == "ID40" or exp_id == "ID41":
                    agent = "PPO & KOPANOS"
                elif exp_id == "ID42" or exp_id == "ID46":
                    agent = "PPO & FIFO"
                elif exp_id == "ID43" or exp_id == "ID47":
                    agent = "PPO & SPT"
                elif exp_id == "ID44" or exp_id == "ID48":
                    agent = "PPO & EDD"
                elif exp_id == "ID45" or exp_id == "ID49":
                    agent = "PPO & SCT"
                else:
                    agent = "PPO"
                values = list(element[name])
                episodes = list(range(1, len(element[name]) + 1))
                #data = pd.DataFrame(values)
                #std = data.rolling(window=window_size, min_periods=1).std()
                std = np.std(element[name])
                plt.plot(episodes, element[name],label=f"{agent} $\mu$ ({exp_id})",color=colors[i])
                plt.fill_between(episodes, element[name]-std,element[name]+std,color=colors[i],alpha=0.2)

                last_y_values = element[name][int(len(element[name]) * 0.9):int(len(element[name]) * 1)]
                min = np.min(last_y_values)
                max = np.max(last_y_values)

                if plot_best and order[y] != "Return":
                    plt.plot(episodes, [min] * len(element[name]), color=colors[i], linestyle="dashed", linewidth=1.5,alpha=0.7,label=f"Best Solution {agent} ({exp_id})")

                print(experiment_name)

                lens.append(len(y_values))
            column_name = order[y]
            x_ax = range(1, np.max(lens) + 1)

            if column_name == "Makespan":
                #plt.gca().set_ylim(bottom=20)

                if plot_heuristics:
                    plt.plot(x_ax, [35.74] * np.max(lens), label=f'Best Solution PPO (ID23)',
                             color=colors_css["springgreen"],
                             linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [42.18] * np.max(lens), label=f'Best Solution SAC (ID16)',
                             color=colors_css["deepskyblue"],
                             linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [44.01] * np.max(lens), label="Best Solution FIFO",
                             color=colors_css[colors_heuristics[0]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [34.25] * np.max(lens), label="Best Solution SPT",
                             color=colors_css[colors_heuristics[1]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [33.55] * np.max(lens), label="Best Solution EDD",
                             color=colors_css[colors_heuristics[2]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [33.07] * np.max(lens), label="Best Solution SCT",
                             color=colors_css[colors_heuristics[3]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [26.56] * np.max(lens), label=f'Best Solution MIP (Kopanos)', color="black",
                             linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [25.01] * np.max(lens), label=f'Best Solution CP (Bleidorn)$^1$',
                             color=colors_css["lightseagreen"],
                             linestyle="dashed", linewidth=1.2)
                else:
                    plt.plot(episodes, [25.01] * len(y_values), label=f'Best Solution CP (Bleidorn)$^1$',
                             color=colors_css["lightseagreen"],
                             linestyle="dashed", linewidth=1.2)
                    plt.plot(episodes, [26.56] * len(y_values), label=f'Best Solution MIP (Kopanos)', color="black",
                             linestyle="dashed",
                             linewidth=1.2)


                plt.plot(episodes, [26.56] * len(element[name]), label=f'Best Solution MIP (Kopanos)', color="black",
                         linestyle="dashed", linewidth=1.5)
                plt.plot(episodes, [25.01] * len(element[name]), label=f'Best Solution CP (Bleidorn)$^1$', color=colors_css["lightseagreen"],
                         linestyle="dashed", linewidth=1.5)
            elif column_name == "Operating & Changeover Costs":
                #plt.gca().set_ylim(bottom=60)
                if plot_heuristics:
                    plt.plot(x_ax, [82.01] * np.max(lens), label=f'Best Solution PPO (ID23)',
                             color=colors_css["springgreen"],
                             linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [88.89] * np.max(lens), label=f'Best Solution SAC (ID19)',
                             color=colors_css["deepskyblue"],
                             linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [87.06] * np.max(lens), label="Best Solution FIFO",
                             color=colors_css[colors_heuristics[0]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [76.56] * np.max(lens), label="Best Solution SPT",
                             color=colors_css[colors_heuristics[1]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [74.77] * np.max(lens), label="Best Solution EDD",
                             color=colors_css[colors_heuristics[2]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [69.89] * np.max(lens), label="Best Solution SCT",
                             color=colors_css[colors_heuristics[3]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [62.91] * np.max(lens), label=f'Best Solution MIP (Kopanos)', color="black",
                             linestyle="dashed", linewidth=1.2)

                else:
                    plt.plot(episodes, [62.91] * len(element[name]), label=f'Best Solution MIP (Kopanos)', color="black",
                             linestyle="dashed", linewidth=1.5)
            elif column_name == "Total Weighted Lateness":
                #plt.gca().set_ylim(bottom=-10)

                if plot_heuristics:
                    plt.plot(x_ax, [762.32] * np.max(lens), label=f'Best Solution PPO (ID32)',
                             color=colors_css["springgreen"],
                             linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [1186.81] * np.max(lens), label=f'Best Solution SAC (ID19)',
                             color=colors_css["deepskyblue"],
                             linestyle="dashed", linewidth=1.2)

                    plt.plot(x_ax, [1180.45] * np.max(lens), label="Best Solution FIFO",
                             color=colors_css[colors_heuristics[0]], linestyle="dashed",
                             linewidth=1.2)
                    plt.plot(x_ax, [627.47] * np.max(lens), label="Best Solution SPT",
                             color=colors_css[colors_heuristics[1]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [230.75] * np.max(lens), label="Best Solution EDD",
                             color=colors_css[colors_heuristics[2]], linestyle="dashed", linewidth=1.2)
                    plt.plot(x_ax, [512.87] * np.max(lens), label="Best Solution SCT",
                             color=colors_css[colors_heuristics[3]], linestyle="dashed", linewidth=1.2)

                    plt.plot(x_ax, [19.09] * np.max(lens), label=f'Best Solution MIP (Kopanos)', color="black",
                             linestyle="dashed", linewidth=1.2)
                else:
                    plt.plot(episodes, [19.09] * len(element[name]), label=f'Best Solution MIP (Kopanos)', color="black",
                             linestyle="dashed", linewidth=1.5)

            print("Super Plot!")
            plt.xlabel("Episodes")
            plt.ylabel(order[y])
            fig = plt.gcf()
            fig.set_size_inches(5,4)
            fig.tight_layout()
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), ncol=1, bbox_to_anchor=(1.01, 0.8), loc='upper left')
            plt.savefig(os.path.join(output, f'Mean_{ids_list_nums[0]}-{ids_list_nums[-1]}_{column_name}.png'), dpi=400)
            plt.close()