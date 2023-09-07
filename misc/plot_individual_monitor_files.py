import matplotlib.pyplot as plt
import scienceplots

import pandas as pd
import numpy as np
import os
import re
from scipy import stats

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
window_size = 20
smoothing = False
output_directory = "plots"

if smoothing:
    smoothing_factor = 0.99
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

def make_homogeneous(array):
    # Find the length of the smallest sublist
    smallest_length = min(len(sublist) for sublist in array)

    # Truncate all sublists to the length of the smallest sublist
    new_array = [sublist[:smallest_length] for sublist in array]
    return new_array

def generate_single_plot(csv_path, column_name, label,idx, is_evaluation):

    df = pd.read_csv(csv_path, header=1)
    df, episodes = filter_outliers(df, df[column_name])
    #episodes = list(range(1, len(df) + 1))
    if smoothing:
        df[column_name] = smooth_values(df[column_name])
    else:
        df[column_name], std = calculate_moving_average(df[column_name])
    plt.plot(episodes, df[column_name], label=label)
    plt.xlabel('Episodes')
    plt.ylabel(format_column(column_name))
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
def generate_individual_plots_from_files(file_paths, column_name, title, is_evaluation):
    plt.figure(figsize=(10, 6))

    values_eval = [[] for _ in range(len(loc_evaluation_files))]
    values_train = [[] for _ in range(len(loc_evaluation_files))]

    for idx, file_path in enumerate(file_paths):
        experiment_name = get_experiment_id(file_path)
        label = f"{experiment_name} ({idx+1})"

        column = generate_single_plot(file_path, column_name,label,idx, is_evaluation=is_evaluation)
        if is_evaluation:
            generate_plots_vs_custom(idx, file_path, "total_reward", "sim_duration", is_evaluation=True)
        else:
            generate_plots_vs_custom(idx, file_path, "total_reward", "sim_duration")

        if is_evaluation:
            if smoothing:
                values_eval[idx] = smooth_values(column)
            else:
                values_eval[idx] , std= calculate_moving_average(column)

        else:
            if smoothing:
                values_train[idx] = smooth_values(column)
            else:
                values_train[idx], std = calculate_moving_average(column)


    if is_evaluation:
        values_eval = make_homogeneous(values_eval) #number of episodes may differ in each experiment
        #values_eval = np.array(values_eval,dtype=float)
        mean_eval = np.mean(values_eval,axis=0)
        std_eval = np.std(values_eval, axis=0)
        episodes = list(range(1, len(mean_eval) + 1))
        plt.plot(episodes, mean_eval, label=f"{experiment_name} ($\mu$)", color="black", linestyle="dashed")
        plt.fill_between(episodes, mean_eval - std_eval, mean_eval + std_eval, color='gray', alpha=0.2, label=f"STD ($\sigma$)")

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
        values_train = make_homogeneous(values_train)
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

    plt.tight_layout()

    if is_evaluation:
        plot_type = 'Evaluation'
    else:
        plot_type = 'Training'
    figure = plt.gcf()
    figure.set_size_inches(8, 5)
    plt.legend(ncol=len(file_paths)+2, loc='upper center')  # 9 means top center
    plt.savefig(os.path.join(output, f'{experiment_name}_{idx}_{plot_type}_{format_column(column_name)}.png'), dpi=400)
    plt.savefig(os.path.join(output_svg, f'{experiment_name}_{idx}_{plot_type}_{format_column(column_name)}.svg'), dpi=400,transparent=True, format="svg")
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
        figure.set_size_inches(8, 5)
        plt.scatter(df[column_x],df[column_y], label=f"{experiment_name}", s=10, alpha=0.7)
        p = calculate_trendline(df[column_x],df[column_y])
        plt.plot(df[column_x], p(df[column_x]), color="red")

        plt.xlabel(format_column(column_x))
        plt.ylabel(format_column(column_y))
        plt.legend(loc='upper right')
        #plt.suptitle(experiment_name)
        #plt.title(f'Run #{idx+1}: {plot_type} - {format_column(column_x)} vs. {format_column(column_y)}')
        #plt.legend(handlelength=1.0, handleheight=0.8)

        # Create a "Monitor Plots" subfolder within the same directory as the CSV file
        if column_y == "l":
            column_y = "episode_length"

        plt.savefig(os.path.join(output, f'{experiment_name}_{idx}_{plot_type}_{column_x}_vs_{column_y}.png'),dpi=400)
        plt.savefig(os.path.join(output_svg, f'{experiment_name}_{idx}_{plot_type}_{column_x}_vs_{column_y}.svg'), dpi=400,transparent=True, format="svg")

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
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_1\ID33"]

    experiments_4 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_2\ID31",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_2\ID33"]
    experiments_5 = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_3\ID32",
         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\Set_3\ID33"]

    experiments_list = [experiments_1,experiments_2,experiments_3,experiments_4,experiments_5]

    colors = ["red", "blue", "green", "black", "orange", "pink"]
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

            for i in range(len(columns_to_plot)):
                generate_individual_plots_from_files(loc_evaluation_files, columns_to_plot[i], (f'{format_column(columns_to_plot[i])} vs. Episodes'), is_evaluation=True)
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

        for y, element in enumerate(eval_list):
            for i in range(len(element)):
                experiment_name = ids_list_nums[i]
                name = f"ID{experiment_name}"
                episodes = list(range(1, len(element[name]) + 1))
                plt.plot(episodes, element[name],label=f"{name} ($\mu$)",color=colors[i])
                print(experiment_name)
            print("Super Plot!")
            plt.xlabel("Episodes")
            plt.ylabel(order[y])
            figure = plt.gcf()
            plt.tight_layout()
            plt.legend(ncol=len(element), loc='upper right')  # 9 means top center
            figure.set_size_inches(11, 5)

            plt.savefig(os.path.join(output, f'Mean_Evaluation_{experiment_name}_{order[y]}.png'), dpi=400)
            plt.savefig(os.path.join(output_svg, f'Mean_Evaluation_{experiment_name}_{order[y]}.svg'), dpi=400,
                        transparent=True, format="svg")
            plt.close()