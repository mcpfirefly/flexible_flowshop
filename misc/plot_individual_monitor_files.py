import matplotlib.pyplot as plt
import scienceplots

import pandas as pd
import numpy as np
import os
from scipy import stats

experiment_name = "ID32"
base_source_directory = "C:/Users/INOSIM/OneDrive - INOSIM Consulting GmbH/Desktop/all_30/all_30/ID32_big_new_os"
output = base_source_directory + "/individual_plots"
os.makedirs(output, exist_ok=True)
plt.style.use(['science','no-latex','grid'])
# Define your base source directory
window_size = 12
output_directory = "plots"
smoothing_factor = 0.99
if smoothing_factor != None or smoothing_factor != 0:
    smoothing = True
    window_size = None

loc_eval_files = []
loc_train_files = []
# Function to calculate moving average
def calculate_moving_average(data):
    return data.rolling(window=window_size, min_periods=1).mean()

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

def generate_single_plot(csv_path, column_name, label, is_evaluation):
    df = pd.read_csv(csv_path, header=1)
    if column_name == "total_reward":
        df[column_name] = -df[column_name]
    df, episodes = filter_outliers(df, df[column_name])
    #episodes = list(range(1, len(df) + 1))
    if smoothing:
        df[column_name] = smooth_values(df[column_name])
    else:
        df[column_name] = calculate_moving_average(df[column_name])
    plt.plot(episodes, df[column_name], label=label)
    plt.xlabel('Episode')
    plt.ylabel(format_column(column_name))
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    plt.suptitle(experiment_name)
    plt.title(f'Episode vs. {format_column(column_name)} ({plot_type})')
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


# Start processing from the base source directory
loc_evaluation_files = []
loc_training_files = []
process_directory(base_source_directory, loc_evaluation_files, loc_training_files)


# Function to generate and save individual plots from file paths
def generate_individual_plots_from_files(file_paths, column_name, title, is_evaluation):
    plt.figure(figsize=(10, 6))

    values_eval = [[] for _ in range(len(loc_evaluation_files))]
    values_train = [[] for _ in range(len(loc_evaluation_files))]

    for idx, file_path in enumerate(file_paths):
        column = generate_single_plot(file_path, column_name, f'{format_column(column_name)} #{idx + 1}', is_evaluation=is_evaluation)
        if is_evaluation:
            generate_plots_vs_custom(idx, file_path, "total_reward", "sim_duration", is_evaluation=True)
        else:
            generate_plots_vs_custom(idx, file_path, "total_reward", "sim_duration")

        if is_evaluation:
            if smoothing:
                values_eval[idx] = smooth_values(column)
            else:
                values_eval[idx] = calculate_moving_average(column)

        else:
            if smoothing:
                values_train[idx] = smooth_values(column)
            else:
                values_train[idx] = calculate_moving_average(column)


    if is_evaluation:
        values_eval = make_homogeneous(values_eval) #number of episodes may differ in each experiment
        #values_eval = np.array(values_eval,dtype=float)
        mean_eval = np.mean(values_eval,axis=0)
        std_eval = np.std(values_eval, axis=0)
        episodes = list(range(1, len(mean_eval) + 1))
        plt.plot(episodes, mean_eval, label='Mean', color="black", linestyle="dashed")
        plt.fill_between(episodes, mean_eval - std_eval, mean_eval + std_eval, color='gray', alpha=0.2)

    else:
        values_train = make_homogeneous(values_train)
        mean_train = np.mean(values_train, axis=0)
        std_train = np.std(values_train, axis=0)
        episodes = list(range(1, len(mean_train) + 1))
        plt.plot(episodes, mean_train, label='Mean', color="black", linestyle="dashed")
        plt.fill_between(episodes, mean_train - std_train, mean_train + std_train, color='gray', alpha=0.2)

    plt.xlabel('Episode')
    plt.ylabel(format_column(column_name))
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    plt.suptitle(experiment_name)
    plt.title(f'Episode vs. {format_column(column_name)} ({plot_type})')

    plt.legend()
    plt.tight_layout()

    if is_evaluation:
        plot_type = 'Evaluation'
    else:
        plot_type = 'Training'
    figure = plt.gcf()
    figure.set_size_inches(11, 5)
    plt.savefig(os.path.join(output, f'{plot_type}_{format_column(column_name)}_Plot.png'), dpi=400)
    plt.close()

def generate_plots_vs_custom(idx, csv_path, column_x, column_y, is_evaluation=False):
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    df = pd.read_csv(csv_path, header=1)
    if column_x != "episodes":
        df, episodes = filter_outliers(df, df[column_x])
        df, episodes = filter_outliers(df, df[column_y])

        if column_y == "total_reward":
            df[column_y] = -df[column_y]
        if column_x == "total_reward":
            df[column_x] = -df[column_x]

        p = calculate_trendline(df[column_x],df[column_y])
        plt.figure()
        figure = plt.gcf()
        figure.set_size_inches(11, 5)
        plt.scatter(df[column_x],df[column_y], label=format_column(column_y), s=10, alpha=0.7)
        plt.plot(df[column_x], p(df[column_x]), color="red")

        plt.xlabel(format_column(column_x))
        plt.ylabel(format_column(column_y))
        plt.suptitle(experiment_name)
        plt.title(f'Run #{idx+1}: {plot_type} - {format_column(column_x)} vs. {format_column(column_y)}')
        plt.legend(handlelength=1.0, handleheight=0.8)

        # Create a "Monitor Plots" subfolder within the same directory as the CSV file
        if column_y == "l":
            column_y = "episode_length"

        plt.savefig(os.path.join(output, f'{idx+1}_{plot_type}_{column_x}_vs_{column_y}_plot.png'),dpi=400)
        print("Plot!")
        plt.close()

for i in range(len(columns_to_plot)):
    generate_individual_plots_from_files(loc_evaluation_files, columns_to_plot[i], (f'{format_column(columns_to_plot[i])} vs. Episodes'), is_evaluation=True)
    generate_individual_plots_from_files(loc_training_files, columns_to_plot[i], (f'{format_column(columns_to_plot[i])} vs. Episodes'), is_evaluation=False)