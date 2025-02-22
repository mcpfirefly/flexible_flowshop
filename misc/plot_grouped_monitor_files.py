import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats

plt.style.use(['science','no-latex','grid'])
# Define your base source directory
window_size = 12
output_directory = r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\plots"
smoothing_factor = 0
if smoothing_factor != None or smoothing_factor != 0:
    smoothing = True
    window_size = None

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
        return 'OCC [$]'
    elif column == 'total_reward':
        return 'Return'
    elif column == "sim_duration":
        return 'Makespan [h]'
    elif column == 'weighted_lateness':
        return 'Total WL [h]'
    elif column == 'l':
        return 'Episode Length [steps]'
    elif column == "completion_score":
        return "Demand Completion [%]"
    return column

def make_homogeneous(array):
    # Find the length of the smallest sublist
    smallest_length = min(len(sublist) for sublist in array)

    # Truncate all sublists to the length of the smallest sublist
    new_array = [sublist[:smallest_length] for sublist in array]
    return new_array

def generate_single_plot(csv_path, column_name, label, is_evaluation):
    df = pd.read_csv(csv_path, header=1)
    df, episodes = filter_outliers(df, df[column_name])
    experiment_name = get_experiment_id(csv_path)
    #episodes = list(range(1, len(df) + 1))
    if column_name == "completion_score":
        df[column_name] = df[column_name]*100
    if smoothing:
        df[column_name] = smooth_values(df[column_name])
    else:
        df[column_name] = calculate_moving_average(df[column_name])
    plt.plot(episodes, df[column_name], label=label)
    plt.xlabel('Episode')
    plt.ylabel(format_column(column_name))
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    #plt.title(f'Episode vs. {format_column(column_name)} ({plot_type})')
    return df[column_name]

# Function to generate and save individual plots from file paths
def generate_individual_plots_from_files(file_paths, column_name, title, is_evaluation, ax=None):
    if ax is None:
        plt.figure(figsize=(8,6))

    values_eval = [[] for _ in range(len(file_paths))]
    values_train = [[] for _ in range(len(file_paths))]

    for idx, file_path in enumerate(file_paths):
        exp_id = get_experiment_id(file_path)
        column = generate_single_plot(file_path, column_name, f'{exp_id} ($\mu$)', is_evaluation=is_evaluation)

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
        values_eval = make_homogeneous(values_eval)
        mean_eval = np.mean(values_eval, axis=0)
        std_eval = np.std(values_eval, axis=0)
        experiment_name = get_experiment_id(file_paths[0])
        episodes = list(range(1, len(mean_eval) + 1))
        ax.plot(episodes, mean_eval, label=f"{exp_id} $(μ)$", color="black", linestyle="dashed")
        ax.fill_between(episodes, mean_eval - std_eval, mean_eval + std_eval, color='gray', alpha=0.2, label ="STD $(σ)$")
    else:
        values_train = make_homogeneous(values_train)
        mean_train = np.mean(values_train, axis=0)
        std_train = np.std(values_train, axis=0)


        episodes = list(range(1, len(mean_train) + 1))
        ax.plot(episodes, mean_train, label=f"{exp_id} $(μ)$", color="black", linestyle="dashed")
        ax.fill_between(episodes, mean_train - std_train, mean_train + std_train, color='gray', alpha=0.2, label = "STD $(σ)$")

    ax.set_xlabel('Episode')
    ax.set_ylabel(format_column(column_name))
    ax.set_aspect("auto")
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    #ax.set_title(f'Episode vs. {format_column(column_name)}')
    #ax.legend(handlelength=1.0, handleheight=0.8)

    if ax is None:
        plt.tight_layout()

        if is_evaluation:
            plot_type = 'Evaluation'
        else:
            plot_type = 'Training'
        figure = plt.gcf()
        figure.set_size_inches(11, 5)
        plt.close()

def generate_plots_vs_custom(idx, csv_path, column_x, column_y, is_evaluation=False, ax=None):
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    df = pd.read_csv(csv_path, header=1)
    exp_id = get_experiment_id(csv_path)
    if column_x != "episodes":
        if column_y == "completion_score":
            df[column_y] = df[column_y] * 100

        df, episodes = filter_outliers(df, df[column_x])
        df, episodes = filter_outliers(df, df[column_y])
        p = calculate_trendline(df[column_x], df[column_y])

        # Use the passed axes instead of creating a new figure
        if ax is None:
            ax.legend()
            ax = plt.gca()

        ax.scatter(df[column_x], df[column_y], label=(f"{exp_id} ($\mu$)"), s=10, alpha=0.7)
        ax.plot(df[column_x], p(df[column_x]), color="red")

        ax.set_xlabel(format_column(column_x))
        ax.set_ylabel(format_column(column_y))
        #ax.set_title(f'{exp_id} ($\mu$)')
        ax.legend(handlelength=0.9, handleheight=0.8)
        # Create a "Monitor Plots" subfolder within the same directory as the CSV file
        if column_y == "l":
            column_y = "episode_length"

        # No need to save or close the figure here


# Create a function to generate subplots for custom plots
def generate_subplots_for_custom_plots(file_paths, column_x, column_y, is_evaluation=False):
    num_plots = len(file_paths)
    num_cols = 3  # Number of columns in the subplot grid
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows
    experiment_name = get_experiment_id(file_paths[0])
    if is_evaluation:
        plot_type = 'Evaluation'
    else:
        plot_type = 'Training'

    plt.figure(figsize=(8,6))
    #plt.suptitle(f'{experiment_name}: {format_column(column_x)} vs. {format_column(column_y)} ({plot_type})')

    for idx, file_path in enumerate(file_paths, 1):
        ax = plt.subplot(num_rows, num_cols, idx)
        generate_plots_vs_custom(idx, file_path, column_x, column_y, is_evaluation=is_evaluation, ax=ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to prevent overlap
    # Add space for the title
    plt.subplots_adjust(top=0.88)
    #figure = plt.gcf()
    #figure.set_size_inches(11, 6)
    plt.savefig(os.path.join(output_directory, f'{experiment_name}_{plot_type}_{column_x}_vs_{column_y}.png'), dpi=400)
    plt.close()

def get_experiment_id(csv_path):
    pattern = r'\\(ID\d+)'
    # Search for the pattern in the path
    match = re.search(pattern, csv_path)
    # Check if a match was found
    if match:
        extracted_string = match.group(1)  # Get the first matching group (ID01 in this case)
        print(extracted_string)
        return extracted_string

# Define a function to generate subplots for a given set of file paths, columns, and plot title
def generate_subplots(file_paths, columns, is_evaluation):
    num_plots = len(columns)
    num_cols = 1
    num_rows = (num_plots + num_cols - 1) // num_cols

    if is_evaluation:
        plot_type = 'Evaluation'
    else:
        plot_type = 'Training'

    plt.figure(figsize=(8,6))

    for idx, column in enumerate(columns, 1):
        ax = plt.subplot(num_rows, num_cols, idx)
        generate_individual_plots_from_files(file_paths, column, f'{format_column(column)} vs. Episodes',
                                             is_evaluation=is_evaluation, ax=ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(top=0.88)
    fig = plt.gcf()
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    # Create a legend with one column outside the plot on the right
    ncol = 1
    plt.legend(lines[0:len(file_paths) + 2], labels[0:len(file_paths) + 2], ncol=ncol, bbox_to_anchor=(1.01,1.6),
               loc='upper left')

    experiment_name = get_experiment_id(file_paths[0])
    plt.savefig(os.path.join(output_directory, (f'{experiment_name}_ {plot_type}_Performance_Overview.png')),
                dpi=400,  )
    plt.close()

# Recursive function to process subdirectories and collect file paths

def get_data_from_files(files,variables):

    min_col = []
    mean_col = []
    median_col = []
    std_col = []
    ids = []
    vars = []
    for path in files:
        df = pd.read_csv(path,skiprows=1)
        for column in variables:
            ids.append(get_experiment_id(path))
            col = df[column]
            col = col[int(len(col)*0.9):int(len(col)*1)]
            min_col.append(np.min(col))
            mean_col.append(np.mean(col))
            median_col.append(np.median(col))
            std_col.append(np.std(col))

            if column == "sim_duration":
                vars.append("makespan")
            elif column == "oc_costs":
                vars.append("occ")
            elif column == "weighted_lateness":
                vars.append("wl")

    values = pd.DataFrame(list(zip(ids,min_col, mean_col,median_col,std_col,vars)),columns =['id','best', 'mean','median','std','variable'])
    subfolder = r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results"
    os.makedirs(subfolder, exist_ok=True)
    values.to_csv(os.path.join(subfolder,"general_data.csv"),index=False)
def process_directory(directory_path, loc_evaluation_files, loc_training_files, columns_to_plot):
    for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            subdir = os.path.join(root, dir)
            eval_csv_path = os.path.join(subdir, 'Logs_evaluation.monitor.csv')
            training_csv_path = os.path.join(subdir, 'Logs_training.monitor.csv')

            if os.path.exists(eval_csv_path):
                loc_evaluation_files.append(eval_csv_path)
            if os.path.exists(training_csv_path):
                loc_training_files.append(training_csv_path)


    get_data_from_files(loc_evaluation_files,columns_to_plot)
    # Call the function to generate subplots for custom plots
    #generate_subplots_for_custom_plots(loc_evaluation_files, 'total_reward', 'sim_duration', is_evaluation=True)
    #generate_subplots_for_custom_plots(loc_training_files, 'total_reward', 'sim_duration', is_evaluation=False)

    # Modify your loop to generate combined plots for evaluation and training
    generate_subplots(loc_evaluation_files, columns_to_plot, is_evaluation=True)
    #generate_subplots(loc_training_files, columns_to_plot, is_evaluation=False)


# List of columns to generate plots for
columns_to_plot = ["sim_duration","oc_costs","weighted_lateness","total_reward"]
columns_to_plot_ = ['total_reward',"oc_costs"]
columns_to_plot_ = ['total_reward',"weighted_lateness"]
columns_to_plot_ = ['total_reward',"completion_score"]

directories_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\ID30"]
directories_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\ID31"]
directories_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_7\ID32"]

directories_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Continuous\ID04",
                r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Continuous\ID05",
                r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Continuous\ID06",]
directories_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID20",
               r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID21",
               r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID22",
               r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\The sis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID23",
               r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_5\Set_1\ID24"]

directories = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\experiments"]


for i, base_source_directory in enumerate(directories):
    loc_evaluation_files = []
    loc_training_files = []
    process_directory(base_source_directory, loc_evaluation_files, loc_training_files,columns_to_plot)
