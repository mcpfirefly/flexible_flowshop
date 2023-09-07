import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats

plt.style.use(['science','no-latex','grid'])



# Define your base source directory
window_size = 20
smoothing = False
moving_avg = True
multiple = False
if smoothing:
    smoothing_factor = 0.9
    window_size = None

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
        return 'Operating & Changeover Costs'
    elif column == 'total_reward':
        return 'Return'
    elif column == "sim_duration":
        return 'Makespan'
    elif column == 'weighted_lateness':
        return 'Total Weighted Lateness'
    elif column == 'l':
        return 'Episode Length'
    return column

def make_homogeneous(array):
    # Find the length of the smallest sublist
    smallest_length = min(len(sublist) for sublist in array)

    # Truncate all sublists to the length of the smallest sublist
    new_array = [sublist[:smallest_length] for sublist in array]
    return new_array

def get_experiment_id(csv_path):
    pattern = r'\\(ID\d+)'
    # Search for the pattern in the path
    match = re.search(pattern, csv_path)
    # Check if a match was found
    if match:
        extracted_string = match.group(1)  # Get the first matching group (ID01 in this case)
        print(extracted_string)
        return extracted_string

def get_batch_id(csv_path):
    pattern = r'\\(Set_\d+)'
    # Search for the pattern in the path
    match = re.search(pattern, csv_path)
    # Check if a match was found
    if match:
        extracted_string = match.group(1)  # Get the first matching group (ID01 in this case)
        print(extracted_string)
        return extracted_string

# Recursive function to process subdirectories and collect file paths
def process_directory(directory_path, loc_evaluation_files, loc_training_files):
    for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            subdir = os.path.join(root, dir)
            eval_csv_path = os.path.join(subdir, 'Logs_evaluation.monitor.csv')
            train_csv_path = os.path.join(subdir, 'Logs_training.monitor.csv')
            if os.path.exists(eval_csv_path):
                loc_evaluation_files.append(eval_csv_path)
                loc_training_files.append(train_csv_path)


def generate_plots_vs_custom_grouped(batch_id, exp_id,csv_path, column_x, column_y, ax, is_evaluation=False):

    df = pd.read_csv(csv_path, header=1)
    df, episodes = filter_outliers(df, df[column_x])
    df, episodes = filter_outliers(df, df[column_y])
    p = calculate_trendline(df[column_x], df[column_y])

    ax.scatter(df[column_x], df[column_y], label=exp_id, s=10, alpha=0.7)
    ax.plot(df[column_x], p(df[column_x]), color="red")


    ax.set_xlabel('Return')
    ax.set_ylabel(format_column(column_y))
    ax.set_aspect("auto")
    #ax.legend(handlelength=1.0, handleheight=0.8)


def generate_plots_vs_custom(batch_id, exp_id,csv_path, column_x, column_y, is_evaluation=False):

    df = pd.read_csv(csv_path, header=1)
    df, episodes = filter_outliers(df, df[column_x])
    df, episodes = filter_outliers(df, df[column_y])
    p = calculate_trendline(df[column_x], df[column_y])

    plt.scatter(df[column_x], df[column_y], label=exp_id, s=10, alpha=0.7)
    plt.plot(df[column_x], p(df[column_x]), color="red")


    plt.xlabel('Return')
    plt.ylabel(format_column(column_y))
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    figure = plt.gcf()
    figure.set_size_inches(8, 5)
    plt.tight_layout()
    plt.legend()

    plt.savefig(os.path.join(output,
                             f'{batch_id}_{exp_id}_{format_column(reward)}_vs_{format_column(column_y)}_{plot_type}.png'),
                dpi=400)
    plt.savefig(os.path.join(output_svg,
                             f'{batch_id}_{exp_id}_{format_column(reward)}_vs_{format_column(column_y)}_{plot_type}.svg'),
                dpi=400,
                transparent=True, format="svg")
    plt.close()
    plt.close()


def generate_single_plot(files, column_name, is_evaluation):
    y_values_array = []
    x_values_array = []
    exp_id_array = []
    colors = ["red","blue","green","black","orange","pink"]
    fig, axs = plt.subplots(len(files))
    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        generate_plots_vs_custom_grouped(batch_id, exp_id, csv_path, reward, column_name, axs[i],is_evaluation)
        axs[i].legend()
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    #fig.legend(ncol=len(files), loc='upper center')
    fig.savefig(os.path.join(output,
                             f'Grouped_{batch_id}_{format_column(reward)}_vs_{format_column(column_name)}_{plot_type}.png'),
                dpi=400)
    fig.savefig(os.path.join(output_svg,
                             f'Grouped_{batch_id}_{format_column(reward)}_vs_{format_column(column_name)}_{plot_type}.svg'),
                dpi=400,
                transparent=True, format="svg")
    plt.close()


    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        generate_plots_vs_custom(batch_id, exp_id, csv_path, reward, column_name, is_evaluation)


    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        df = pd.read_csv(csv_path, header=1)
        df, episodes = filter_outliers(df, df[column_name])
        y_values = df[column_name]
        #episodes = list(range(1, len(df) + 1))
        if smoothing:
            y_values_smoothed = smooth_values(y_values)
            std = np.std(y_values)
        elif moving_avg:
            y_values_ma, std = calculate_moving_average(y_values)
            episodes = list(range(1, len(y_values) + 1))

        plt.plot(episodes, y_values, alpha=0.1, color=colors[i])
        if smoothing:
            plt.plot(episodes, y_values_smoothed, label=f'{exp_id}',color=colors[i])
        elif moving_avg:
            plt.plot(episodes, y_values_ma, label=f'{exp_id}', color=colors[i])
        #plt.fill_between(episodes, y_values - std, y_values + std, color='gray', alpha=0.2)
        y_values_array.append(y_values)
        x_values_array.append(episodes)
        exp_id_array.append(exp_id)
    plt.xlabel('Episodes')
    plt.ylabel(format_column(column_name))
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    figure = plt.gcf()
    figure.set_size_inches(8, 5)
    plt.tight_layout()
    plt.legend(ncol=len(files), loc=9)  # 9 means top center
    plt.savefig(os.path.join(output, f'{batch_id}_{format_column(column_name)}_{plot_type}.png'), dpi=400)
    plt.savefig(os.path.join(output_svg, f'{batch_id}_{format_column(column_name)}_{plot_type}.svg'), dpi=400, transparent=True, format="svg")
    plt.close()

def generate_single_plot_ALL(files, column_name, is_evaluation):
    y_values_array = []
    x_values_array = []
    exp_id_array = []
    colors = ["red","blue","green","black","orange","pink"]
    fig, axs = plt.subplots(len(files))
    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        generate_plots_vs_custom_grouped(batch_id, exp_id, csv_path, reward, column_name, axs[i],is_evaluation)
        axs[i].legend()
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    fig.tight_layout()
    #fig.legend(ncol=len(files), loc='upper center')
    fig.savefig(os.path.join(output,
                             f'Grouped_{batch_id}_{format_column(reward)}_vs_{format_column(column_name)}_{plot_type}.png'),
                dpi=400)
    fig.savefig(os.path.join(output_svg,
                             f'Grouped_{batch_id}_{format_column(reward)}_vs_{format_column(column_name)}_{plot_type}.svg'),
                dpi=400,
                transparent=True, format="svg")
    plt.close()


    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        generate_plots_vs_custom(batch_id, exp_id, csv_path, reward, column_name, is_evaluation)


    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        df = pd.read_csv(csv_path, header=1)
        df, episodes = filter_outliers(df, df[column_name])
        y_values = df[column_name]
        #episodes = list(range(1, len(df) + 1))
        if smoothing:
            y_values_smoothed = smooth_values(y_values)
            std = np.std(y_values)
        elif moving_avg:
            y_values_ma, std = calculate_moving_average(y_values)
            episodes = list(range(1, len(y_values) + 1))

        plt.plot(episodes, y_values, alpha=0.1, color=colors[i])
        if smoothing:
            plt.plot(episodes, y_values_smoothed, label=f'{exp_id}',color=colors[i])
        elif moving_avg:
            plt.plot(episodes, y_values_ma, label=f'{exp_id}', color=colors[i])
        #plt.fill_between(episodes, y_values - std, y_values + std, color='gray', alpha=0.2)
        y_values_array.append(y_values)
        x_values_array.append(episodes)
        exp_id_array.append(exp_id)
    plt.xlabel('Episodes')
    plt.ylabel(format_column(column_name))
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    figure = plt.gcf()
    figure.set_size_inches(8, 5)
    plt.tight_layout()
    plt.legend(ncol=len(files), loc=9)  # 9 means top center
    plt.savefig(os.path.join(output, f'{batch_id}_{format_column(column_name)}_{plot_type}.png'), dpi=400)
    plt.savefig(os.path.join(output_svg, f'{batch_id}_{format_column(column_name)}_{plot_type}.svg'), dpi=400, transparent=True, format="svg")
    plt.close()


reward = "total_reward"

base_source_directory = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Set_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Set_3",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_2\Set_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_2\Set_2",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_3\Set_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_4\Set_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_8\Set_1"]



for i, folder in enumerate(base_source_directory):
    loc_evaluation_files = []
    loc_training_files = []

    process_directory(folder, loc_evaluation_files, loc_training_files)
    variables = ['sim_duration', "oc_costs", "weighted_lateness"]

    base = "C:/Users/INOSIM/OneDrive - INOSIM Consulting GmbH/General/Thesis Overviews - MCPF/03_Others/results"
    output = base + "/plots"
    output_svg = base + "/plots_svg"

    os.makedirs(output, exist_ok=True)
    os.makedirs(output_svg, exist_ok=True)

    for i in range(len(variables)):
        generate_single_plot(loc_evaluation_files, variables[i], is_evaluation=True)
        generate_single_plot(loc_training_files, variables[i], is_evaluation=False)