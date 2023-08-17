import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Define your base source directory
base_source_directory = 'C:/Users/INOSIM/PycharmProjects/flexible_flowshop/src/experiments/results'
output_directory = "plots"

# Function to calculate moving average
def calculate_moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# Function to capitalize the first letter of each word
def format_column_name(column_name):
    if column_name == 'oc_costs':
        return 'OCC'
    elif column_name == 'total_reward':
        return 'Return'
    elif column_name == "sim_duration":
        return 'Makespan'
    elif column_name == 'weighted_lateness':
        return 'Weighted Lateness'
    elif column_name == 'l':
        return 'Episode Length'

    return column_name

# Function to generate and save individual plots
def generate_single_plots(csv_path, column_name, use_steps=False, window_size=12, is_evaluation=False):
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    df = pd.read_csv(csv_path, header=1)
    if column_name == "total_reward":
        df[column_name] = -df[column_name]

    # Calculate mean and standard deviation of the column
    mean_value = df[column_name].mean()
    std_value = df[column_name].std()
    filtered_column = df[column_name].apply(lambda x: x if abs(x - mean_value) <= 3 * std_value else np.nan)

    df[column_name + '_MA'] = calculate_moving_average(df[column_name], window_size)



    # Apply a logarithm transformation to handle outliers

    episode = list(range(0, len(df)*100, 100)) if use_steps else list(range(1, len(df)+1))
    aligned_episode = [ep for ep, val in zip(episode, filtered_column) if not np.isnan(val)]

    plt.figure()
    #plt.plot(episode, df[column_name], label=format_column_name(column_name),alpha=0.7)
    plt.scatter(aligned_episode, filtered_column.dropna(), label=f'{format_column_name(column_name)}', s=10,alpha=0.7,color="lightgray")
    plt.plot(episode, df[column_name + '_MA'], label=f"Average {format_column_name(column_name)}")
    plt.xlabel('Episodes')
    plt.ylabel(format_column_name(column_name))
    plt.title(f'{plot_type} - {format_column_name(column_name)}')
    plt.legend(handlelength=1.0, handleheight=0.8)

    # Create a "Monitor Plots" subfolder within the same directory as the CSV file
    plots_subfolder = os.path.join(os.path.dirname(csv_path), 'Monitor Plots')
    os.makedirs(plots_subfolder, exist_ok=True)
    if column_name == "l":
        column_name = "episode_length"
    plt.savefig(os.path.join(plots_subfolder, f'{plot_type}_{column_name}_plot.png'), dpi=400)
    plt.close()
    print(f"Plot! {os.path.dirname(csv_path)} {plot_type}_{column_name}_plot.png")
# Function to generate and save combined plots
def generate_combined_plots(csv_path, column_name1, column_name2, window_size=12, is_evaluation=False, use_steps=False):
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    df = pd.read_csv(csv_path, header=1)


    if column_name1 == "total_reward":
        df[column_name1] = -df[column_name1]
    if column_name2 == "total_reward":
        df[column_name2] = -df[column_name2]

    df[column_name1 + '_MA'] = calculate_moving_average(df[column_name1], window_size)
    df[column_name2 + '_MA'] = calculate_moving_average(df[column_name2], window_size)

    mean_value1 = df[column_name1].mean()
    std_value1 = df[column_name1].std()
    mean_value2 = df[column_name2].mean()
    std_value2 = df[column_name2].std()

    # Filter out values that are too far from the mean (e.g., within 3 standard deviations)
    filtered_column1 = df[column_name1].apply(lambda x: x if abs(x - mean_value1) <= 3 * std_value1 else np.nan)
    filtered_column2 = df[column_name2].apply(lambda x: x if abs(x - mean_value2) <= 3 * std_value2 else np.nan)

    episode = list(range(1, len(df) + 1)) if use_steps else list(range(len(df)))

    # Align episode list with non-null values in both filtered columns
    aligned_episode1 = [ep for ep, val in zip(episode, filtered_column1) if not np.isnan(val)]
    aligned_episode2 = [ep for ep, val in zip(episode, filtered_column2) if not np.isnan(val)]


    #plt.figure()
    #plt.plot(episode, df[column_name1], label=format_column_name(column_name1),alpha=0.7)
    #plt.plot(episode, df[column_name2], label=format_column_name(column_name2),alpha=0.7)
    plt.figure()
    plt.scatter(aligned_episode1, filtered_column1.dropna(), label=f'Original {format_column_name(column_name1)}', s=10, color='lightgray')
    plt.scatter(aligned_episode2, filtered_column2.dropna(), label=f'Original {format_column_name(column_name2)}', s=10, color='gray')
    plt.plot(episode, df[column_name1 + '_MA'], label=f'Average {format_column_name(column_name1)}')
    plt.plot(episode, df[column_name2 + '_MA'], label=f'Average { format_column_name(column_name2)}')
    plt.xlabel('Episodes')
    plt.title(f'{plot_type} - {format_column_name(column_name1)} and {format_column_name(column_name2)}')
    plt.legend(handlelength=0.5, handleheight=0.3)

    # Create a "Monitor Plots" subfolder within the same directory as the CSV file
    plots_subfolder = os.path.join(os.path.dirname(csv_path), 'Monitor Plots')
    os.makedirs(plots_subfolder, exist_ok=True)


    plt.savefig(os.path.join(plots_subfolder, f'{plot_type}_{column_name1}_{column_name2}_plot.png'), dpi=400)
    plt.close()

# List of columns to generate combined plots for
combined_columns_to_plot = [('total_reward', 'sim_duration'),
                            ('total_reward', 'oc_costs'),
                            ('total_reward', 'weighted_lateness')]
# List of columns to generate plots for
columns_to_plot = ['total_reward', 'sim_duration', 'oc_costs', 'weighted_lateness', 'l']

# Recursive function to process subdirectories and generate plots
def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            subdir = os.path.join(root, dir)
            eval_csv_path = os.path.join(subdir, 'Logs_evaluation.monitor.csv')
            training_csv_path = os.path.join(subdir, 'Logs_training.monitor.csv')

            if os.path.exists(eval_csv_path) and os.path.exists(training_csv_path):

                for column in columns_to_plot:
                    generate_single_plots(eval_csv_path, column, is_evaluation=True)
                    generate_single_plots(training_csv_path, column)

                generate_combined_plots(eval_csv_path,
                                        'total_reward', 'sim_duration', is_evaluation=True)
                generate_combined_plots(eval_csv_path,
                                        'total_reward', 'oc_costs', is_evaluation=True)
                generate_combined_plots(eval_csv_path,
                                        'total_reward', 'weighted_lateness', is_evaluation=True)

                generate_combined_plots(training_csv_path,
                                        'total_reward', 'sim_duration',)
                generate_combined_plots(training_csv_path,
                                        'total_reward', 'oc_costs',)
                generate_combined_plots(training_csv_path,
                                        'total_reward', 'weighted_lateness',)

        for file in files:
            if file.startswith('Logs_evaluation') or file.startswith('Logs_training'):
                csv_path = os.path.join(root, file)
                for column in columns_to_plot:
                    generate_single_plots(csv_path, column)

# Start processing from the base source directory
process_directory(base_source_directory)
