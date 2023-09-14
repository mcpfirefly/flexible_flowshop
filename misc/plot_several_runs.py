import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats

plt.style.use(['science','no-latex','grid'])



# Define your base source directory
window_size = 30
smoothing = False
moving_avg = True
multiple = False
plot_best = True
plot_heuristics = False
colors_heuristics = ["purple", "orange", "green", "gray"]
if smoothing:
    smoothing_factor = 0.99
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
        return 'OCC [$]'
    elif column == 'total_reward' or  column == "r":
        return 'Return'
    elif column == "sim_duration":
        return 'Makespan [h]'
    elif column == "completion_score":
        return "Demand Completion [%]"
    elif column == 'weighted_lateness':
        return 'Total WL [h]'
    elif column == 'l':
        return 'Episode Length [steps]'
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
def process_directory(directory_path,loc_evaluation_files, loc_training_files):
    for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            subdir = os.path.join(root, dir)
            eval_csv_path = os.path.join(subdir, 'Logs_evaluation.monitor.csv')
            train_csv_path = os.path.join(subdir, 'Logs_training.monitor.csv')
            if os.path.exists(eval_csv_path):
                loc_evaluation_files.append(eval_csv_path)
                loc_training_files.append(train_csv_path)

def process_directory_grouped(directory_path,a,b):
    for root, dirs, files in os.walk(directory_path):
        for dir in dirs:
            subdir = os.path.join(root, dir)
            eval_csv_path = os.path.join(subdir, 'Logs_evaluation.monitor.csv')
            train_csv_path = os.path.join(subdir, 'Logs_training.monitor.csv')
            if os.path.exists(eval_csv_path):
                a.append(eval_csv_path)
                b.append(train_csv_path)

def generate_plots_vs_custom_grouped(batch_id, exp_id,csv_path, column_x, column_y, ax, is_evaluation=False):

    df = pd.read_csv(csv_path, header=1)
    if column_x == "completion_score":
        df[column_x] = df[column_x]*100
    elif column_y == "completion_score":
        df[column_y] = df[column_y]*100

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



    if column_x == "completion_score":
        df[column_x] = df[column_x]*100
    elif column_y == "completion_score":
        df[column_y] = df[column_y]*100

    p = calculate_trendline(df[column_x], df[column_y])

    plt.scatter(df[column_x], df[column_y], label=exp_id, s=10, alpha=0.7)
    plt.plot(df[column_x], p(df[column_x]), color="red")


    plt.xlabel('Return')
    plt.ylabel(format_column(column_y))
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    figure = plt.gcf()
    figure.set_size_inches(5,2)
    plt.tight_layout()
    plt.legend()

    plt.savefig(os.path.join(output,
                             f'{exp_id}_Return_vs_{column_y}.png'),
                dpi=400)
    plt.close()




def generate_single_plot(files, column_name, is_evaluation):
    y_values_array = []
    x_values_array = []
    exp_id_array = []
    colors = ["red","blue","green","black","orange","pink"]
    fig, axs = plt.subplots(ncols=len(files))
    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        generate_plots_vs_custom_grouped(batch_id, exp_id, csv_path, reward, column_name, axs[i],is_evaluation)
        axs[i].legend()
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    fig = plt.gcf()
    fig.set_size_inches(6,2)
    fig.tight_layout()
    fig.savefig(os.path.join(output,f'G_{exp_id}_Return_vs_{column_name}.png'),dpi=400)

    fig.savefig(os.path.join(output,f'G_{exp_id}_Return_vs_{column_name}.png'),dpi=400)
    plt.close()
    lens = []
    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        generate_plots_vs_custom(batch_id, exp_id, csv_path, reward, column_name, is_evaluation)

    exp_ids = []
    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        exp_ids.append(exp_id)
        batch_id = get_batch_id(files[0])
        df = pd.read_csv(csv_path, header=1)
        #df, episodes = filter_outliers(df, df[column_name])
        df, episodes = filter_outliers(df, df[column_name])
        y_values = df[column_name]

        if column_name == "completion_score":
            y_values = y_values * 100

        #episodes = list(range(1, len(df) + 1))
        if smoothing:
            y_values = smooth_values(y_values)
            std = np.std(y_values)
        elif moving_avg:
            y_values, std = calculate_moving_average(y_values)
            episodes = list(range(1, len(y_values) + 1))

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

        if smoothing:
            plt.plot(episodes, y_values, label=f'{agent} ({exp_id})',color=colors[i])
        elif moving_avg:
            plt.plot(episodes, y_values, label=f'{agent} ({exp_id})', color=colors[i])

        #plt.plot(episodes, y_values, alpha=0.1, color=colors[i])

        plt.fill_between(episodes, y_values - std, y_values + std, color=colors[i], alpha=0.1)
        y_values_array.append(y_values)
        x_values_array.append(episodes)
        exp_id_array.append(exp_id)
        last_y_values = y_values[int(len(y_values)*0.9):int(len(y_values)*1)]
        min = np.min(last_y_values)
        max = np.max(last_y_values)
        if plot_best and column_name != "total_reward" and column_name != "completion_score":
            plt.plot(episodes, [min] * len(y_values), color=colors[i], linestyle="dashed", linewidth=1.2, alpha = 0.7,label=f"Best Solution {exp_id}")

        lens.append(len(y_values))

    x_ax = range(1,np.max(lens)+1)
    if column_name == "sim_duration":
        plt.gca().set_ylim(bottom=20)
        if plot_heuristics:
            plt.plot(x_ax, [44.01] *np.max(lens), label="Best Solution FIFO", color=colors_heuristics[0], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [34.25] *np.max(lens), label="Best Solution SPT", color=colors_heuristics[1], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [33.55] *np.max(lens), label="Best Solution EDD", color=colors_heuristics[2], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [33.07] *np.max(lens), label="Best Solution SCT", color=colors_heuristics[3], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [26.56]*np.max(lens), label=f'Best Solution Kopanos', color="black", linestyle="dashed",linewidth=1.2)
        else:
            plt.plot(episodes, [26.56] * len(y_values), label=f'Best Solution Kopanos', color="black", linestyle="dashed",
                     linewidth=1.2)

    elif column_name == "oc_costs":
        plt.gca().set_ylim(bottom=60)
        if plot_heuristics:
            plt.plot(x_ax, [87.06] *np.max(lens), label="Best Solution FIFO", color=colors_heuristics[0], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [76.56] *np.max(lens), label="Best Solution SPT", color=colors_heuristics[1], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [74.77] *np.max(lens), label="Best Solution EDD", color=colors_heuristics[2], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [69.89] *np.max(lens), label="Best Solution SCT", color=colors_heuristics[3], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [62.91]*np.max(lens), label=f'Best Solution Kopanos', color="black", linestyle="dashed",linewidth=1.2)
        else:
            plt.plot(episodes, [62.91] *  len(y_values), label=f'Best Solution Kopanos', color="black", linestyle="dashed",
                     linewidth=1.2)
    elif column_name == "weighted_lateness":
        plt.gca().set_ylim(bottom=-10)

        if plot_heuristics:
            plt.plot(x_ax, [1180.45] * np.max(lens), label="Best Solution FIFO", color=colors_heuristics[0], linestyle="dashed",
                     linewidth=1.2)
            plt.plot(x_ax, [627.47] * np.max(lens), label="Best Solution SPT", color=colors_heuristics[1], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [230.75] * np.max(lens), label="Best Solution EDD", color=colors_heuristics[2], linestyle="dashed", linewidth=1.2)
            plt.plot(x_ax, [512.87] * np.max(lens), label="Best Solution SCT", color=colors_heuristics[3], linestyle="dashed", linewidth=1.2)

            plt.plot(x_ax, [19.09]*np.max(lens), label=f'Best Solution Kopanos', color="black", linestyle="dashed",linewidth=1.2)
        else:
            plt.plot(episodes, [19.09] * len(y_values), label=f'Best Solution Kopanos', color="black", linestyle="dashed",linewidth=1.2)

    plt.xlabel('Episodes')
    plt.ylabel(format_column(column_name))
    plot_type = 'Evaluation' if is_evaluation else 'Training'

    min = []
    for i in range(len(x_values_array)):
        min.append(len(x_values_array[i]))
    #plt.xlim([0, np.min(min)])
    figure = plt.gcf()
    figure.set_size_inches(5,3)
    plt.tight_layout()
    legend = plt.legend(bbox_to_anchor=(1.01,0.7), loc='upper left')  # 9 means top center
    # Adjust the font size of the legend
    for text in legend.get_texts():
        text.set_fontsize(10)
    plt.savefig(os.path.join(output, f'{exp_ids[0]}_vs_{exp_ids[1]}_{column_name}.png'), dpi=400)
    plt.close()

def generate_single_plot_grouped(files, column_name, is_evaluation):
    y_values_array = []
    x_values_array = []
    exp_id_array = []
    colors = ["blue","red","green","black","orange","pink"]
    fig, axs = plt.subplots(len(files))
    for i in range(len(files)):
        csv_path = files[i]
        exp_id = get_experiment_id(csv_path)
        batch_id = get_batch_id(files[0])
        #generate_plots_vs_custom_grouped(batch_id, exp_id, csv_path, reward, column_name, axs[i],is_evaluation)
        axs[i].legend()
    plot_type = 'Evaluation' if is_evaluation else 'Training'
    fig = plt.gcf()
    fig.set_size_inches(5,4)
    fig.tight_layout()
    #fig.legend(ncol=len(files), loc='upper center')
    #fig.savefig(os.path.join(output,
    #                         f'G_{exp_id}_Return_vs_{column_name}.png'),
    #            dpi=400)

    fig.savefig(os.path.join(output,
                             f'G_{exp_id}_Return_vs_{column_name}.png'),dpi=400)
    plt.close()


    for i in range(len(files)):
        csv_path = files[i]
        df = pd.read_csv(csv_path, header=1)
        #df, episodes = filter_outliers(df, df[column_name])
        df, episodes = filter_outliers(df, df[column_name])
        y_values = df[column_name]

        if column_name == "completion_score":
            y_values = y_values * 100

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

        min = []
        for i in range(len(x_values_array)):
            min.append(len(x_values_array[i]))
        #plt.set_xlim([0, np.min(min)])

    plt.xlabel('Episodes')
    plt.ylabel(format_column(column_name))
    plot_type = 'Evaluation' if is_evaluation else 'Training'

    figure = plt.gcf()
    figure.set_size_inches(5,4)
    plt.tight_layout()
    plt.legend(ncol=len(files))  # 9 means top center
    plt.savefig(os.path.join(output, f'{exp_id}_{column_name}.png'), dpi=400)
    plt.close()

#####AQUI NO LE CAMBIES A NADA, SOLO VETE UNO POR UNO
reward = "total_reward"

base_source_directory_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Set_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Set_2",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Set_3"]

base_source_directory_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Comparing_Batch1_2\Set_1_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Comparing_Batch1_2\Set_2_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Comparing_Batch1_2\Set_1_2"]

base_source_directory_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_1\Continuous"]


base_source_directory_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Comparing_Batch1_2\Set_2_2",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Comparing_Batch1_2\Set_3_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Comparing_Batch1_2\Set_3_2"]

base_source_directory_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\best_mks",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\best_occ",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\best_wl"]

base_source_directory_ = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_3\Set_1",
                         r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_3\Set_2"]

base_source_directory = [r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_4\Set_1"]


loc_evaluation_files_g = [[] for i in range(len(base_source_directory))]
loc_training_files_g = [[] for i in range(len(base_source_directory))]

for i, folder in enumerate(base_source_directory):
    loc_evaluation_files = []
    loc_training_files = []

    process_directory(folder, loc_evaluation_files, loc_training_files)
    process_directory_grouped(folder, loc_evaluation_files_g[i], loc_training_files_g[i])

    variables = ["total_reward",'sim_duration','oc_costs','weighted_lateness']
    #variables = ["completion_score"]

    base = "C:/Users/INOSIM/OneDrive - INOSIM Consulting GmbH/General/Thesis Overviews - MCPF/03_Others/results"
    output = base + "/plots"

    os.makedirs(output, exist_ok=True)


def plot_grouped(files,is_evaluation):
    colors = ["red", "blue", "green", "black", "orange", "pink"]
    fig, axs = plt.subplots(len(files))

    y_values_array = []
    x_values_array = []
    exp_id_array = []

    for j in range(len(files)):
        ax = axs[j]
        for i in range(len(files[j])):
            column_name = "total_reward"
            csv_path = files[j][i]
            exp_id = get_experiment_id(csv_path)
            df = pd.read_csv(csv_path, header=1)
            #df, episodes = filter_outliers(df, df[column_name])
            df, episodes = filter_outliers(df, df[column_name])
            y_values = df[column_name]

            if column_name == "completion_score":
                y_values = y_values * 100

            #episodes = list(range(1, len(df) + 1))
            if smoothing:
                y_values_smoothed = smooth_values(y_values)
                std = np.std(y_values)
            elif moving_avg:
                y_values_ma, std = calculate_moving_average(y_values)
                episodes = list(range(1, len(y_values) + 1))

            ax.plot(episodes, y_values, alpha=0.1, color=colors[i])
            if smoothing:
                ax.plot(episodes, y_values_smoothed, label=f'PPO ({exp_id})',color=colors[i])
            elif moving_avg:
                ax.plot(episodes, y_values_ma, label=f'PPO ({exp_id})', color=colors[i])
            #plt.fill_between(episodes, y_values - std, y_values + std, color='gray', alpha=0.2)
            y_values_array.append(y_values)
            x_values_array.append(episodes)
            exp_id_array.append(exp_id)

            min = []
            for i in range(len(x_values_array)):
                min.append(len(x_values_array[i]))
                #ax.set_xlim([0, np.min(min)])
                #ax.legend(bbox_to_anchor=(1.15,1), loc='upper right', ncol=1)
            ax.set_ylabel(format_column(column_name))

    plt.xlabel('Episodes')
    plot_type = 'Evaluation' if is_evaluation else 'Training'

    fig.set_size_inches(5,4)
    #fig.tight_layout()
    fig.savefig(os.path.join(output, f'{exp_id_array[0]}-{exp_id_array[-1]}_{column_name}.png'), dpi=400)
    plt.close()


for i in range(len(loc_evaluation_files_g)):
    for j in range(len(variables)):
        #plot_grouped(loc_evaluation_files_g,is_evaluation=True)
        generate_single_plot(loc_evaluation_files_g[i], variables[j], is_evaluation=True)
        #generate_single_plot(loc_training_files, variables[i], is_evaluation=False)