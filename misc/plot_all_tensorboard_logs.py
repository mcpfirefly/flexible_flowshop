
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use(['science','no-latex','grid'])
import re
import pickle, os
import tensorflow as tf
import numpy as np
# Function to generate and save plots
smoothing = True
moving_average = False
moving_average_period = 12

def moving_average(vector, window):
    vector = pd.DataFrame(vector)
    rolling_mean = vector[0].rolling(window=window, min_periods=1).mean()
    rolling_std = vector[0].rolling(window=window, min_periods=1).std()
    return rolling_mean, rolling_std

def calculate_trendline(x,y):
    # calculate equation for trendline
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p


def generate_and_save_plots(event_file, log_path, extracted_string):
    agent = "PPO Agent"

    if "30" in extracted_string or "27" in extracted_string or "28" in extracted_string:
        agent = "SAC Agent"

    # Load TensorBoard log file
    store_variables = {}

    # Load the event file
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                store_variables.setdefault(value.tag, []).append((event.wall_time, value.simple_value))


    # Smoothing factor
    smoothing_factor = 0.999
    base = "C:/Users/INOSIM/OneDrive - INOSIM Consulting GmbH/General/Thesis Overviews - MCPF/03_Others/results"
    base_output_directory = base + "/logs_plots"
    base_output_directory2 = base + "/logs_plots_svg"
    # Create the base output directory if it doesn't exist
    os.makedirs(base_output_directory, exist_ok=True)
    os.makedirs(base_output_directory2, exist_ok=True)
    # Create separate plots for each key in the dictionary
    steps = event.step

    for key, value in store_variables.items():
        if ("eval" in key) and ("total_waiting" not in key) and ("mean" not in key) and ("times" not in key):
            filename_safe_key = key.split('/')[-1]
            # Generate the modified label for plotting
            modified_label = key.split('/')[-1].title()
            # Remove underscores and replace them with spaces
            modified_label = modified_label.replace('_', ' ')

            if key == "advanced_eval/oc_costs":
                modified_label = "OCC [$]"
            elif key == "advanced_eval/weighted_lateness":
                modified_label = "Total WL [h]"
            elif key == "advanced_eval/makespan":
                modified_label = "Makespan [h]"


            num_steps = len(value)
            step_size = steps / (num_steps - 1)
            # Generate x-values with steps of 1000
            x_values = [i * step_size for i in range(num_steps)]

            # Replace x-values in each tuple
            updated_value = [(x, y) for x, (_, y) in zip(x_values, value)]

            # Update the value in data_dict
            store_variables[key] = updated_value

            # Extract x and y values from the list of tuples
            _, y_values = zip(*value)

            #################
            # Plot the original data points in a lighter color

            #plt.scatter(x_values, y_values, alpha=0.6, label=f'ID33')

            moving_avg_mean, moving_avg_std = moving_average(y_values,moving_average_period)
            a=False
            if a:
                moving_avg_mean = moving_avg_mean[moving_average_period-1:]
                moving_avg_std = moving_avg_std[moving_average_period - 1:]
                x_values = x_values[moving_average_period - 1:]

            plt.plot(x_values, moving_avg_mean, color="blue", label=f'ID33')

            #plt.fill_between(x_values, moving_avg_mean - moving_avg_std, moving_avg_mean + moving_avg_std, color='gray', alpha=0.2, label = (r'STD ($\sigma$)'))
            # Set title and labels
            #plt.title(f'{exp_id}')
            plt.xlabel('Learning Steps')
            plt.ylabel(f'{modified_label}')

            # Add legend
            plt.legend()

            # Save the plot to the specified location
            plot_filename = f'{extracted_string}_{filename_safe_key}.png'  # You can use any desiblue file format (e.g., .svg, .jpg, etc.)
            plot_filename2 = f'{extracted_string}_{filename_safe_key}.svg'
            save_path = os.path.join(base_output_directory, plot_filename)
            save_path2 = os.path.join(base_output_directory2, plot_filename2)

            figure = plt.gcf()
            figure.set_size_inches(11, 5)
            plt.savefig(save_path, dpi=400)
            plt.savefig(save_path2, dpi=400, format='svg',transparent=True)
            plt.close()

            # Save data dictionary as pickle
            pickle_filename = 'stoblue_variables.pkl'
            save_pickle = os.path.join(log_path, pickle_filename)

            with open(save_pickle, 'wb') as save_pickle:
                pickle.dump(store_variables, save_pickle)

    keys = ["advanced_eval/makespan", "advanced_eval/oc_costs", "advanced_eval/weighted_lateness"]
    for key in keys:

        filename_safe_key = key.replace('/', '-')
        # Generate the modified label for plotting
        modified_label = key.split('/')[-1].title()
        # Remove underscores and replace them with spaces
        modified_label = modified_label.replace('_', ' ')

        if key == "advanced_eval/oc_costs":
            modified_label = "OCC [$]"
        elif key == "advanced_eval/weighted_lateness":
            modified_label = "Total WL [h]"
        elif key == "advanced_eval/makespan":
            modified_label = "Makespan [h]"

        _, y_values = zip(*store_variables[key])
        if agent == "PPO Agent":
            _, x_values = zip(*store_variables["eval/mean_reward"])
        else:
            _, x_values = zip(*store_variables["performance/eval_return"])
        y_values = list(y_values)
        x_values = list(x_values)
        idx = np.argmin(y_values)
        x_values.pop(idx)
        y_values.pop(idx)
        if smoothing:
            # Apply exponential moving average (EMA) smoothing to y-values
            smoothed_y_values = [y_values[0]]
            for y in y_values[1:]:
                smoothed_y = smoothing_factor * smoothed_y_values[-1] + (1 - smoothing_factor) * y
                smoothed_y_values.append(smoothed_y)

            # Create a new figure
            plt.figure()

            # Plot the original data points in a lighter color
            plt.scatter(x_values, y_values, alpha=0.6, label=f'ID33')
            p = calculate_trendline(x_values, y_values)
            plt.plot(x_values, p(x_values), color="blue")
            # Plot the smoothed data
            plt.plot(x_values, smoothed_y_values, label=f'ID33 (smoothed: {smoothing_factor})')

            # Set title and labels
            #plt.title(f'{exp_id}')
            plt.xlabel('Return')
            plt.ylabel(f'{modified_label}')

            # Add legend
            plt.legend()

            # Save the plot to the specified location
            plot_filename = f'{extracted_string}_{filename_safe_key}.png'  # You can use any desiblue file format (e.g., .svg, .jpg, etc.)
            plot_filename2 = f'{extracted_string}_{filename_safe_key}.svg'

            save_path = os.path.join(base_output_directory, plot_filename)
            save_path2 = os.path.join(base_output_directory2, plot_filename2)
            figure = plt.gcf()
            figure.set_size_inches(11, 5)
            plt.savefig(save_path, dpi=400)
            plt.savefig(save_path2, dpi=400, format='svg',transparent=True)
            plt.close()

        elif moving_average:
            # Calculate the moving average

            # Create a new figure
            plt.figure()

            idx = np.argmin(x_values)
            x_values.pop(idx)
            y_values.pop(idx)
            # Plot the original data points in a lighter color
            plt.scatter(x_values, y_values, alpha=0.6, label=agent)
            p = calculate_trendline(x_values, y_values)
            plt.plot(x_values, p(x_values), color="blue")
            # Set title and labels
            #plt.title(f'{exp_id}')
            plt.xlabel('Return')
            plt.ylabel(f'{modified_label}')

            # Add legend
            plt.legend()

            # Save the plot to the specified location
            plot_filename = f'{extracted_string}_ return_vs_{filename_safe_key}.png'  # You can use any desiblue file format (e.g., .svg, .jpg, etc.)
            plot_filename2 = f'{extracted_string}_ return_vs_{filename_safe_key}.svg'
            save_path = os.path.join(base_output_directory, plot_filename)
            save_path2 = os.path.join(base_output_directory2, plot_filename2)

            figure = plt.gcf()
            figure.set_size_inches(11, 5)
            plt.savefig(save_path, dpi=400)
            plt.savefig(save_path2, dpi=400, format='svg',transparent=True)
            plt.close()

    if agent == "PPO Agent":
        key = "eval/mean_reward"
    else:
        key = "performance/eval_return"



    filename_safe_key = key.replace('/', '-')
    # Generate the modified label for plotting
    modified_label = "Mean Return"

    num_steps = len(store_variables[key])
    step_size = steps / (num_steps - 1)
    # Generate x-values with steps of 1000
    x_values = [i * step_size for i in range(num_steps)]

    if agent == "PPO Agent":
        _, y_values = zip(*store_variables["eval/mean_reward"])
    else:
        _, y_values = zip(*store_variables["performance/eval_return"])
    # Create a new figure
    y_values = list(y_values)
    x_values = list(x_values)
    idx = np.argmin(y_values)
    x_values.pop(idx)
    y_values.pop(idx)
    idx = np.argmin(y_values)
    x_values.pop(idx)
    y_values.pop(idx)

    plt.figure()
    plt.plot(x_values, y_values, label=f'ID33')

    # Set title and labels
    #plt.title(f'{exp_id}')
    plt.xlabel('Learning Steps')

    plt.ylabel(f'{modified_label}')

    # Add legend
    plt.legend()

    # Save the plot to the specified location
    plot_filename = f'{extracted_string}_{filename_safe_key}.png'  # You can use any desiblue file format (e.g., .svg, .jpg, etc.)
    plot_filename2 = f'{extracted_string}_{filename_safe_key}.svg'
    save_path = os.path.join(base_output_directory, plot_filename)
    save_path2 = os.path.join(base_output_directory2, plot_filename2)

    figure = plt.gcf()
    figure.set_size_inches(11, 5)
    plt.savefig(save_path, dpi=400)
    plt.savefig(save_path2, dpi=400, format='svg',transparent=True)
    plt.close()

# Recursive function to process subdirectories and generate plots
def process_directory_tensorboard(directory_path, exp_id):
    i=0
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file_path = os.path.join(root, file)

                # Define a regular expression pattern to match the level (e.g., ID01, ID02, etc.)
                pattern = r'\\(ID\d+)'

                # Search for the pattern in the path
                match = re.search(pattern, root)

                # Check if a match was found
                if match:
                    extracted_string = match.group(1)  # Get the first matching group (ID01 in this case)
                    print(extracted_string)



                generate_and_save_plots(event_file_path, root, extracted_string)


if __name__ == "__main__":
    path = r"C:\Users\INOSIM\OneDrive - INOSIM Consulting GmbH\General\Thesis Overviews - MCPF\03_Others\results\02_RL\Batch_8\ID33"
    process_directory_tensorboard(path, exp_id=0)