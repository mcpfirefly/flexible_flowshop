
import matplotlib.pyplot as plt
import pickle, os
import tensorflow as tf
import numpy as np
# Function to generate and save plots
smoothing = False
moving_average = True
moving_average_period = 12

def generate_and_save_plots(event_file, log_path):

    # Load TensorBoard log file
    store_variables = {}

    # Load the event file
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                store_variables.setdefault(value.tag, []).append((event.wall_time, value.simple_value))


    # Smoothing factor
    smoothing_factor = 0.8
    base_output_directory = log_path + "/Event Plots"
    # Create the base output directory if it doesn't exist
    os.makedirs(base_output_directory, exist_ok=True)
    # Create separate plots for each key in the dictionary
    steps = event.step

    for key, value in store_variables.items():
        filename_safe_key = key.replace('/', '-')
        # Generate the modified label for plotting
        modified_label = key.split('/')[-1].title()
        # Remove underscores and replace them with spaces
        modified_label = modified_label.replace('_', ' ')

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


        if smoothing:
            # Apply exponential moving average (EMA) smoothing to y-values
            smoothed_y_values = [y_values[0]]
            for y in y_values[1:]:
                smoothed_y = smoothing_factor * smoothed_y_values[-1] + (1 - smoothing_factor) * y
                smoothed_y_values.append(smoothed_y)

            # Create a new figure
            plt.figure()

            # Plot the original data points in a lighter color
            plt.scatter(x_values, y_values, color='gray', alpha=0.4, label=f'{modified_label}')

            # Plot the smoothed data
            plt.plot(x_values, smoothed_y_values, label=f'{modified_label} (smoothed: {smoothing_factor})')

            # Set title and labels
            plt.title(f'{modified_label}')
            plt.xlabel('Timesteps')
            plt.ylabel(f'{modified_label}')

            # Add legend
            plt.legend()

            # Save the plot to the specified location
            plot_filename = f'{filename_safe_key}_plot.png'  # You can use any desired file format (e.g., .eps, .jpg, etc.)
            save_path = os.path.join(base_output_directory, plot_filename)
            plt.savefig(save_path, dpi=400)
            plt.close()

        elif moving_average:
            # Calculate the moving average
            moving_averages = np.convolve(y_values, np.ones(moving_average_period) / moving_average_period,
                                          mode='valid')

            # Create a new figure
            plt.figure()

            # Plot the original data points in a lighter color
            plt.scatter(x_values, y_values, color='gray', alpha=0.4, label="Samples")

            # Plot the moving average
            plt.plot(x_values[moving_average_period - 1:], moving_averages, label="Average")

            # Set title and labels
            plt.title(f'{modified_label}')
            plt.xlabel('Timesteps')
            plt.ylabel(f'{modified_label}')

            # Add legend
            plt.legend()

            # Save the plot to the specified location
            plot_filename = f'{filename_safe_key}_plot.png'  # You can use any desired file format (e.g., .eps, .jpg, etc.)
            save_path = os.path.join(base_output_directory, plot_filename)
            plt.savefig(save_path, dpi=400)
            plt.close()

        # Save data dictionary as pickle
        pickle_filename = 'stored_variables.pkl'
        save_pickle = os.path.join(log_path, pickle_filename)

        with open(save_pickle, 'wb') as save_pickle:
            pickle.dump(store_variables, save_pickle)


# Recursive function to process subdirectories and generate plots
def process_directory_tensorboard(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file_path = os.path.join(root, file)
                generate_and_save_plots(event_file_path, root)
