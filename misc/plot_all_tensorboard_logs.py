
import matplotlib.pyplot as plt
import pickle, os
import tensorflow as tf

# Function to generate and save plots
def generate_and_save_plots(event_file, log_path):

    # Load TensorBoard log file
    store_variables = {}

    # Load the event file
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.HasField('simple_value'):
                store_variables.setdefault(value.tag, []).append((event.wall_time, value.simple_value))

    # Save data dictionary as pickle

    pickle_filename = 'stored_variables.pkl'
    save_pickle = os.path.join(log_path, pickle_filename)

    with open(save_pickle, 'wb') as save_pickle:
        pickle.dump(store_variables, save_pickle)


    # Smoothing factor
    smoothing_factor = 0.8
    base_output_directory = log_path + "/plots"
    # Create the base output directory if it doesn't exist
    os.makedirs(base_output_directory, exist_ok=True)
    # Create separate plots for each key in the dictionary
    for key, value in store_variables.items():
        filename_safe_key = key.replace('/', '-')

        # Extract x and y values from the list of tuples
        x_values, y_values = zip(*value)

        # Apply exponential moving average (EMA) smoothing to y-values
        smoothed_y_values = [y_values[0]]
        for y in y_values[1:]:
            smoothed_y = smoothing_factor * smoothed_y_values[-1] + (1 - smoothing_factor) * y
            smoothed_y_values.append(smoothed_y)

        # Create a new figure
        plt.figure()

        # Plot the original data points in a lighter color
        plt.scatter(x_values, y_values, color='gray', alpha=0.4, label=f'{filename_safe_key} (original)')

        # Plot the smoothed data
        plt.plot(x_values, smoothed_y_values, label=f'{filename_safe_key} (smoothed)')

        # Set title and labels
        plt.title(f'{filename_safe_key} with Smoothing')
        plt.xlabel('X values')
        plt.ylabel('Y values')

        # Add legend
        plt.legend()

        # Save the plot to the specified location
        plot_filename = f'{filename_safe_key}_plot.png'  # You can use any desired file format (e.g., .eps, .jpg, etc.)
        save_path = os.path.join(base_output_directory, plot_filename)
        plt.savefig(save_path, dpi=400)
        plt.close()


# Recursive function to process subdirectories and generate plots
def process_directory(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file_path = os.path.join(root, file)
                generate_and_save_plots(event_file_path, root)

if __name__ == "__main__":
    # Define your base source directory
    base_source_directory = 'C:/Users/INOSIM/PycharmProjects/flexible_flowshop/src/experiments/results'

    # Start processing from the base source directory
    process_directory(base_source_directory)