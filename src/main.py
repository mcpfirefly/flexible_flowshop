from experiments import experiments
from flexible_flow_shop.resources.functions.scheduling_functions import StudyCaseFilename, KopanosStudyCase, GeneralStudyCase
from omegaconf import DictConfig
import numpy as np
import os
import pandas as pd
import hydra, datetime

experiment_date = datetime.datetime.now().strftime("%Y-%m-%d")
experiment_time = datetime.datetime.now().strftime("%H-%M-%S")
experiment_folder = "{}/{}".format(experiment_date, experiment_time)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def config_read(cfg : DictConfig) -> DictConfig:
    global config
    global cwd

    cwd = os.getcwd()
    for key, value in cfg.params[0].items():
        if value == 'None':
            cfg.params[0][key] = None
    config = cfg

config_read()

# Access the variables with the desired format
seed = config.params[0].seed
N_TIMESTEPS = config.params[0].N_TIMESTEPS
masking = config.params[0].masking
reward = config.params[0].reward
buffer_usage = config.params[0].buffer_usage
time_in_buffer_custom = config.params[0].time_in_buffer_custom
workbook = config.params[0].workbook
obs_size = config.params[0].obs_size
action_space = config.params[0].action_space
solution_hints = config.params[0].solution_hints
kopanos_products = config.params[0].kopanos_products
generate_heuristic_schedules = config.params[0].generate_heuristic_schedules
recreate_solution = config.params[0].recreate_solution
test = config.params[0].test
np.random.seed(seed)
filename = StudyCaseFilename(workbook)

if workbook == "kopanos" and recreate_solution == None:
    GENERAL_DATA = KopanosStudyCase(kopanos_products, action_space, filename, recreate_solution, solution_hints)
    SOLUTION = None
if workbook == "general" and recreate_solution == None:
    GENERAL_DATA = GeneralStudyCase(filename)
    SOLUTION = None
if recreate_solution != None or (
        workbook == "kopanos" and action_space == "discrete_probs" and kopanos_products == 30 and solution_hints != None):
    FIFO = False
    buffer_usage = "no_buffers"
    workbook = "kopanos"
    filename = StudyCaseFilename(workbook)
    kopanos_products = 30
    GENERAL_DATA = pd.read_excel(filename, sheet_name="general_data_30")
    SOLUTION = KopanosStudyCase(kopanos_products, action_space, filename, recreate_solution, solution_hints)

if __name__ == '__main__':
    if test == "ppo_manual":
        experiments.PPO_Manual_Parameters().PPO_Manual_Parameters_run()
    elif test == "simple_run":
        experiments.PPO_Simple_Run().PPO_Simple_Run_run()
    elif test == "ppo_optuna":
        experiments.PPO_Optuna().PPO_Optuna_run()
    elif test == "ppo_test_trained":
        experiments.PPO_Test_Trained().PPO_Test_Trained_Run()

