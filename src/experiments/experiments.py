from src.experiments.rl_algorithms.unstable_baselines.baselines.redq.trainer import REDQTrainer
from src.experiments.rl_algorithms.unstable_baselines.common.logger import Logger
from src.experiments.rl_algorithms.unstable_baselines.baselines.sac.trainer import SACTrainer
from src.experiments.rl_algorithms.unstable_baselines.baselines.sac.agent import SACAgent
from src.experiments.rl_algorithms.unstable_baselines.common.util import set_device_and_logger, set_global_seed
from src.experiments.rl_algorithms.unstable_baselines.common.buffer import ReplayBuffer

from src.experiments.rl_algorithms.unstable_baselines.baselines.redq.agent import REDQAgent
from src.flexible_flow_shop.resources.functions.main_wrapper import MakeEnvironment
from src.flexible_flow_shop.environment import flexible_flow_shop
from src.experiments.rl_algorithms.maskable_ppo import MaskablePPO
from src.experiments.rl_algorithms.ppo import PPO
from src.flexible_flow_shop.resources.functions.custom_wrappers import (
    CustomMaskableEvalCallback as MaskableEvalCallback,
)
from src.flexible_flow_shop.resources.functions.custom_wrappers import (
    CustomEvalCallback as EvalCallback,
)
from src.flexible_flow_shop.resources.functions.scheduling_functions import (
    get_action_heuristics,
    GenerateHeuristicResultsFiles,
)

from src.custom_plotters.raincloud_plotter.raincloud_plotter import raincloud_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from src.experiments.rl_algorithms.unstable_baselines.baselines.sac.configs.sac_discrete_args import *
from src.experiments.rl_algorithms.unstable_baselines.baselines.redq.redq_args import *
import pandas as pd
import numpy as np
import optuna, tempfile, gym, torch, datetime, os
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import torch.nn as nn
from typing import Dict, Any, Union, Callable
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer

def make_environment(study, suffix):
    return MakeEnvironment(
        seed=study.seed,
        env=flexible_flow_shop(study),
        reward=study.reward,
        masking=study.masking,
        norm_obs=True,
        norm_rew=True,
        action_mode=study.action_space,
        obs_size=study.obs_size,
        buffer_usage=study.buffer_usage,
        log_path=(study.log_path + suffix),
        heuristics_policy_rl=study.heuristics_policy_rl,
    )


##### A CLASS FOR EACH DIFFERENT TEST
class PPO_Optuna:
    def __init__(self, study):
        self.DEVICE = torch.device("cpu")
        self.initialize_global_variables(study)
        self.N_TRIALS = self.optuna_trials
        self.N_STARTUP_TRIALS = 1
        self.N_EVALUATIONS = self.N_TIMESTEPS / 1000  # evaluations per trial
        self.EVAL_FREQ = int(self.N_TIMESTEPS / self.N_EVALUATIONS)
        self.N_EVAL_EPISODES = 10
        self.optuna_log_path = "outputs/{}/{}/Training/Optuna/Logs".format(
            self.experiment_folder, self.test
        )

    def initialize_global_variables(self, study):
        self.N_TIMESTEPS = study.N_TIMESTEPS  # for every trial
        self.ORDERS = study.ORDERS
        self.CHANGEOVER = study.CHANGEOVER
        self.optuna_trials = study.optuna_trials
        self.recreate_solution = study.recreate_solution
        self.generate_heuristic_schedules = study.generate_heuristic_schedules
        self.test = study.test
        self.experiment_folder = study.experiment_folder
        self.seed = study.seed
        self.masking = study.masking
        self.solution_hints = study.solution_hints
        self.reward = study.reward
        self.obs_size = study.obs_size
        self.action_space = study.action_space
        self.buffer_usage = study.buffer_usage
        self.experiment_date = study.experiment_date
        self.experiment_time = study.experiment_time
        self.config = study.config
        self.log_path = study.log_path
        self.best_model_save_path = study.best_model_save_path
        self.train_env = make_environment(study, "_training")
        self.eval_env = make_environment(study, "_evaluation")

    def linear_schedule(
        self, initial_value: Union[float, str]
    ) -> Callable[[float], float]:
        if isinstance(initial_value, str):
            initial_value = float(initial_value)

        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func

    def sample_ppo_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
        gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
        n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)
        learning_rate = trial.suggest_float("lr", 1e-5, 0.003, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
        lr_schedule = trial.suggest_categorical("lr_schedule", ["constant", "linear"])
        ortho_init = trial.suggest_categorical("ortho_init", [False, True])
        net_arch = trial.suggest_categorical("net_arch", ["a", "b", "c", "d"])
        activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

        trial.set_user_attr("gamma_", gamma)
        trial.set_user_attr("gae_lambda_", gae_lambda)
        trial.set_user_attr("n_steps", n_steps)

        if net_arch == "a":
            net_arch = [64, 64, 64, 64, 64, 64]
        if net_arch == "b":
            net_arch = [128, 64, 128, 64]
        if net_arch == "c":
            net_arch = [128, 128, 128, 128, 128, 128]
        if net_arch == "d":
            net_arch = [256, 128, 256]
        if net_arch == "e":
            net_arch = [256, 256]

        if lr_schedule == "linear":
            learning_rate = self.linear_schedule(learning_rate)

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "ortho_init": ortho_init,
            },
        }

    class MaskableTrialEvalCallback(MaskableEvalCallback):
        """Callback used for evaluating and reporting a trial."""

        def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int,
            eval_freq: int,
            verbose: int = 1,
            best_model_save_path_in: str = "input path - best_model_save_path",
            log_path_in: str = "input path - log_path ",
        ):
            super().__init__(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                verbose=verbose,
                best_model_save_path=best_model_save_path_in,
                log_path=log_path_in,
            )
            self.trial = trial
            self.eval_idx = 0
            self.is_pruned = False

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                super()._on_step()
                self.eval_idx += 1
                self.trial.report(self.last_mean_reward, self.eval_idx)
                print("self.last_mean_reward: ", self.last_mean_reward)
                # Prune trial if need
                if self.trial.should_prune():
                    # logging.debug("TrialEvalCallback___on_step2")
                    self.is_pruned = True
                    print("Trial pruned!")
                    return False
            # logging.debug("TrialEvalCallback___on_step3")
            # print("Trial not pruned!")
            return True

    class TrialEvalCallback(EvalCallback):
        """Callback used for evaluating and reporting a trial."""

        def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int,
            eval_freq: int,
            deterministic: bool = True,
            verbose: int = 1,
            best_model_save_path_in: str = "input path - best_model_save_path",
            log_path_in: str = "input path - log_path ",
        ):
            super().__init__(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                deterministic=deterministic,
                verbose=verbose,
                best_model_save_path=best_model_save_path_in,
                log_path=log_path_in,
            )
            self.trial = trial
            self.eval_idx = 0
            self.is_pruned = False

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                super()._on_step()
                self.eval_idx += 1
                self.trial.report(self.last_mean_reward, self.eval_idx)
                print("self.last_mean_reward: ", self.last_mean_reward)
                # Prune trial if need
                if self.trial.should_prune():
                    self.is_pruned = True
                    print("Trial pruned!")
                    return False
            return True

    def objective(self, trial: optuna.Trial) -> float:
        DEFAULT_HYPERPARAMS = {
            "policy": "MlpPolicy",
            "env": self.train_env,
            "verbose": 2,
            "seed": self.seed,
            "tensorboard_log": self.optuna_log_path,
        }
        kwargs = DEFAULT_HYPERPARAMS.copy()
        # Sample hyperparameters
        kwargs.update(self.sample_ppo_hyperparams(trial))
        # Create the RL model
        if self.masking and self.action_space == "discrete":
            model = MaskablePPO(**kwargs)
            eval_callback = self.MaskableTrialEvalCallback(
                self.eval_env,
                trial,
                n_eval_episodes=self.N_EVAL_EPISODES,
                eval_freq=self.EVAL_FREQ,
                best_model_save_path_in=self.best_model_save_path,
                log_path_in=self.optuna_log_path,
            )
        else:
            model = PPO(**kwargs)
            eval_callback = self.TrialEvalCallback(
                self.eval_env,
                trial,
                n_eval_episodes=self.N_EVAL_EPISODES,
                eval_freq=self.EVAL_FREQ,
                deterministic=False,
                best_model_save_path_in=self.best_model_save_path,
                log_path_in=self.optuna_log_path,
            )
        # new trial variables:
        print("new trial, new variables")
        print("Model created, Callback created")
        nan_encountered = False
        try:
            iteration_no = 1  # initial iteration number of saving/ loading runs
            save_after_n_steps = self.N_TIMESTEPS
            total_iterations = int(
                self.N_TIMESTEPS / save_after_n_steps
            )  # total number of iterations to be saved/ loaded

            # check if total_iterations is zero, set to 1 if
            if total_iterations == 0:
                total_iterations = 1

            steps_so_far = 0  # steps so far in trained in this test
            steps_per_iteration = (
                self.N_TIMESTEPS / total_iterations
            )  # steps to be trained per iteration of saving and loading

            model.learn(
                steps_so_far + steps_per_iteration,
                callback=eval_callback,
                reset_num_timesteps=False,
            )
            model.save(self.best_model_save_path + "model_save")
            model.save(self.best_model_save_path + "model_save_" + str(steps_so_far))

            steps_so_far = steps_so_far + steps_per_iteration
            iteration_no = iteration_no + 1

            while iteration_no <= total_iterations:
                print("start iteration_no: " + str(iteration_no))
                print("steps_so_far: " + str(steps_so_far))
                print("steps_per_iteration: " + str(steps_per_iteration))

                del model
                if self.action_space == "discrete" and self.masking:
                    model = MaskablePPO.load(self.best_model_save_path + "model_save")
                else:
                    model = PPO.load(self.best_model_save_path + "model_save")
                print("loaded model")
                model.set_env(self.train_env)
                print("set env model")

                model.env.reset()
                print("reseted")
                print("start learning")
                model.learn(
                    steps_per_iteration,
                    callback=eval_callback,
                    reset_num_timesteps=False,
                )
                model.save(self.best_model_save_path + "model_save")
                model.save(
                    self.best_model_save_path + "model_save_" + str(steps_so_far)
                )
                steps_so_far = steps_so_far + steps_per_iteration
                iteration_no = iteration_no + 1

                print("Model learning finished")

        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
        finally:
            # Free memory
            print("Close Model and Env!")

            model.env.close()
            self.eval_env.close()

        if nan_encountered:
            # logging.debug("objective___nan")
            print("nan encountered")
            return float("nan")

            # logging.debug("objective___no nan")
        if eval_callback.is_pruned:
            # logging.debug("objective___is_pruned")
            raise optuna.exceptions.TrialPruned()

            # logging.debug("objective___last_mean_reward")
        ret_val = eval_callback.last_mean_reward
        # logging.debug("objective___last_mean_reward = " + str(ret_val))
        print("objective function return value: ", ret_val)
        return ret_val
        {}

    def PPO_Optuna_run(self):
        torch.set_num_threads(1)

        sampler = TPESampler(n_startup_trials=self.N_STARTUP_TRIALS, seed=self.seed)
        pruner = MedianPruner(
            n_startup_trials=self.N_STARTUP_TRIALS, n_warmup_steps=self.N_TIMESTEPS // 3
        )

        study = optuna.create_study(
            storage="sqlite:///outputs/{}/Optuna_Trial.db".format(
                self.experiment_folder
            ),
            study_name="Optuna_Trial",
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
        )

        print("created optuna study")
        print("start study optimize")
        study.optimize(self.objective, n_trials=self.N_TRIALS)
        print("study optimize finished")

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        with open("best_trial.txt", "w") as f:
            print("Best trial:", file=f)
            print("  Value: ", trial.value, file=f)
            print("  Params: ", file=f)
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value), file=f)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print("    {}: {}".format(key, value))


class PPO_Manual_Parameters:
    def __init__(self, study):
        self.initialize_global_variables(study)
        self.N_EVALUATIONS = self.N_TIMESTEPS / 1000  # evaluations
        self.EVAL_FREQ = int(self.N_TIMESTEPS / self.N_EVALUATIONS)
        self.N_EVAL_EPISODES = 10

    def initialize_global_variables(self, study):
        self.N_TIMESTEPS = study.N_TIMESTEPS  # for every trial
        self.ORDERS = study.ORDERS
        self.CHANGEOVER = study.CHANGEOVER
        self.recreate_solution = study.recreate_solution
        self.generate_heuristic_schedules = study.generate_heuristic_schedules
        self.test = study.test
        self.experiment_folder = study.experiment_folder
        self.seed = study.seed
        self.masking = study.masking
        self.solution_hints = study.solution_hints
        self.reward = study.reward
        self.obs_size = study.obs_size
        self.action_space = study.action_space
        self.buffer_usage = study.buffer_usage
        self.experiment_date = study.experiment_date
        self.experiment_time = study.experiment_time
        self.config = study.config
        self.log_path = study.log_path
        self.best_model_save_path = study.best_model_save_path
        self.train_env = make_environment(study, "_training")
        self.eval_env = make_environment(study, "_evaluation")

    def linear_schedule(
        self, initial_value: Union[float, str]
    ) -> Callable[[float], float]:
        if isinstance(initial_value, str):
            initial_value = float(initial_value)

        def func(progress_remaining: float) -> float:
            return progress_remaining * initial_value

        return func

    def ppo_hyperparams_discrete_probs(self) -> Dict[str, Any]:
        gamma = 0.9574554332294194
        max_grad_norm = 0.3178814931041116
        gae_lambda = 0.8170314419786968
        n_steps = 2**6
        learning_rate = 0.0021724161836662818
        ent_coef = 0.0321016635395011
        lr_schedule = "linear"
        ortho_init = True
        net_arch = [128, 64, 128, 64]
        activation_fn = nn.ReLU

        if lr_schedule == "linear":
            learning_rate = self.linear_schedule(learning_rate)

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "ortho_init": ortho_init,
            },
        }

    def ppo_hyperparams_discrete(self) -> Dict[str, Any]:
        gamma = 0.9966873310693646
        max_grad_norm = 0.35270779889789233
        gae_lambda = 0.9721977226163961
        n_steps = 2**4
        learning_rate = 2.1120967608851453e-05
        ent_coef = 0.003309847983555168
        lr_schedule = "linear"
        ortho_init = False
        net_arch = [128, 128, 128, 128, 128, 128]
        activation_fn = nn.Tanh

        if lr_schedule == "linear":
            learning_rate = self.linear_schedule(learning_rate)

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "ortho_init": ortho_init,
            },
        }

    def ppo_hyperparams_continuous(self) -> Dict[str, Any]:
        gamma = 0.9822742318642296
        max_grad_norm = 1.3746408813884265
        gae_lambda = 0.9885707604345161
        n_steps = 2**8
        learning_rate = 0.0006070259771275158
        ent_coef = 4.93550898574578e-06
        lr_schedule = "linear"
        ortho_init = False
        net_arch = [128, 64, 128, 64]
        activation_fn = nn.Tanh

        if lr_schedule == "linear":
            learning_rate = self.linear_schedule(learning_rate)

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "policy_kwargs": {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "ortho_init": ortho_init,
            },
        }

    def PPO_Manual_Parameters_run(self):
        policy_kwargs = {"net_arch": [64, 64, 64, 64]}

        DEFAULT_HYPERPARAMS = {
            "policy": "MlpPolicy",
            "env": self.train_env,
            "verbose": 2,
            "seed": self.seed,
            "tensorboard_log": self.log_path,
            "policy_kwargs": policy_kwargs,
        }
        kwargs = DEFAULT_HYPERPARAMS.copy()
        if self.action_space == "discrete_probs" and self.solution_hints == "kopanos":
            kwargs.update(self.ppo_hyperparams_discrete_probs())
        elif self.action_space == "continuous":
            kwargs.update(self.ppo_hyperparams_continuous())
        elif self.action_space == "discrete":
            kwargs.update(self.ppo_hyperparams_discrete())

        if self.masking and self.action_space == "discrete":
            model = MaskablePPO(**kwargs)
            eval_callback = MaskableEvalCallback(
                self.eval_env,
                n_eval_episodes=self.N_EVAL_EPISODES,
                eval_freq=self.EVAL_FREQ,
                best_model_save_path=self.best_model_save_path,
                deterministic = False,
                verbose=2,
            )
        else:
            model = PPO(**kwargs)
            eval_callback = EvalCallback(
                self.eval_env,
                n_eval_episodes=self.N_EVAL_EPISODES,
                eval_freq=self.EVAL_FREQ,
                best_model_save_path=self.best_model_save_path,
                verbose=2,
                deterministic=True,
            )

        obs = model.env.reset()
        print("learning started")
        model.learn(
            total_timesteps=self.N_TIMESTEPS,
            reset_num_timesteps=False,
            callback=eval_callback,
        )
        print("learning finished")


class PPO_Test_Trained:
    def __init__(self, study):
        self.initialize_global_variables(study)
        self.N_EVALUATIONS = self.N_TIMESTEPS / 1000  # evaluations
        self.EVAL_FREQ = int(self.N_TIMESTEPS / self.N_EVALUATIONS)
        self.N_EVAL_EPISODES = 10


    def initialize_global_variables(self, study):
        self.N_TIMESTEPS = study.N_TIMESTEPS  # for every trial
        self.N_TEST_EPISODES = 100
        self.ORDERS = study.ORDERS
        self.CHANGEOVER = study.CHANGEOVER
        self.recreate_solution = study.recreate_solution
        self.generate_heuristic_schedules = study.generate_heuristic_schedules
        self.heuristics_policy_rl = study.heuristics_policy_rl
        self.test = study.test
        self.experiment_folder = study.experiment_folder
        self.seed = study.seed
        self.masking = study.masking
        self.solution_hints = study.solution_hints
        self.reward = study.reward
        self.obs_size = study.obs_size
        self.action_space = study.action_space
        self.buffer_usage = study.buffer_usage
        self.experiment_date = study.experiment_date
        self.experiment_time = study.experiment_time
        self.config = study.config
        self.log_path = study.log_path
        self.best_model_save_path = study.best_model_save_path
        self.test_env = make_environment(study, "_testing")
        self.train_env = make_environment(study, "_training")
        self.eval_env = make_environment(study, "_evaluation")
    def PPO_Test_Trained_Run(self):

        if self.masking and self.action_space == "discrete":
            model_to_load = "C:/Users/INOSIM/PycharmProjects/flexible_flowshop/src/outputs/2023-08-11/14-38-00/ppo_manual/Training/Saved_Models/best_model.zip"
            model = MaskablePPO.load(model_to_load)
            print("loaded model")
            model.set_env(self.train_env)
            print("set env model")
            eval_callback = MaskableEvalCallback(
                self.eval_env,
                n_eval_episodes=self.N_EVAL_EPISODES,
                eval_freq=self.EVAL_FREQ,
                best_model_save_path=self.best_model_save_path,
                deterministic = False,
                verbose=2,
            )
        else:
            model_to_load = "C:/Users/INOSIM/PycharmProjects/flexible_flowshop2/src/outputs/2023-08-11/14-37-56/ppo_manual/Training/Saved_Models/best_model.zip"
            model = PPO.load(model_to_load)
            print("loaded model")
            model.set_env(self.train_env)
            print("set env model")
            eval_callback = EvalCallback(
                self.eval_env,
                n_eval_episodes=self.N_EVAL_EPISODES,
                eval_freq=self.EVAL_FREQ,
                best_model_save_path=self.best_model_save_path,
                verbose=2,
                deterministic=True,
            )

        obs = model.env.reset()
        print("env reset")
        print("learning started")
        model.learn(
            total_timesteps=self.N_TIMESTEPS,
            reset_num_timesteps=False,
            callback=eval_callback,
        )
        print("learning finished")

        print("start testing!")

        for episode in range(self.N_TEST_EPISODES):
            print("Environment is reset")
            obs = model.env.reset()
            done = False

            while not done:
                if self.masking and self.action_space == "discrete":
                    action, _ = model.predict(
                        obs,
                        action_masks=self.env.valid_action_mask(),
                        deterministic=True,
                    )
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, rewards, done, info = model.env.step(action)
            print("Episode terminated!")


class PPO_Simple_Run:
    def __init__(self, study):
        self.initialize_global_variables(study)
        self.terminated_makespan = {key: [] for key in range(self.N_TEST_EPISODES)}
        self.terminated_occ = {key: [] for key in range(self.N_TEST_EPISODES)}
        self.terminated_wl = {key: [] for key in range(self.N_TEST_EPISODES)}

    def initialize_global_variables(self, study):
        self.N_TEST_EPISODES = study.N_TIMESTEPS
        self.ORDERS = study.ORDERS
        self.CHANGEOVER = study.CHANGEOVER
        self.recreate_solution = study.recreate_solution
        self.generate_heuristic_schedules = study.generate_heuristic_schedules
        self.heuristics_policy_rl = study.heuristics_policy_rl
        self.test = study.test
        self.experiment_folder = study.experiment_folder
        self.seed = study.seed
        self.masking = study.masking
        self.solution_hints = study.solution_hints
        self.reward = study.reward
        self.obs_size = study.obs_size
        self.action_space = study.action_space
        self.buffer_usage = study.buffer_usage
        self.experiment_date = study.experiment_date
        self.experiment_time = study.experiment_time
        self.config = study.config
        self.log_path = study.log_path
        self.best_model_save_path = study.best_model_save_path
        self.env = flexible_flow_shop(study)

    def PPO_Simple_Run_run(self):
        for episode in range(self.N_TEST_EPISODES):
            obs = self.env.reset()
            sim_init = datetime.datetime.now()
            done = False
            counter = np.random.randint(2)
            while not done:
                if self.env.legal_operations.any():  # IF ANY LEGAL OPERATION EXISTS
                    current_legal_operations = np.where(self.env.legal_operations)[0]

                    # ACTION SELECTION DEPENDING ON THE EXPERIMENT SETTING
                    if self.recreate_solution != None:
                        action = current_legal_operations[0]
                    elif self.generate_heuristic_schedules != None:
                        action, counter = get_action_heuristics(
                            self.env,
                            current_legal_operations,
                            counter,
                            self.generate_heuristic_schedules,
                            self.heuristics_policy_rl,
                            self.CHANGEOVER,
                        )
                    else:
                        action = np.random.choice(current_legal_operations)

                    # SEND ACTION
                    obs, reward, done, info = self.env.step(action)

                else:  # SEND NOOP ACTION
                    self.env.legal_operations[-1] = True
                    obs, reward, done, info = self.env.step(len(self.ORDERS))

            array_makespans = list(self.terminated_makespan.values())
            array_occ = list(self.terminated_occ.values())
            array_wl = list(self.terminated_wl.values())

            flatten_list_makespans = [i for sublist in array_makespans for i in sublist]
            flatten_list_occ = [i for sublist in array_occ for i in sublist]
            flatten_list_wl = np.float64([i for sublist in array_wl for i in sublist])

            if self.generate_heuristic_schedules != None:
                if flatten_list_makespans == []:  # ARRAY VALUE IN FIRST SIMULATION
                    self.env.render()

                else:
                    if (
                        self.generate_heuristic_schedules == "FIFO"
                        or self.generate_heuristic_schedules == "SPT"
                    ):
                        env_variable = self.env.sim_duration
                        env_list_optimization_variable = flatten_list_makespans

                    elif self.generate_heuristic_schedules == "EDD":
                        env_variable = self.env.weighted_total_lateness
                        env_list_optimization_variable = flatten_list_wl

                    elif self.generate_heuristic_schedules == "SCT":
                        env_variable = self.env.oc_costs
                        env_list_optimization_variable = flatten_list_occ

                    mean = np.mean(env_list_optimization_variable)
                    std = np.std(env_list_optimization_variable)
                    upper_condition = mean + 0.01 * std
                    lower_condition = mean - 0.01 * std

                    if (
                        env_variable <= upper_condition
                        and env_variable >= lower_condition
                    ) or env_variable < np.min(
                        env_list_optimization_variable
                    ):  # IF ANY SIMULATION IS BETTER THAN PAST ONES, RENDER
                        self.env.render()
                        print("PLOT: YES")
                    else:
                        print("PLOT: NO")
                self.terminated_makespan[episode].append(self.env.sim_duration)
                self.terminated_occ[episode].append(self.env.oc_costs)
                self.terminated_wl[episode].append(self.env.weighted_total_lateness)

            else:
                self.env.render()

            sim_end = datetime.datetime.now()
            print("#############################")
            print("Episode #{}".format(episode))
            print("Computation time: {}".format(sim_end - sim_init))
            print("#############################")

        results_path = "outputs/{}/{}/render".format(self.experiment_folder, self.test)

        if self.generate_heuristic_schedules != None and flatten_list_makespans != []:
            info_heuristics = {
                "POLICY": self.generate_heuristic_schedules,
                "NUMBER OF EPISODES": self.N_TEST_EPISODES,
                "AVERAGE": np.average(env_list_optimization_variable),
                "MEAN": np.mean(env_list_optimization_variable),
                "STD": np.std(env_list_optimization_variable),
                "RSTD": 100
                * np.std(env_list_optimization_variable)
                / np.mean(env_list_optimization_variable),
                "MIN": np.min(env_list_optimization_variable),
                "MAX": np.max(env_list_optimization_variable),
            }

            data = pd.DataFrame(
                {
                    "MAKESPAN": flatten_list_makespans,
                    "OCC": flatten_list_occ,
                    "WL": flatten_list_wl,
                }
            )

            GenerateHeuristicResultsFiles(
                data, results_path, self.generate_heuristic_schedules, info_heuristics
            )
            raincloud_plotter(
                data,
                results_path,
                self.N_TEST_EPISODES,
                self.generate_heuristic_schedules,
            )


class PPO_Imitation_DAgger:

    def __init__(self, study):
        self.initialize_global_variables(study)
        self.N_EVALUATIONS = self.N_TIMESTEPS / 1000  # evaluations
        self.EVAL_FREQ = int(self.N_TIMESTEPS / self.N_EVALUATIONS)
        self.N_EVAL_EPISODES = 10

    def initialize_global_variables(self, study):
        self.N_TIMESTEPS = study.N_TIMESTEPS  # for every trial
        self.ORDERS = study.ORDERS
        self.CHANGEOVER = study.CHANGEOVER
        self.recreate_solution = study.recreate_solution
        self.generate_heuristic_schedules = study.generate_heuristic_schedules
        self.heuristics_policy_rl = study.heuristics_policy_rl
        self.test = study.test
        self.experiment_folder = study.experiment_folder
        self.seed = study.seed
        self.masking = study.masking
        self.solution_hints = study.solution_hints
        self.reward = study.reward
        self.obs_size = study.obs_size
        self.action_space = study.action_space
        self.buffer_usage = study.buffer_usage
        self.experiment_date = study.experiment_date
        self.experiment_time = study.experiment_time
        self.config = study.config
        self.log_path = study.log_path
        self.best_model_save_path = study.best_model_save_path
        self.env = make_environment(study,"")

    def PPO_Imitation_DAgger_run(self):
        rng = np.random.default_rng(0)
        env = self.env

        if self.action_space == "discrete" and self.masking:
            expert_to_load = None
            expert = MaskablePPO.load(expert_to_load)
        else:
            expert_to_load = "C:/Users/{}/PycharmProjects/flexible_flowshop/src/experiments/results/04_rl_larger_network/rl+spt_mks/Saved_Models/best_model.zip".format(os.getlogin())
            expert = PPO.load(expert_to_load)

        print("loaded model")
        expert.set_env(env)
        print("set env model")
        expert.env.reset()
        print("env reset")

        venv = DummyVecEnv([lambda: self.env])

        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,
        )
        with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=venv,
                scratch_dir=tmpdir,
                expert_policy=expert,
                bc_trainer=bc_trainer,
                rng=rng,
            )
            dagger_trainer.train(2000)

        reward, _ = evaluate_policy(dagger_trainer.policy, env, self.N_EVAL_EPISODES)
        print("Reward:", reward)


class SAC_Discrete:
    def __init__(self, study):
        self.initialize_global_variables(study)
        self.N_EVALUATIONS = self.N_TIMESTEPS / 1000  # evaluations
        self.EVAL_FREQ = int(self.N_TIMESTEPS / self.N_EVALUATIONS)
        self.N_EVAL_EPISODES = 10

    def initialize_global_variables(self, study):
        self.N_TIMESTEPS = study.N_TIMESTEPS  # for every trial
        self.ORDERS = study.ORDERS
        self.CHANGEOVER = study.CHANGEOVER
        self.recreate_solution = study.recreate_solution
        self.generate_heuristic_schedules = study.generate_heuristic_schedules
        self.test = study.test
        self.experiment_folder = study.experiment_folder
        self.seed = study.seed
        self.masking = study.masking
        self.solution_hints = study.solution_hints
        self.reward = study.reward
        self.obs_size = study.obs_size
        self.action_space = study.action_space
        self.buffer_usage = study.buffer_usage
        self.experiment_date = study.experiment_date
        self.experiment_time = study.experiment_time
        self.config = study.config
        self.log_path = study.log_path
        self.best_model_save_path = study.best_model_save_path
        self.train_env = make_environment(study, "_training")
        self.eval_env = make_environment(study, "_evaluation")

    def SAC_Discrete_run(self):
        args = sac_args_default(self).copy()
        if self.reward == "MAKESPAN":
            args.update(sac_args_optuna_makespan(self))
        elif self.reward == "OCC":
            args.update(sac_args_optuna_occ(self))
        elif self.reward == "TASSEL":
            args.update(sac_args_optuna_tassel(self))
        # set global seed
        set_global_seed(self.seed)

        # initialize logger
        env_name = args['env_name']
        logger = Logger(self.log_path, env_name, self.seed, info_str="", print_to_terminal=True)

        # set device and logger
        set_device_and_logger(-1, logger)

        # save args
        logger.log_str_object("parameters", log_dict=args)

        # initialize environment
        logger.log_str("Initializing Environment")
        train_env = self.train_env
        eval_env = self.eval_env
        observation_space = train_env.observation_space
        action_space = train_env.action_space

        # initialize buffer
        logger.log_str("Initializing Buffer")
        buffer = ReplayBuffer(observation_space, action_space, **args['buffer'])

        # initialize agent
        logger.log_str("Initializing Agent")
        agent = SACAgent(observation_space, action_space, **args['agent'])

        # initialize trainer
        logger.log_str("Initializing Trainer")
        trainer = SACTrainer(
            agent,
            train_env,
            eval_env,
            buffer,
            load_path="",
            **args['trainer']
        )

        logger.log_str("Started training")
        trainer.train()

class REDQ:
    def __init__(self, study):
        self.initialize_global_variables(study)
        self.N_EVALUATIONS = self.N_TIMESTEPS / 1000  # evaluations
        self.EVAL_FREQ = int(self.N_TIMESTEPS / self.N_EVALUATIONS)
        self.N_EVAL_EPISODES = 10

    def initialize_global_variables(self, study):
        self.N_TIMESTEPS = study.N_TIMESTEPS  # for every trial
        self.ORDERS = study.ORDERS
        self.CHANGEOVER = study.CHANGEOVER
        self.recreate_solution = study.recreate_solution
        self.generate_heuristic_schedules = study.generate_heuristic_schedules
        self.test = study.test
        self.experiment_folder = study.experiment_folder
        self.seed = study.seed
        self.masking = study.masking
        self.solution_hints = study.solution_hints
        self.reward = study.reward
        self.obs_size = study.obs_size
        self.action_space = study.action_space
        self.buffer_usage = study.buffer_usage
        self.experiment_date = study.experiment_date
        self.experiment_time = study.experiment_time
        self.config = study.config
        self.log_path = study.log_path
        self.best_model_save_path = study.best_model_save_path
        self.train_env = make_environment(study, "_training")
        self.eval_env = make_environment(study, "_evaluation")

    def REDQ_run(self):

        args = redq_args(self)
        # set global seed
        set_global_seed(self.seed)
        # initialize logger
        env_name = args['env_name']
        logger = Logger(self.log_path, env_name, self.seed, info_str="", print_to_terminal=True)

        # set device and logger
        set_device_and_logger(-1, logger)

        # save args
        logger.log_str_object("parameters", log_dict=args)

        # initialize environment
        logger.log_str("Initializing Environment")
        train_env = self.train_env
        eval_env = self.eval_env
        observation_space = train_env.observation_space
        action_space = train_env.action_space

     # initialize buffer
        buffer = ReplayBuffer(observation_space, action_space, **args['buffer'])

        # initialize agent
        logger.log_str("Initializing Agent")
        agent = REDQAgent(observation_space, action_space, **args['agent'])

        # initialize trainer
        logger.log_str("Initializing Trainer")
        trainer = REDQTrainer(
            agent,
            train_env,
            eval_env,
            buffer,
            load_path="",
            **args['trainer']
        )

        logger.log_str("Started training")
        trainer.train()



class optuna_SAC_discrete:

    def __init__(self, study):
        self.DEVICE = torch.device("cpu")
        self.initialize_global_variables(study)
        self.N_TRIALS = self.optuna_trials
        self.N_STARTUP_TRIALS = 1
        self.N_EVALUATIONS = self.N_TIMESTEPS / 1000  # evaluations per trial
        self.EVAL_FREQ = int(self.N_TIMESTEPS / self.N_EVALUATIONS)
        self.N_EVAL_EPISODES = 10
        self.optuna_log_path = "outputs/{}/{}/Training/Optuna/Logs".format(
            self.experiment_folder, self.test
        )
    def initialize_global_variables(self, study):
        self.N_TIMESTEPS = study.N_TIMESTEPS  # for every trial
        self.ORDERS = study.ORDERS
        self.CHANGEOVER = study.CHANGEOVER
        self.recreate_solution = study.recreate_solution
        self.generate_heuristic_schedules = study.generate_heuristic_schedules
        self.test = study.test
        self.experiment_folder = study.experiment_folder
        self.seed = study.seed
        self.masking = study.masking
        self.solution_hints = study.solution_hints
        self.reward = study.reward
        self.obs_size = study.obs_size
        self.action_space = study.action_space
        self.buffer_usage = study.buffer_usage
        self.experiment_date = study.experiment_date
        self.experiment_time = study.experiment_time
        self.config = study.config
        self.log_path = study.log_path
        self.optuna_trials = study.optuna_trials
        self.best_model_save_path = study.best_model_save_path
        self.train_env = make_environment(study, "_training")
        self.eval_env = make_environment(study, "_evaluation")

    def sample_sac_hyperparams(self, trial: optuna.Trial) -> Dict[str, Any]:
        # Define your environment and agent configurations
        args = sac_args_default(self)

        # Define hyperparameter search space
        args["agent"]["alpha"] = trial.suggest_float("alpha", 0.1, 1.0)
        args["agent"]["q_network"]["learning_rate"] = trial.suggest_float("q_learning_rate", 1e-5, 1e-3)
        args["agent"]["policy_network"]["learning_rate"] = trial.suggest_float("policy_learning_rate", 1e-5, 1e-3)
        args["agent"]["entropy"]["learning_rate"] = trial.suggest_float("entropy_learning_rate", 1e-5, 1e-3)
        args["agent"]["reward_scale"] = trial.suggest_float("reward_scale", 1.0, 10.0)

        # Run your training and evaluation using the specified hyperparameters
        # You need to implement your training loop here
        # Return the value you want to minimize (e.g., makespan, costs, etc.)
        return args

    def objective(self, trial: optuna.Trial) -> float:
        args = sac_args_default(self).copy()
        args.update(self.sample_sac_hyperparams(trial))

        # set global seed
        set_global_seed(self.seed)

        # initialize logger
        env_name = args['env_name']
        logger = Logger(self.log_path, env_name, self.seed, info_str="", print_to_terminal=True)

        # set device and logger
        set_device_and_logger(-1, logger)

        # save args
        logger.log_str_object("parameters", log_dict=args)

        # initialize environment
        logger.log_str("Initializing Environment")
        train_env = self.train_env
        eval_env = self.eval_env
        observation_space = train_env.observation_space
        action_space = train_env.action_space

        # initialize buffer
        logger.log_str("Initializing Buffer")
        buffer = ReplayBuffer(observation_space, action_space, **args['buffer'])

        # initialize agent
        logger.log_str("Initializing Agent")
        agent = SACAgent(observation_space, action_space, **args['agent'])

        # initialize trainer
        logger.log_str("Initializing Trainer")
        trainer = SACTrainer(
            agent,
            train_env,
            eval_env,
            buffer,
            load_path="",
            **args['trainer']
        )

        logger.log_str("Started training")
        trainer.train()
        ret_eval = trainer.post_step("get_last_mean_reward")
        print("objective function return value: ", ret_eval)
        return ret_eval
        {}
    def optuna_SAC_discrete_run(self):
        torch.set_num_threads(1)

        sampler = TPESampler(n_startup_trials=self.N_STARTUP_TRIALS, seed=self.seed)
        pruner = MedianPruner(
            n_startup_trials=self.N_STARTUP_TRIALS, n_warmup_steps=self.N_TIMESTEPS // 3
        )

        study = optuna.create_study(
            storage="sqlite:///outputs/{}/Optuna_Trial.db".format(
                self.experiment_folder
            ),
            study_name="Optuna_Trial",
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
        )

        print("created optuna study")
        print("start study optimize")
        study.optimize(self.objective, n_trials=self.N_TRIALS)
        print("study optimize finished")

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        with open("best_trial.txt", "w") as f:
            print("Best trial:", file=f)
            print("  Value: ", trial.value, file=f)
            print("  Params: ", file=f)
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value), file=f)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print("    {}: {}".format(key, value))