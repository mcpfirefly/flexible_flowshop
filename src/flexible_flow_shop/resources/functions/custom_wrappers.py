import numpy as np
import gym, torch
from torch.distributions import Categorical
from torchrl.modules import MaskedCategorical
from gym.spaces import Discrete, Box, MultiDiscrete
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import EvalCallback


class CustomMaskableEvalCallback(MaskableEvalCallback):
    def __init__(
        self,
        eval_env,
        n_eval_episodes,
        eval_freq,
        best_model_save_path,
        verbose=0,
        log_path=None,
    ):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            verbose=verbose,
        )

    def _on_step(self):
        # Call the parent class's on_step_end method to handle the evaluation
        super()._on_step()
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Record additional data
            self.logger.record(
                "advanced_eval/makespan",
                np.float32(self.eval_env.buf_infos[0]["sim_duration"]),
            )
            self.logger.record(
                "advanced_eval/oc_costs",
                np.float32(self.eval_env.buf_infos[0]["oc_costs"]),
            )
            self.logger.record(
                "advanced_eval/weighted_lateness",
                np.float32(self.eval_env.buf_infos[0]["weighted_lateness"]),
            )
            self.logger.record(
                "advanced_eval/total_waiting",
                np.float32(self.eval_env.buf_infos[0]["total_waiting"]),
            )
            self.logger.record(
                "advanced_eval/completion_score",
                np.float32(self.eval_env.buf_infos[0]["completion_score"]),
            )
            self.logger.record(
                "advanced_eval/total_reward",
                np.float32(self.eval_env.buf_infos[0]["total_reward"]),
            )
        return continue_training


class CustomEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env,
        n_eval_episodes,
        eval_freq,
        best_model_save_path,
        verbose,
        deterministic,
        log_path=None,
    ):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            verbose=verbose,
            deterministic=deterministic,
        )

    def _on_step(self):
        # Call the parent class's on_step_end method to handle the evaluation
        super()._on_step()
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Record additional data
            self.logger.record(
                "advanced_eval/makespan",
                np.float32(self.eval_env.buf_infos[0]["sim_duration"]),
            )
            self.logger.record(
                "advanced_eval/oc_costs",
                np.float32(self.eval_env.buf_infos[0]["oc_costs"]),
            )
            self.logger.record(
                "advanced_eval/weighted_lateness",
                np.float32(self.eval_env.buf_infos[0]["weighted_lateness"]),
            )
            self.logger.record(
                "advanced_eval/total_waiting",
                np.float32(self.eval_env.buf_infos[0]["total_waiting"]),
            )
            self.logger.record(
                "advanced_eval/completion_score",
                np.float32(self.eval_env.buf_infos[0]["completion_score"]),
            )
            self.logger.record(
                "advanced_eval/total_reward",
                np.float32(self.eval_env.buf_infos[0]["total_reward"]),
            )
        return continue_training

class RewardWrapper_TASSEL(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper_TASSEL, self).__init__(env)

    def reward(self, reward):
        # Modify the reward calculation here
        modified_reward = self.tassel_reward + reward
        return modified_reward



class RewardWrapper_MAKESPAN(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper_MAKESPAN, self).__init__(env)

    def reward(self, reward):
        # Modify the reward calculation here
        modified_reward = -(
            np.around(self.sim_duration_current, 4)
            - np.around(self.sim_duration_past, 4)
        )
        modified_reward = reward + modified_reward
        return modified_reward


class RewardWrapper_OCC(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper_OCC, self).__init__(env)

    def reward(self, reward):
        # Modify the reward calculation here
        modified_reward = -(self.oc_costs_current - self.oc_costs_past)
        modified_reward = reward + modified_reward
        return modified_reward


class RewardWrapper_WL(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper_WL, self).__init__(env)

    def reward(self, reward):
        # Modify the reward calculation here
        modified_reward = -(self.wl_current - self.wl_past)
        modified_reward = reward + modified_reward
        return modified_reward


class RewardWrapper_LENGTH(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper_LENGTH, self).__init__(env)

    def reward(self, reward):
        # Modify the reward calculation here
        modified_reward = -(self.episode_length_current - self.episode_length_past)
        modified_reward = reward + modified_reward
        return modified_reward


class RewardWrapper_Default(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper_Default, self).__init__(env)

    def reward(self, reward):
        # Modify the reward calculation here
        return reward


class RewardWrapper_Completion(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper_Completion, self).__init__(env)

    def reward(self, reward):
        # Modify the reward calculation here
        modified_reward = self.job_completion_current - self.job_completion_past
        modified_reward = reward + modified_reward
        return modified_reward


class ActionWrapper_Discrete(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._action_space = Discrete(len(self.orders) + 1)

    def action(self, act):
        return act


class ActionWrapper_DiscreteProbs(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        high = np.ones(len(self.orders) + 1)
        low = np.zeros(len(self.orders) + 1)
        self._action_space = Box(
            low=-high, high=high, shape=(len(self.orders) + 1,), dtype=np.float
        )

    def action(self, act):
        act = np.argmax(act.probs)
        return act


class ActionWrapper_Continuous(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        high = np.ones(2)
        self.action_space = Box(low=-high, high=high, shape=(2,), dtype=np.float)

    def action(self, action):
        if (self.action_space.low == -self.action_space.high).all():
            action = (action + self.action_space.high) / (
                self.action_space.high - self.action_space.low
            )  # transform normalized action vector [-1,1] to [0,1]
        product = self.env.env._get_product_number(action[0])
        machine = self.env.env._get_machine_number(action[1])
        action = product * machine  # map tuple (product,machine) to operation
        return action


class ObservationWrapper_Big(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs = self.env._observations_big()
        self.observation_space = Box(shape=(len(obs),), low=-np.inf, high=np.inf)

    def observation(self, obs):
        obs = self.env._observations_big()
        return obs


class ObservationWrapper_Medium(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs = self.env._observations_medium()
        self.observation_space = Box(shape=(len(obs),), low=-np.inf, high=np.inf)

    def observation(self, obs):
        obs = self.env._observations_medium()
        return obs


class ObservationWrapper_Small(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs = self.env._observations_small()
        self.observation_space = Box(shape=(len(obs),), low=-np.inf, high=np.inf)

    def observation(self, obs):
        obs = self.env._observations_small()
        return obs


class ProbabilitiesActionMaskEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if (self.action_space.low == -self.action_space.high).all():
            action = (action + self.action_space.high) / (
                self.action_space.high - self.action_space.low
            )  # transform normalized action vector [-1,1] to [0,1]

        action_mask_tensor = torch.from_numpy((self.env.valid_action_mask()).flatten())
        action_tensor = torch.from_numpy(action).float()

        normalized_probs = (action_tensor / action_tensor.sum() + 1e-4).clone().detach()
        action_probs = MaskedCategorical(probs=normalized_probs, mask=action_mask_tensor)
        return action_probs

class ActionWrapper_DiscreteWithBuffer(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._action_space = MultiDiscrete([len(self.orders) + 1, 2, 6])
        # Operation to allocate, buffer_usage, buffer_time/5

    def action(self, act):
        action = act[0]
        if action != len(self.orders):
            self.orders[action].go_to_buffer = act[1]
            self.orders[action].time_in_buffer = act[2] / 5
        return action


class ActionWrapper_DiscreteProbsWithBuffer(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape_with_buffer = (
            len(self.orders) + 1 + 2 + 6
        )  # Operations, NOOP, use_buffer, time_buffer
        high = np.ones(shape_with_buffer)
        low = np.zeros(shape_with_buffer)
        self._action_space = Box(
            low=low, high=high, shape=(shape_with_buffer,), dtype=np.float
        )

    def action(self, act):
        action = np.argmax(act[0 : len(self.orders) + 1])
        if action != len(self.orders):
            self.orders[action].go_to_buffer = np.argmax(
                act[len(self.orders) + 1 :][:2]
            )
            self.orders[action].time_in_buffer = (
                np.argmax(act[len(self.orders) + 1 + 2 :][:6]) / 5
            )
        return action


class ActionWrapper_ContinuousWithBuffer(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        high = np.ones(4)
        self.action_space = Box(low=-high, high=high, shape=(4,), dtype=np.float)
        # action = product, machine, use_buffer, time_in_buffer

    def action(self, action):
        if (self.action_space.low == -self.action_space.high).all():
            action = (action + self.action_space.high) / (
                self.action_space.high - self.action_space.low
            )  # transform normalized action vector [-1,1] to [0,1]

        product = self.env.env._get_product_number(action[0])
        machine = self.env.env._get_machine_number(action[1])
        action_pair = product * machine  # map tuple (product,machine) to operation

        if action[2] > 0.5:
            self.orders[action_pair].go_to_buffer = 1
        else:
            self.orders[action_pair].go_to_buffer = 0

        self.orders[action_pair].time_in_buffer = action[3]

        return action_pair
