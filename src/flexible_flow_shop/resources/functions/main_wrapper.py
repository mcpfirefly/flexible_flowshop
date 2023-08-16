from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from src.flexible_flow_shop.resources.functions.custom_wrappers import *
import numpy as np
import gym


# Custom environment action masking for probabilities in discrete action space


def mask_fn(env):
    return env.valid_action_mask()


def MakeEnvironment(
    seed,
    env,
    masking,
    reward,
    norm_obs,
    norm_rew,
    action_mode,
    obs_size,
    buffer_usage,
    log_path,
    heuristics_policy_rl,
):
    """OBSERVATION SIZE"""
    if obs_size == "big":
        env = ObservationWrapper_Big(env)
    elif obs_size == "medium":
        env = ObservationWrapper_Medium(env)
    elif obs_size == "small":
        env = ObservationWrapper_Small(env)
    elif obs_size == "new":
        env = ObservationWrapper_New(env)
    elif obs_size == "big_old":
        env = ObservationWrapper_Big_old(env)
    elif obs_size == "small_old":
        env = ObservationWrapper_Small_old(env)

    """ACTION MODE"""

    if buffer_usage == "agent_decides":
        """ACTION MODE WITH BUFFER TANKS"""
        if action_mode == "discrete":
            env = ActionWrapper_DiscreteWithBuffer(env)
        elif action_mode == "discrete_probs":
            env = ActionWrapper_DiscreteProbsWithBuffer(env)
        elif action_mode == "continuous":
            env = ActionWrapper_ContinuousWithBuffer(env)
            env = gym.wrappers.ClipAction(env)
    else:
        """ACTION MODE WITHOUT BUFFER TANKS/ NOT DECIDED BY AGENT"""
        if action_mode == "discrete":
            env = ActionWrapper_Discrete(env)
        elif action_mode == "discrete_probs":
            env = ActionWrapper_DiscreteProbs(env)
        elif action_mode == "continuous":
            env = ActionWrapper_Continuous(env)
            env = gym.wrappers.ClipAction(env)

    """MASKING USE (ONLY DISCRETE AND DISCRETE WITH PROBS)"""
    if (masking and action_mode == "discrete") or (heuristics_policy_rl != None):
        env = ActionMasker(env, mask_fn)
    elif masking and action_mode == "discrete_probs":
        env = ProbabilitiesActionMaskEnv(env)

    """REWARD FUNCTION"""
    if reward == "OCC":
        env = RewardWrapper_OCC(env)
    elif reward == "WL":
        env = RewardWrapper_WL(env)
    elif reward == "MAKESPAN":
        env = RewardWrapper_MAKESPAN(env)
    elif reward == "LENGTH":
        env = RewardWrapper_LENGTH(env)
    elif reward == "COMPLETION":
        env = RewardWrapper_Completion(env)
    elif reward == "TASSEL":
        env = RewardWrapper_TASSEL(env)
    else:
        print(
            "No reward wrapper was selected! Default will be used instead: PENALIZE ILLEGAL ACTIONS, REWARD LEGAL ACTIONS"
        )
        env = RewardWrapper_Default(env)

    """INPUT/OUTPUT NORMALIZATION"""
    if norm_obs:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    if norm_rew:
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

    """SEED FOR REPRODUCIBILITY"""
    if seed != None:
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    info_keywords = (
        "sim_duration",
        "oc_costs",
        "weighted_lateness",
        "completion_score",
        "schedule",
        "total_reward",
    )
    env = Monitor(env=env, filename=log_path, info_keywords=info_keywords)
    return env
