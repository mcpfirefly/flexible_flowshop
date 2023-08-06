from src.experiments.rl_algorithms.unstable_baselines.common.util import second_to_time_str
from src.experiments.rl_algorithms.unstable_baselines.common.trainer import BaseTrainer
import numpy as np
from tqdm import tqdm
from time import time
import cv2
import os
from tqdm import trange
import torch

class SACTrainer(BaseTrainer):
    def __init__(self, 
            agent, 
            train_env, 
            eval_env, 
            buffer,  
            batch_size,
            max_env_steps,
            start_timestep,
            random_policy_timestep, 
            load_path="",
            **kwargs):
        super(SACTrainer, self).__init__(agent, train_env, eval_env, **kwargs)
        self.agent = agent
        self.buffer = buffer
        #hyperparameters
        self.batch_size = batch_size
        self.max_env_steps = max_env_steps
        self.start_timestep = start_timestep
        self.random_policy_timestep = random_policy_timestep
        if load_path != "":
            self.load_snapshot(load_path)

        

    def train(self):
        train_traj_returns = [0]
        train_traj_lengths = [0]
        tot_env_steps = 0
        traj_return = 0
        traj_length = 0
        done = False
        obs = self.train_env.reset()
        # Generating masked action probabilities from network


        for env_step in trange(self.max_env_steps): # if system is windows, add ascii=True to tqdm parameters to avoid powershell bugs
            self.pre_iter()
            log_infos = {}

            legal_actions = self.train_env.valid_action_mask()
            action = self.agent.select_action(obs, legal_actions)['action']
            next_obs, reward, done, info = self.train_env.step(action)


            traj_length += 1
            traj_return += reward
            self.buffer.add_transition(obs, action, next_obs, reward, done, truncated=False)
            obs = next_obs
            if done or traj_length >= self.max_trajectory_length:
                obs = self.train_env.reset()
                train_traj_returns.append(traj_return)
                train_traj_lengths.append(traj_length)
                traj_length = 0
                traj_return = 0
            log_infos['performance/train_return'] = train_traj_returns[-1]
            log_infos['performance/train_length'] = train_traj_lengths[-1]
            tot_env_steps += 1
            if tot_env_steps < self.start_timestep:
                continue
    
            data_batch = self.buffer.sample(self.batch_size)
            train_agent_log_infos = self.agent.update(data_batch)
            log_infos.update(train_agent_log_infos)

            self.post_step(tot_env_steps)
            self.post_iter(log_infos, tot_env_steps)



