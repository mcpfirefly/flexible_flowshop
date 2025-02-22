import numpy as np
from abc import ABC, abstractmethod
import torch
import os
import cv2
from time import time
from src.experiments.rl_algorithms.unstable_baselines.common import util
from src.experiments.rl_algorithms.unstable_baselines.common import util
class BaseTrainer():
    def __init__(self, agent, train_env, eval_env, 
            max_trajectory_length,
            log_interval,
            eval_interval,
            num_eval_trajectories,
            save_video_demo_interval,
            snapshot_interval,
            **kwargs):
        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.max_trajectory_length = max_trajectory_length
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_trajectories = num_eval_trajectories
        self.save_video_demo_interval = save_video_demo_interval
        self.snapshot_interval = snapshot_interval
        self.last_log_timestep = 0
        self.last_eval_timestep = 0
        self.last_mean_return = 0
        self.last_snapshot_timestep = 0
        self.last_video_demo_timestep = 0
        pass

    @abstractmethod
    def train(self):
        #do training 
        pass
    def pre_iter(self):
        self.ite_start_time = time()

    def post_step(self, timestep):
        log_dict = {}
        if timestep == "get_last_mean_reward":
            ret_val = self.last_mean_return
            return ret_val

        elif timestep % self.eval_interval == 0 or timestep - self.last_eval_timestep > self.eval_interval:
            eval_start_time = time()
            log_dict.update(self.evaluate())
            eval_used_time = time() - eval_start_time
            avg_test_return = log_dict['performance/eval_return']
            for log_key in log_dict:
                util.logger.log_var(log_key, log_dict[log_key], timestep)
            util.logger.log_var("times/eval", eval_used_time, timestep)
            summary_str = "Timestep:{}\tEvaluation return {:02f}".format(timestep, avg_test_return)
            util.logger.log_str(summary_str)
            self.last_eval_timestep = timestep
            self.last_mean_return = avg_test_return




    def post_iter(self, log_dict, timestep):
        if timestep % self.log_interval == 0 or timestep - self.last_log_timestep > self.log_interval:
            for loss_name in log_dict:
                util.logger.log_var(loss_name, log_dict[loss_name], timestep)
            self.last_log_timestep = timestep

        if timestep % self.snapshot_interval == 0 or timestep - self.last_snapshot_timestep > self.snapshot_interval:
            self.snapshot(timestep)
            self.last_snapshot_timestep = timestep
        
        if self.save_video_demo_interval > 0 and (timestep % self.save_video_demo_interval == 0 or timestep - self.last_video_demo_timestep > self.save_video_demo_interval ):
            self.save_video_demo(timestep)
            self.last_video_demo_timestep = timestep

    @torch.no_grad()
    def evaluate(self):
        traj_returns = []
        traj_lengths = []
        traj_successes = []
        for traj_id in range(self.num_eval_trajectories):
            traj_return = 0
            traj_length = 0
            success = False
            obs = self.eval_env.reset()

            # Generating masked action probabilities from network


            for step in range(self.max_trajectory_length):
                legal_actions = self.eval_env.valid_action_mask()
                action = self.agent.select_action(obs, legal_actions, deterministic=True)['action']

                next_obs, reward, done, info = self.eval_env.step(action)

                #self.eval_env.render()
                traj_return += reward
                obs = next_obs
                traj_length += 1
                success = success or info.get("success", False)
                if done:
                    break
            traj_lengths.append(traj_length)
            traj_returns.append(traj_return)
            traj_successes.append(success)
        ret_info = {
            "performance/eval_return": np.mean(traj_returns),
            "performance/eval_length": np.mean(traj_lengths),
            "advanced_eval/makespan": self.eval_env.unwrapped.sim_duration,
            "advanced_eval/oc_costs": self.eval_env.unwrapped.oc_costs,
            "advanced_eval/weighted_lateness": self.eval_env.unwrapped.weighted_total_lateness,
            "advanced_eval/total_reward": self.eval_env.unwrapped.total_reward,
            "advanced_eval/completion_score": self.eval_env.unwrapped.completion_score,
        }
        if len(traj_successes) > 0:
            ret_info["performance/eval_success"] = np.mean(traj_successes)

        return ret_info
    
        
        
    def save_video_demo(self, ite, width=210, height=160, fps=30):
        video_demo_dir = os.path.join(util.logger.log_dir,"demos")
        if not os.path.exists(video_demo_dir):
            os.makedirs(video_demo_dir)
        video_size = (height, width)
        video_save_path = os.path.join(video_demo_dir, "ite_{}.mp4".format(ite))

        #initilialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

        #rollout to generate pictures and write video
        obs = self.eval_env.reset()
        
        img = self.eval_env.render(mode="rgb_array", width=width, height=height).astype('uint8')
        video_writer.write(img)
        for step in range(self.max_trajectory_length):
            action = self.agent.select_action(obs)['action']
            next_obs, reward, done, _ = self.eval_env.step(action)
            obs = next_obs
            img = self.eval_env.render(mode="rgb_array", width=width, height=height).astype('uint8')
            video_writer.write(img)
            if done:
                break
                
        video_writer.release()

    def snapshot(self, timestamp):
        save_dir = os.path.join(util.logger.log_path, 'models')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, "ite_{}.pt".format(timestamp))
        torch.save(self.agent.state_dict(), model_save_path)

    def load_snapshot(self, load_path):
        if not os.path.exists(load_path):
            print("\033[31mLoad path not found:{}\033[0m".format(load_path))
            exit(0)
        self.agent.load_state_dict(torch.load(load_path, map_location=util.device))
        