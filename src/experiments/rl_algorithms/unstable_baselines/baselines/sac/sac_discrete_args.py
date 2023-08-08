def sac_args(self):
    args = {
        "env_name": "",
        "env": {
        },
        "buffer": {
            "max_buffer_size": 100000
        },
        "agent": {
            "gamma": 0.99,
            "update_target_network_interval": 1,
            "target_smoothing_tau": 0.005,
            "alpha": 0.5,
            "reward_scale": 5.0,
            "q_network": {
                "network_params": [("mlp", 64), ("mlp", 64), ("mlp", 64), ("mlp", 64)],
                "optimizer_class": "Adam",
                "learning_rate": 0.0005,
                "act_fn": "relu",
                "out_act_fn": "identity"
            },
            "policy_network": {
                "network_params": [("mlp", 64), ("mlp", 64), ("mlp", 64), ("mlp", 64)],
                "optimizer_class": "Adam",
                "learning_rate": 0.0005,
                "act_fn": "relu",
                "out_act_fn": "identity",
            },
            "entropy": {
                "automatic_tuning": True,
                "learning_rate": 0.0005,
                "optimizer_class": "Adam",
                "scale": 0.5
            }
        },
        "trainer": {
            "max_env_steps": self.N_TIMESTEPS,
            "batch_size": 128,
            "eval_interval": self.EVAL_FREQ,
            "num_eval_trajectories": self.N_EVAL_EPISODES,
            "snapshot_interval": 10000,
            "start_timestep": 2000,
            "random_policy_timestep": 1000,
            "save_video_demo_interval": -1,
            "log_interval": 100,
            "max_trajectory_length": 2000
        }
    }
    return args