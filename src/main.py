import hydra
from omegaconf import DictConfig
from flexible_flow_shop.resources.functions.global_variables import StudyCase
from experiments.experiments import (
    PPO_Optuna,
    PPO_Manual_Parameters,
    PPO_Simple_Run,
    PPO_Test_Trained,
    PPO_Imitation_DAgger,
    SAC_Discrete,
    REDQ
)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> DictConfig:
    for key, value in cfg.params[0].items():
        if value == "None":
            cfg.params[0][key] = None
    config = cfg.params[0]
    study = StudyCase(config)

    if config.test == "ppo_manual":
        PPO_Manual_Parameters(study).PPO_Manual_Parameters_run()
    elif config.test == "simple_run":
        PPO_Simple_Run(study).PPO_Simple_Run_run()
    elif config.test == "ppo_optuna":
        PPO_Optuna(study).PPO_Optuna_run()
    elif config.test == "ppo_test_trained":
        PPO_Test_Trained(study).PPO_Test_Trained_Run()
    elif config.test == "imitation_dagger":
        PPO_Imitation_DAgger(study).PPO_Imitation_DAgger_run()
    elif config.test == "sac_discrete":
        SAC_Discrete(study).SAC_Discrete_run()
    elif config.test == "redq":
        REDQ(study).REDQ_run()

if __name__ == "__main__":
    main()
