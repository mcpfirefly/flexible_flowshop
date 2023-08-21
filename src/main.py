import hydra
from omegaconf import DictConfig
from src.flexible_flow_shop.resources.functions.global_variables import StudyCase
from src.experiments.experiments import (
    PPO_Optuna,
    PPO_Manual_Parameters,
    PPO_Simple_Run,
    PPO_Test_Trained,
    PPO_Imitation_DAgger,
    SAC_Discrete,
    REDQ,
    manual,
    optuna_SAC_discrete
)

debug = False
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> DictConfig:
    for key, value in cfg.params[0].items():
        if value == "None":
            cfg.params[0][key] = None
    config = cfg.params[0]
    config.seed = i
    study = StudyCase(config)

    if debug:
        manual(study)
    else:

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
        elif config.test == "sac_discrete_optuna":
            optuna_SAC_discrete(study).optuna_SAC_discrete_run()

if __name__ == "__main__":
    global i
    for i in range(6):
        print(f"Experiment with seed {i}")
        main()
