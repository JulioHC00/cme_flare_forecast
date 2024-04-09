import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.setup_model import parse_config, setup
from src.utils.train import train
from src.utils.global_signals import cleanup_event
from src.utils.vacuum_db import vacuum_database

OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="configs", config_name="config", version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    try:
        cfg = parse_config(cfg)

        # I want the cfg as a dictionary
        cfg = OmegaConf.to_container(cfg, resolve=True)

        print(cfg["model"]["args"])

        db_path = cfg["local"]["db_path"]

        model_parts = setup(cfg)
        metrics, best_epoch = train(model_parts, cfg)

        # Print with lots of flourish to make it super noticeable
        print(
            "================================================================================"
        )
        print(
            "================================================================================"
        )
        print("Metrics from the best epoch:")
        print(metrics)
        print("\n")
        print(f"Best epoch: {best_epoch}")
        print(
            "================================================================================"
        )
        print(
            "================================================================================"
        )

    except Exception as e:
        print(e)
        raise e


if __name__ == "__main__":
    main()
