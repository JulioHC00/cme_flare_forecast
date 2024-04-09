from ..wandb_logger import WandBLogger


def logger_factory(config):
    logger_name = config["logger"]["name"]
    logger_args = config["logger"]["args"]
    device = config["shared"]["device"]
    if logger_name == "wandb":
        logger = WandBLogger(device=device, full_config=config, **logger_args)
    else:
        raise AttributeError(f"Logger {logger_name} not found")
    return logger
