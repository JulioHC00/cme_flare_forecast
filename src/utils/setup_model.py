import torch
from ..loggers.utils.logger_factory import logger_factory
from ..loss.utils.loss_factory import loss_factory
from ..models.utils.model_factory import model_factory
from ..optimizers.utils.optimizer_factory import optimizer_factory
from ..schedulers.utils.scheduler_factory import scheduler_factory
from ..data_loaders.utils.dataloader_factory import dataloader_factory
from ..data_loaders.utils.dataset_factory import dataset_factory
from omegaconf import OmegaConf, DictConfig
from torchinfo import summary


def parse_config(config) -> DictConfig:
    # Device, just to check it exists
    try:
        torch.device(config["shared"]["device"])
    except AttributeError:
        raise AttributeError(f"Device {config['shared']['device']} not found")

    return config


def setup(config):
    model_parts = {}
    # 1. Setup random seed
    torch.manual_seed(config["seed"])
    # 5. Setup dataset
    datasets = dataset_factory(config)
    # 6. Setup dataloader
    dataloaders = dataloader_factory(config, datasets)
    model_parts["dataloaders"] = dataloaders
    steps_per_epoch = len(dataloaders["train"])

    # Need to update this
    input_size = dataloaders["train"].dataset.output_shape
    config["model"]["args"]["input_size"] = input_size
    config["loss"]["args"]["pos_weight"] = dataloaders["train"].dataset._pos_weight

    batch_size = config["data"]["batch_size"]

    # 3. Setup model
    model = model_factory(config)
    model.to(config["shared"]["device"])
    summary(model, input_size=(batch_size, *input_size))

    # Check if checkpoint to be loaded
    load_checkpoint = config["train"]["load_checkpoint"]

    if load_checkpoint:
        checkpoint_path = config["train"]["checkpoint_path"]

        checkpoint = torch.load(
            checkpoint_path, map_location=config["shared"]["device"]
        )

        model.load_state_dict(checkpoint["model_state_dict"])

    model_parts["model"] = model

    # 8. Setup loss
    loss = loss_factory(config)
    model_parts["loss"] = loss

    # 4. Setup optimizer
    optimizer = optimizer_factory(config, model)
    model_parts["optimizer"] = optimizer

    # 7. Setup scheduler
    scheduler = scheduler_factory(config, optimizer, steps_per_epoch=steps_per_epoch)
    model_parts["scheduler"] = scheduler

    # 2. Setup logger last so can make sure cfg is final
    logger = logger_factory(config)
    model_parts["logger"] = logger

    return model_parts
