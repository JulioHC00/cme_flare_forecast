from omegaconf import DictConfig
import torch
from typing import Union
from ..base_loss import BaseLoss
from ..bcewithlogits_loss import CustomBCEWithLogitsLoss


def loss_factory(
    config: DictConfig,
) -> Union[BaseLoss, torch.nn.Module]:
    loss_config = config["loss"]
    loss_args = loss_config["args"]
    if loss_config["name"] == "bcewithlogits":
        return CustomBCEWithLogitsLoss(**loss_args)
    else:
        raise NotImplementedError(f"Loss {loss_config['name']} is not implemented")
