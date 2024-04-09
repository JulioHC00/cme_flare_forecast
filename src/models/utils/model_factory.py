import torch.nn as nn
from ..rotary_transformer import RotaryTransformerForecastModel

models = {
    "rotary_transformer": RotaryTransformerForecastModel,
}


def model_factory(config: dict) -> nn.Module:
    model_config = config["model"]
    model_name = model_config["name"]
    model_args = model_config["args"]

    model_obj = models.get(model_name, None)

    if model_obj:
        return model_obj(**model_args)
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented")
