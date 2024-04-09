import torch.optim as optim


def optimizer_factory(config, model):
    optimizer_name = config["optimizer"]["name"]
    optimizer_args = config["optimizer"]["args"]

    # Only train parameters that require grad
    params = filter(lambda p: p.requires_grad, model.parameters())

    try:
        optimizer_obj = getattr(optim, optimizer_name)
    except AttributeError:
        raise AttributeError(f"Optimizer {optimizer_name} not found")
    optimizer = optimizer_obj(params, **optimizer_args)

    return optimizer
