from ..mock_scheduler import MockScheduler
from torch.optim import lr_scheduler
import warnings


def scheduler_factory(config, optimizer, **kwargs):
    scheduler_name = config["scheduler"]["name"]
    scheduler_args = config["scheduler"]["args"]
    if scheduler_name == "None":
        scheduler = MockScheduler()
    elif scheduler_name == "OneCycleLR":
        try:
            steps_per_epoch = kwargs["steps_per_epoch"]
        except AttributeError:
            raise AttributeError("Steps per epoch required for OneCycleLR")

        scheduler = getattr(lr_scheduler, scheduler_name)(
            optimizer,
            steps_per_epoch=steps_per_epoch,
            **scheduler_args,
        )
    elif scheduler_name == "CyclicLR":
        try:
            steps_per_epoch = kwargs["steps_per_epoch"]
        except AttributeError:
            raise AttributeError("Steps per epoch required for OneCycleLR")

        scheduler_args["step_size_up"] = (
            steps_per_epoch * scheduler_args["step_size_up"]
        )

        print(scheduler_args)

        scheduler = getattr(lr_scheduler, scheduler_name)(
            optimizer,
            **scheduler_args,
        )

    else:
        try:
            scheduler = getattr(lr_scheduler, scheduler_name)(
                optimizer, **scheduler_args
            )
        except AttributeError:
            raise AttributeError(f"Scheduler {scheduler_name} not found")
    return scheduler
