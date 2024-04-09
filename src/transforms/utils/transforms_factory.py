from .. import custom_transforms
from torch.nn import Sequential


def transforms_factory(transforms_config, device):
    all_transforms = Sequential()

    for transform_name in transforms_config.keys():
        transform_params = transforms_config[transform_name]["args"]

        try:
            transform_class = getattr(custom_transforms, transform_name)
        except AttributeError:
            raise AttributeError("Transform {} not found".format(transform_name))

        transform = transform_class(device=device, **transform_params)
        all_transforms.add_module(transform_name, transform)

    return all_transforms
