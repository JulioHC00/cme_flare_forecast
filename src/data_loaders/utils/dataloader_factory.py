from torch.utils.data import WeightedRandomSampler, DataLoader
from .collate_function_factory import collate_function_factory


def dataloader_factory(conf, datasets):
    dataloaders = {}
    dataloader_conf = conf["data"]["data_loader"]
    for mode in ["train", "val", "test"]:
        dataset = datasets[mode]
        mode_conf = dataloader_conf[mode]
        collate_fn = collate_function_factory(mode_conf)

        # First need to get the sampler
        if mode_conf["sampler"]["name"] == "weightedrandomsampler":
            weights = dataset.get_dataloader_weights()
            args = mode_conf["sampler"]["args"]

            # Create the sampler
            sampler = WeightedRandomSampler(
                weights=weights, num_samples=len(dataset), **args
            )

        elif mode_conf["sampler"]["name"] == "default":
            sampler = None
        else:
            raise NotImplementedError(
                f"Sampler {mode_conf['sampler']['name']} is not implemented"
            )

        # Now create the dataloader
        if mode_conf["name"] == "default":
            args = mode_conf["args"]
            dataloader = DataLoader(
                dataset, sampler=sampler, collate_fn=collate_fn, **args
            )
            dataloaders[mode] = dataloader
        else:
            raise NotImplementedError(
                f"Dataloader {mode_conf['name']} is not implemented"
            )

    return dataloaders
