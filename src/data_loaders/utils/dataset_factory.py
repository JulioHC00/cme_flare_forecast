from .create_swan_dataset import create_swan_dataset
from ..base_swan_dataset import PreProcessSWANDataset
import uuid


def dataset_factory(config):
    print("Loading Datasets")
    print("----------------")

    datasets = {}
    dataset_config = config["data"]["dataset"]
    temp_db_id = str(uuid.uuid4())

    # For SWAN, we need to get the stats from the training set
    stats = None
    PCA = None

    for mode in ["train", "val", "test"]:
        print(f"Loading {mode} dataset")
        mode_config = dataset_config[mode]
        if mode_config["name"][0] == "swan":
            swan_mode = mode_config["name"][1]
            if mode == "train":
                dataset, stats, PCA = create_swan_dataset(
                    mode_config["args"], swan_mode=swan_mode, temp_db_id=temp_db_id
                )
            elif stats:
                args = mode_config["args"]
                means, stds = stats
                args["features_means"] = means
                args["features_stds"] = stds

                if PCA:
                    args["PCA_object"] = PCA

                dataset, _, _ = create_swan_dataset(
                    args, swan_mode=swan_mode, temp_db_id=temp_db_id
                )
            else:
                raise ValueError(
                    "Stats not found for SWAN dataset. Please run SWAN dataset in train mode first"
                )

            datasets[mode] = dataset

        else:
            raise NotImplementedError(
                f"Dataset {mode_config['name']} is not implemented"
            )

    return datasets
