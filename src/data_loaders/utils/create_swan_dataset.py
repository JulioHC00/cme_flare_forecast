from ..base_swan_dataset import PreProcessSWANDataset
from ..flare_class_dataset import FlareClassSWANDataset
from ..flare_cme_dataset import FlareCMESWANDataset
from ..cme_dataset import CMESWANDataset
from copy import deepcopy


def create_swan_dataset(args, swan_mode: str, temp_db_id: str):
    pre_processor = PreProcessSWANDataset(temp_db_id=temp_db_id, **args)

    table_name, stats, temp_db_path, PCA, pca_columns = pre_processor.get_output()

    dataset_args = deepcopy(args)
    dataset_args["db_path"] = temp_db_path

    print("Not using keywords: ", args["not_keywords"])
    print(dataset_args)

    # If using PCA, change columns to use
    if dataset_args["use_PCA"]:
        dataset_args["keywords"] = pca_columns
        print("USING PCA")
    else:
        print("NOT USING PCA")

    if swan_mode == "flare_class":
        dataset = FlareClassSWANDataset(temp_table_name=table_name, **dataset_args)
    elif swan_mode == "flare_cme":
        dataset = FlareCMESWANDataset(temp_table_name=table_name, **dataset_args)
    elif swan_mode == "cme":
        dataset = CMESWANDataset(temp_table_name=table_name, **dataset_args)
    else:
        raise ValueError(f"SWAN mode {swan_mode} not recognized")

    return dataset, stats, PCA
