from torch.utils.data._utils.collate import default_collate
import torch
import numpy as np
from typing import Tuple


def collate_metadata(batch) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    # Separate the tensors and the metadata
    batch_data = [item[0] for item in batch]
    batch_label = [item[1] for item in batch]
    batch_metadata = [item[2] for item in batch]

    # Use PyTorch's default collate function for the data and labels
    collated_data = default_collate(batch_data)
    collated_label = default_collate(batch_label)

    # Collate the metadata manually
    collated_metadata = {}
    for key in batch_metadata[0].keys():
        # Need to handle strings differently, because they are not tensors
        if isinstance(batch_metadata[0][key], str):
            collated_metadata[key] = np.array([meta[key] for meta in batch_metadata])
            continue

        to_stack = [
            meta[key] if type(meta[key]) == torch.Tensor else torch.tensor(meta[key])
            for meta in batch_metadata
        ]
        collated_metadata[key] = torch.stack(to_stack)

    return collated_data, collated_label, collated_metadata
