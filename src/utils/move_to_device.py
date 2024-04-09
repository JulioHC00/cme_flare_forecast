from typing import Union, Any, Dict, List
import torch


def move_to_device(data: Any, device: str) -> Any:
    """
    Recursively move data to the specified device.

    Args:
        data (Any): Data to be moved. Can be a Tensor, dict, or list.
        device (str): The device to which the data should be moved.

    Returns:
        Any: Data moved to the specified device.
    """

    # Handle dictionaries recursively
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}

    # Move tensor to device
    elif isinstance(data, torch.Tensor):
        return data.to(device)

    # Handle list recursively
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]

    # If none of the above, return data as is
    else:
        return data
