import torch
import numpy as np
from typing import Any
from copy import deepcopy


def is_iterable(obj: Any) -> bool:
    """Check if an object is iterable."""
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """Merge two dictionaries recursively.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Merged dictionary.

    Raises:
        ValueError: If either argument is not a dictionary.
    """
    # Check if arguments are dictionaries
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise ValueError("Arguments must be dictionaries")

    # Copy dict1 to not modify the original
    merged = deepcopy(dict1)

    for key, value in dict2.items():
        # If key is not in dict1, add it
        if key not in merged:
            merged[key] = value
        # If key is in both dicts and the values are also dicts, merge those
        elif isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        # If key is in both dicts and the values are lists, concatenate them
        elif isinstance(merged[key], list) and isinstance(value, list):
            merged[key].extend(value)
        # If key is in both dicts and the values are torch tensors, concatenate them
        elif isinstance(merged[key], torch.Tensor) and isinstance(value, torch.Tensor):
            merged[key] = torch.cat((merged[key], value))
        # If key is in both dicts and the values are numpy arrays, concatenate them
        elif isinstance(merged[key], np.ndarray) and isinstance(value, np.ndarray):
            merged[key] = np.concatenate((merged[key], value))
        # Handle non-iterable items
        elif not is_iterable(merged[key]) and not is_iterable(value):
            merged[key] = [merged[key], value]
        else:
            raise NotImplementedError(
                f"Cannot merge values of type {type(merged[key])} and {type(value)}"
            )

    return merged


def main():
    # Test basic dictionary merge
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    # Test nested dictionary merge
    assert merge_dicts({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}

    # Test list concatenation
    assert merge_dicts({"a": [1, 2]}, {"a": [3, 4]}) == {"a": [1, 2, 3, 4]}

    # Test tensor concatenation
    tensor1 = torch.tensor([1, 2])
    tensor2 = torch.tensor([3, 4])
    merged_tensor = torch.tensor([1, 2, 3, 4])
    assert torch.equal(merge_dicts({"a": tensor1}, {"a": tensor2})["a"], merged_tensor)

    # Test numpy array concatenation
    array1 = np.array([1, 2])
    array2 = np.array([3, 4])
    merged_array = np.array([1, 2, 3, 4])
    assert np.array_equal(merge_dicts({"a": array1}, {"a": array2})["a"], merged_array)

    # Test handling of non-iterables
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": [1, 2]}

    print("All tests passed.")


if __name__ == "__main__":
    main()
