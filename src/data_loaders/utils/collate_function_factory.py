from .collate_metadata import collate_metadata


def collate_function_factory(mode_config):
    collate_fn_name = mode_config["collate_fn"]["name"]

    if collate_fn_name == "collate_metadata":
        return collate_metadata
    elif collate_fn_name == "default":
        return None
    else:
        raise NotImplementedError(
            f"Collate function {collate_fn_name} is not implemented"
        )
