class MockScheduler:
    """
    Mock scheduler for when not using one
    It can return an empty state dict
    """

    def __init__(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def step(self) -> None:
        pass
