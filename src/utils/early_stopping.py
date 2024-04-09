class EarlyStopping:
    """
    Early stopping utility.

    Attributes:
    patience (int): Number of epochs with no improvement to wait before stopping the training.
    min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    counter (int): Counter for epochs with no improvement.
    best_score (float): Best score achieved so far in terms of the monitored quantity.
    stop (bool): Whether to stop the training or not.

    Methods:
    __call__(score: float): Updates the state of the early stopping mechanism.
    should_stop() -> bool: Returns whether training should be stopped.
    """

    def __init__(self, patience: int, per_min_delta: float, criterion: str = "min"):
        """
        Initialize EarlyStopping.

        Parameters:
        patience (int): Number of epochs with no improvement to wait before stopping the training.
        per_min_delta (float): Minimum change in the monitored quantity as a % of the initial score.
        """
        self.patience = patience
        self.per_min_delta = per_min_delta
        if self.per_min_delta < 0 or self.per_min_delta > 1:
            raise ValueError("per min delta must be between 0 and 1")

        self.counter = 0
        self.best_score = None
        self.stop = False
        self.criterion = criterion

        self.min_delta = None

        if self.criterion == "min":
            self.compare = lambda x, y: (x < y) and ((y - x) > self.min_delta)

        elif self.criterion == "max":
            self.compare = lambda x, y: (x > y) and ((x - y) > self.min_delta)

        else:
            raise ValueError("Invalid criterion.")

    def __call__(self, score: float) -> None:
        """
        Updates the state of the early stopping mechanism.

        Parameters:
        score (float): The current score to check against.
        """
        if self.min_delta is None:
            self.min_delta = score * self.per_min_delta

        if self.best_score is None or self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def should_stop(self) -> bool:
        """
        Returns whether training should be stopped.

        Returns:
        bool: True if training should be stopped, False otherwise.
        """
        return self.stop
