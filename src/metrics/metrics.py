import numpy as np
import torch
from .base_metric import BaseMetric
from .threshold_base_metric import ThresholdBasedMetric
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from torch.nn.functional import sigmoid
from wandb.plot import confusion_matrix as wandb_confusion_matrix
from wandb.plot import roc_curve
from src.utils.metrics.true_skill_statistic import (
    true_skill_statistic,
    vectorized_true_skill_statistic,
)
from typing import Union, Any


class CoverageProbability(BaseMetric):
    """
    A class to compute the Coverage Probability of a quantile regression model.

    Methods:
    reset(): Resets the internal counters to zero.
    update(y_actual, q_lower, q_upper): Updates the counters with new data.
    compute(): Computes and returns the coverage probability.
    """

    def __init__(self, device=None):
        """
        Initialize the CoverageProbability class by resetting internal counters.
        """
        self.reset()
        self.device = device

    def reset(self):
        """
        Resets the internal counters to zero.
        """
        self.total = np.longlong(0)  # Total number of data points
        self.covered = np.longlong(
            0
        )  # Number of data points covered by the quantile range

    def update(self, targets: torch.Tensor, preds: torch.Tensor):
        """
        Update the counters with new data.

        Parameters:
        y_actual (numpy array): The actual target values.
        q_lower (numpy array): The lower quantile estimates.
        q_upper (numpy array): The upper quantile estimates.
        """
        q_lower = preds[:, 0]
        q_upper = preds[:, -1]

        self.total += len(targets)
        self.covered += torch.sum((targets >= q_lower) & (targets <= q_upper)).item()

    def compute(self):
        """
        Compute and return the coverage probability.

        Returns:
        float: The coverage probability.
        """
        return self.covered / self.total if self.total > 0 else 0

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class PredictionIntervalLength(BaseMetric):
    """
    A class to compute the average Prediction Interval Length for a quantile regression model.

    Methods:
    reset(): Resets the internal counters to zero.
    update(q_lower, q_upper): Updates the counters with new data.
    compute(): Computes and returns the average prediction interval length.
    """

    def __init__(self, device=None):
        """
        Initialize the PredictionIntervalLength class by resetting internal counters.
        """
        self.reset()
        self.device = device

    def reset(self):
        """
        Resets the internal counters to zero.
        """
        self.total_length = np.float64(0)  # Total length of all prediction intervals
        self.count = np.ulonglong(0)  # Total number of intervals

    def update(self, targets: torch.Tensor, preds: torch.Tensor):
        """
        Update the counters with new data.

        Parameters:
        q_lower (numpy array): The lower quantile estimates.
        q_upper (numpy array): The upper quantile estimates.
        """
        # Even though we don't need y_actual, we still need to pass it in to
        # keep the interface consistent with CoverageProbability.
        _ = targets  # Not used here

        q_lower = preds[:, 0]
        q_upper = preds[:, -1]

        self.total_length += torch.sum(q_upper - q_lower).item()
        self.count += len(q_lower)

    def compute(self):
        """
        Compute and return the average prediction interval length.

        Returns:
        float: The average prediction interval length.
        """
        return self.total_length / self.count if self.count > 0 else 0

    def is_better(self, new_metric, old_metric):
        return new_metric < old_metric


class BrierSkillScore(BaseMetric):
    def __init__(self, input_logits: bool, device: Union[str, None] = None) -> None:
        """
        Computes the brier skill score.

        Each call to update is used to store new predictions and targets.
        Once all predictions and targets are collected, computed() can be called
        to calculate the BSS.

        The BSS is calculate by calculating the Brier Score for the model predictions
        and comparing it to the Brier Score for the climatological rate.

        That is, we compare the model to a model that always predicts the climatological rate.
        In order to do this, we require that the target are either 0 and 1 and the predictions
        are probabilities between 0 and 1. Otherwise, this won't work.

        This metric is independent of the threshold used to convert probabilities to
        predictions but it's sensitive to unbalanced datasets.
        """
        self.device = device
        self.probs = []
        self.targets = []

        # Since as usual I want to be super general, here we have to take care to the case where either we have logits or probabilities

        self.input_logits = input_logits

        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        # First, if the input are logits, we need to convert them to probabilities
        if self.input_logits:
            preds = sigmoid(preds)

        # Check all probs are between 0 and 1
        if not torch.all((preds >= 0) & (preds <= 1)):
            raise ValueError("All predictions must be between 0 and 1.")

        # Check all targets are either 1 or 0
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("All targets must be either 0 or 1.")

        # Check both have same length

        if len(targets) != len(preds):
            raise ValueError("Targets and predictions must have the same length.")

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def get_climatological_rate(self) -> np.float64:
        return np.mean(self.targets)

    def compute(self) -> float:
        model_brier_score: np.floating[Any] = brier_score_loss(self.targets, self.probs)
        climatological_rate: np.float64 = self.get_climatological_rate()

        climatological_predictions: np.ndarray = np.full_like(
            self.probs, climatological_rate
        )

        climatological_brier_score: np.floating[Any] = brier_score_loss(
            self.targets, climatological_predictions
        )

        return float(1 - (model_brier_score / climatological_brier_score))

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class BestTrueSkillStatistic(BaseMetric):
    def __init__(
        self,
        input_logits: bool,
        device: Union[str, None] = None,
    ) -> None:
        self.device = device
        self.input_logits = input_logits
        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("All targets must be either 0 or 1.")
        if self.input_logits:
            preds = sigmoid(preds)
        if not torch.all((preds >= 0) & (preds <= 1)):
            raise ValueError("All predictions must be between 0 and 1.")
        if len(targets) != len(preds):
            raise ValueError("Targets and predictions must have the same length.")

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def compute(self) -> float:
        # Generate a range of thresholds to find the best TSS
        thresholds = np.linspace(0, 1, 100)

        # Calculate TSS for each threshold
        tss_values = vectorized_true_skill_statistic(
            targets=self.targets, preds=self.probs, thresholds=thresholds
        )

        # Find the best TSS
        best_tss = np.nanmax(tss_values)

        return float(best_tss)

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class TrueSkillStatistic(BaseMetric):
    def __init__(
        self,
        input_logits: bool,
        threshold: float = 0.5,
        device: Union[str, None] = None,
    ) -> None:
        """
        Calculates the true skill score. This metrics is sensitive to the threshold used.
        However, it's not sensitive to unbalanced datasets.
        """
        self.device = device
        self.input_logits = input_logits
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        # Check all targets are either 1 or 0
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("All targets must be either 0 or 1.")

        # Now, if the inputs are logits, convert them to probabilities
        if self.input_logits:
            preds = sigmoid(preds)

        # Check all probs are between 0 and 1
        if not torch.all((preds >= 0) & (preds <= 1)):
            raise ValueError("All predictions must be between 0 and 1.")

        # Check both have same length
        if len(targets) != len(preds):
            raise ValueError("Targets and predictions must have the same length.")

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def compute(self) -> float:
        tss = true_skill_statistic(
            targets=torch.tensor(self.targets),
            preds=torch.tensor(self.probs),
            threshold=self.threshold,
            # The update method takes care of the logits
            input_logits=False,
        )

        return float(tss)

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class ConfusionMatrix(BaseMetric):
    def __init__(
        self,
        input_logits: bool,
        class_names: list,
        threshold: float = 0.5,
        device: Union[str, None] = None,
    ) -> None:
        self.device = device
        self.input_logits = input_logits
        self.threshold = threshold
        self.class_names = class_names
        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        # Check all targets are either 1 or 0
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("All targets must be either 0 or 1.")

        # Now, if the inputs are logits, convert them to probabilities
        if self.input_logits:
            preds = sigmoid(preds)

        # Check all probs are between 0 and 1
        if not torch.all((preds >= 0) & (preds <= 1)):
            raise ValueError("All predictions must be between 0 and 1.")

        # Check both have same length
        if len(targets) != len(preds):
            raise ValueError("Targets and predictions must have the same length.")

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def compute(self):
        conf_matrix = wandb_confusion_matrix(
            probs=None,
            y_true=self.targets,
            preds=list(np.array(self.probs) > self.threshold),
            class_names=self.class_names,
        )

        return conf_matrix


class ROC(BaseMetric):
    def __init__(
        self,
        input_logits: bool,
        labels: list,
        device: Union[str, None] = None,
    ) -> None:
        self.device = device
        self.input_logits = input_logits
        self.labels = labels
        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        # Check all targets are either 1 or 0
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("All targets must be either 0 or 1.")

        # Now, if the inputs are logits, convert them to probabilities
        if self.input_logits:
            preds = sigmoid(preds)

        # Check all probs are between 0 and 1
        if not torch.all((preds >= 0) & (preds <= 1)):
            raise ValueError("All predictions must be between 0 and 1.")

        # Check both have same length
        if len(targets) != len(preds):
            raise ValueError("Targets and predictions must have the same length.")

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def compute(self):
        targets = np.array(self.targets)
        probs = np.array(self.probs)

        # Wandb wants a score for each even though it's binary
        scores = np.array([1 - probs, probs]).T

        return roc_curve(targets, scores, labels=self.labels)


class AUC(BaseMetric):
    def __init__(
        self,
        input_logits: bool,
        device: Union[str, None] = None,
    ) -> None:
        self.device = device
        self.input_logits = input_logits
        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        # Check all targets are either 1 or 0
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("All targets must be either 0 or 1.")

        # Now, if the inputs are logits, convert them to probabilities
        if self.input_logits:
            preds = sigmoid(preds)

        # Check all probs are between 0 and 1
        if not torch.all((preds >= 0) & (preds <= 1)):
            raise ValueError("All predictions must be between 0 and 1.")

        # Check both have same length
        if len(targets) != len(preds):
            raise ValueError("Targets and predictions must have the same length.")

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def compute(self):
        targets = np.array(self.targets)
        probs = np.array(self.probs)
        return roc_auc_score(targets, probs)

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class AveragePrecisionScore(BaseMetric):
    def __init__(
        self,
        input_logits: bool,
        device: Union[str, None] = None,
    ) -> None:
        self.device = device
        self.input_logits = input_logits
        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        # Check all targets are either 1 or 0
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("All targets must be either 0 or 1.")

        # Now, if the inputs are logits, convert them to probabilities
        if self.input_logits:
            preds = sigmoid(preds)

        # Check all probs are between 0 and 1
        if not torch.all((preds >= 0) & (preds <= 1)):
            raise ValueError("All predictions must be between 0 and 1.")

        # Check both have same length
        if len(targets) != len(preds):
            raise ValueError("Targets and predictions must have the same length.")

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def compute(self):
        targets = np.array(self.targets)
        probs = np.array(self.probs)
        return average_precision_score(targets, probs)

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class TrueSkillStatisticProfile(BaseMetric):
    def __init__(
        self,
        input_logits: bool,
        device: Union[str, None] = None,
    ) -> None:
        """
        Calculates the true skill score. This metrics is sensitive to the threshold used.
        However, it's not sensitive to unbalanced datasets.
        """
        self.device = device
        self.input_logits = input_logits
        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        # Check all targets are either 1 or 0
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("All targets must be either 0 or 1.")

        # Now, if the inputs are logits, convert them to probabilities
        if self.input_logits:
            preds = sigmoid(preds)

        # Check all probs are between 0 and 1
        if not torch.all((preds >= 0) & (preds <= 1)):
            raise ValueError("All predictions must be between 0 and 1.")

        # Check both have same length
        if len(targets) != len(preds):
            raise ValueError("Targets and predictions must have the same length.")

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def get_climatological_rate(self) -> np.float64:
        return np.mean(self.targets)

    def compute(self):
        thresholds = np.linspace(0, 1, 100)

        tss_values = []

        for threshold in thresholds:
            preds = np.array(self.probs) > threshold
            conf_matrix = confusion_matrix(self.targets, preds)

            # True positive
            tp = conf_matrix[1, 1]
            # False positive
            fp = conf_matrix[0, 1]
            # True negative
            tn = conf_matrix[0, 0]
            # False negative
            fn = conf_matrix[1, 0]

            positive = tp + fn
            negative = fp + tn

            tss = (tp / positive) - (fp / negative)

            tss_values.append(tss)

        best_threshold = thresholds[np.argmax(tss_values)]

        return {
            "thresholds": thresholds,
            "tss_values": tss_values,
            "climatological_rate": self.get_climatological_rate(),
            "best_threshold": best_threshold,
        }


class GaussianCoverageMetric(BaseMetric):
    def __init__(self, confidence_interval=1):
        super().__init__()
        self.confidence_interval = confidence_interval
        self.covered = 0
        self.total = 0

    def update(self, targets, preds, *args, **kwargs):
        y_pred = preds[:, 0]
        y_std = preds[:, 1]
        lower_bound = y_pred - self.confidence_interval * y_std
        upper_bound = y_pred + self.confidence_interval * y_std
        self.covered += ((targets >= lower_bound) & (targets <= upper_bound)).sum()
        self.total += len(y_pred)

    def compute(self):
        return self.covered / self.total if self.total > 0 else 0

    def reset(self):
        self.covered = 0
        self.total = 0

    def is_better(self, new_metric, old_metric) -> bool:
        return new_metric > old_metric


class GaussianAverageConfidence(BaseMetric):
    def __init__(self):
        super().__init__()
        self.total_std = 0
        self.count = 0

    def update(self, targets, preds, *args, **kwargs):
        y_pred = preds[:, 0]
        y_std = preds[:, 1]
        self.total_std += y_std.sum()
        self.count += len(y_std)

    def compute(self):
        return self.total_std / self.count if self.count > 0 else 0

    def reset(self):
        self.total_std = 0
        self.count = 0

    def is_better(self, new_metric, old_metric) -> bool:
        return new_metric < old_metric


class SmoothTrueSkillStatistic(ThresholdBasedMetric):
    def __init__(self, threshold):
        super().__init__(threshold)

    def compute(self):
        return true_skill_statistic(
            targets=torch.tensor(self.actuals),
            preds=torch.tensor(self.smooth_predictions),
            threshold=self.threshold,
            input_logits=False,
        )

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class SmoothTrueSkillStatisticProfile(ThresholdBasedMetric):
    def __init__(
        self, threshold: float
    ) -> None:  # NOTE threshold here only affects labels
        super().__init__(threshold)

    def get_climatological_rate(self) -> np.float64:
        return np.mean(self.actuals)

    def compute(self) -> dict:
        thresholds = np.linspace(0, 1, 100)

        tss_values = []

        for threshold in thresholds:
            tss = true_skill_statistic(
                targets=torch.tensor(self.actuals),
                preds=torch.tensor(self.smooth_predictions),
                threshold=threshold,
                input_logits=False,
            )

            tss_values.append(tss)

        best_threshold = thresholds[np.argmax(tss_values)]

        return {
            "x": list(thresholds),
            "y": list(tss_values),
            "climatological_rate": float(self.get_climatological_rate()),
            "best_threshold": float(best_threshold),
        }

    def is_better(self, new_metric, old_metric):
        return max(new_metric["tss_values"]) > max(old_metric["tss_values"])


class SmoothAveragePrecisionScore(ThresholdBasedMetric):
    def __init__(
        self,
        threshold: float,
    ) -> None:
        super().__init__(threshold)

    def compute(self):
        targets = np.array(self.actuals)
        probs = np.array(self.smooth_predictions)
        return average_precision_score(targets, probs)

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class SmoothAUC(ThresholdBasedMetric):
    def __init__(
        self,
        threshold: float,
    ) -> None:
        super().__init__(threshold)

    def compute(self):
        targets = np.array(self.actuals)
        probs = np.array(self.smooth_predictions)
        return roc_auc_score(targets, probs)

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric


class SmoothBrierSkillScore(ThresholdBasedMetric):
    def __init__(self, threshold: float, eps: float = 1e-6) -> None:
        super().__init__(threshold)
        self.eps = eps

    def get_climatological_rate(self) -> np.float64:
        return np.mean(self.actuals)

    def compute(self) -> float:
        model_brier_score: np.floating[Any] = brier_score_loss(
            self.actuals, self.smooth_predictions
        )
        climatological_rate: np.float64 = self.get_climatological_rate()

        climatological_predictions: np.ndarray = np.full_like(
            self.smooth_predictions, climatological_rate
        )

        climatological_brier_score: float = float(
            brier_score_loss(self.actuals, climatological_predictions)
        )

        return float(
            1 - (model_brier_score / max(climatological_brier_score, self.eps))
        )

    def is_better(self, new_metric, old_metric):
        return new_metric > old_metric
