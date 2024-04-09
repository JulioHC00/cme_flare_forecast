from sklearn.metrics import confusion_matrix
import numpy as np
import torch


def true_skill_statistic(
    targets: torch.Tensor, preds: torch.Tensor, threshold: float, input_logits: bool
) -> torch.Tensor:
    """
    Calculates the true skill statistic. This metrics is sensitive to the threshold used.
    However, it's not sensitive to unbalanced datasets.
    """
    # Check all targets are either 1 or 0
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("All targets must be either 0 or 1.")

    # Now, if the inputs are logits, convert them to probabilities
    if input_logits:
        preds = torch.sigmoid(preds)

    # Check all probs are between 0 and 1
    if not torch.all((preds >= 0) & (preds <= 1)):
        raise ValueError("All predictions must be between 0 and 1.")

    # Check both have same length
    if len(targets) != len(preds):
        raise ValueError("Targets and predictions must have the same length.")

    # Calculate the confusion matrix
    cm = confusion_matrix(targets, preds > threshold)

    # Calculate the true skill score
    # step by step

    tn, fp, fn, tp = cm.ravel()

    positive = tp + fn
    negative = fp + tn

    if (positive == 0) or (negative == 0):
        return torch.tensor(torch.nan)

    tss = (tp / positive) - (fp / negative)

    return tss


def vectorized_true_skill_statistic(targets, preds, thresholds):
    """
    Vectorized calculation of the True Skill Statistic for multiple thresholds.
    """
    targets = np.array(targets, dtype=bool)
    preds = np.array(preds)

    # Initialize arrays to store true positives, false positives, etc.
    tp = np.zeros_like(thresholds, dtype=float)
    fp = np.zeros_like(thresholds, dtype=float)
    fn = np.zeros_like(thresholds, dtype=float)
    tn = np.zeros_like(thresholds, dtype=float)

    # Iterate over thresholds to calculate confusion matrix elements
    for i, threshold in enumerate(thresholds):
        predicted_positives = preds > threshold
        predicted_negatives = ~predicted_positives

        tp[i] = np.sum(targets & predicted_positives)
        fp[i] = np.sum(~targets & predicted_positives)
        fn[i] = np.sum(targets & predicted_negatives)
        tn[i] = np.sum(~targets & predicted_negatives)

    # Calculate TSS for each threshold
    positive = tp + fn
    negative = fp + tn

    with np.errstate(divide="ignore", invalid="ignore"):
        tss = np.where(
            (positive == 0) | (negative == 0), np.nan, (tp / positive) - (fp / negative)
        )

    return tss
