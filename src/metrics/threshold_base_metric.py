from .base_metric import BaseMetric
import torch


class ThresholdBasedMetric(BaseMetric):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.binary_predictions = []
        self.smooth_predictions = []
        self.actuals = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor, *args, **kwargs):
        _ = args, kwargs
        y_pred_means = preds[:, 0]
        binary_pred = self.convert_to_binary(y_pred_means)
        binary_actuals = self.convert_to_binary(targets)
        self.binary_predictions.extend(binary_pred.tolist())
        self.smooth_predictions.extend(y_pred_means.tolist())
        self.actuals.extend(binary_actuals.tolist())

    def convert_to_binary(self, y_pred_means):
        return (y_pred_means < self.threshold).int()

    def compute(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def reset(self):
        self.binary_predictions = []
        self.smooth_predictions = []
        self.actuals = []
