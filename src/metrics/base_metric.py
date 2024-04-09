from abc import ABC, abstractmethod
import torch


class BaseMetric(ABC):
    def __init__(self, *args, **kwargs):
        self.reset()

    @abstractmethod
    def update(self, targets: torch.Tensor, preds: torch.Tensor, *args, **kwargs):
        """
        Update the metric with new data.
        Parameters can include additional data needed for complex metrics.
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Compute the metric based on the accumulated data.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the metric to its initial state.
        """
        pass

    def is_better(self, new_metric, old_metric) -> bool:
        """
        Compare two metric values to determine if the current is better than the best.
        Override this method based on the specific metric (e.g., higher is better, or lower is better).
        """
        pass
