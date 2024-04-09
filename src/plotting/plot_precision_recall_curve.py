import matplotlib.pyplot as plt
import torch
from torch.nn.functional import sigmoid
import numpy as np
from sklearn.metrics import precision_recall_curve
import time


class PrecisionRecallCurve:
    def __init__(self, input_logits: bool) -> None:
        """
        Class for plotting the precision-recall curve using matplotlib.
        This class calculates and plots precision and recall values.
        """
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

    def get_climatological_rate(self) -> np.float64:
        return np.mean(self.targets)

    def get_plots(self, metadata: dict, mode: str):
        start = time.time()
        raw_targets = metadata["labels"].clone().detach()
        raw_preds = metadata["preds"].clone().detach()

        self.update(raw_targets, raw_preds)

        climatological_rate = self.get_climatological_rate()
        precision, recall, thresholds = precision_recall_curve(self.targets, self.probs)

        # Sample at most 200 points, sorting by recall
        precision, recall, thresholds = zip(
            *sorted(zip(precision, recall, thresholds), key=lambda x: x[1])
        )

        sample_rate = max(int(len(precision) / 200), 1)
        precision = precision[::sample_rate]
        recall = recall[::sample_rate]

        # Matplotlib figure
        fig, ax = plt.subplots()

        # Add precision-recall curve
        ax.plot(recall, precision, label="Model", color="blue")

        # Add horizontal line for no skill
        ax.axhline(y=climatological_rate, color="red", linestyle="--", linewidth=2)

        # Update layout
        ax.set_title(f"Precision-Recall Curve: {mode}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(title="Legend")

        end = time.time()

        print(f"Plotting precision-recall curve took {end - start:.2f} seconds")

        self.reset()

        return fig

    def plot(self, metadata: dict, mode: str):
        return self.get_plots(metadata, mode)
