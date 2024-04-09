import torch
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
import numpy as np
from sklearn.metrics import confusion_matrix
from src.utils.metrics.true_skill_statistic import true_skill_statistic
import time
from src.utils.metrics.true_skill_statistic import (
    vectorized_true_skill_statistic,
)  # Assume a vectorized version exists


class TrueSkillStatisticProfile:
    def __init__(self, input_logits: bool) -> None:
        self.input_logits = input_logits
        self.reset()

    def reset(self) -> None:
        self.probs = []
        self.targets = []

    def update(self, targets: torch.Tensor, preds: torch.Tensor) -> None:
        if self.input_logits:
            preds = torch.sigmoid(preds)

        self.probs.extend(preds.tolist())
        self.targets.extend(targets.tolist())

    def get_climatological_rate(self) -> np.float64:
        return np.mean(self.targets)

    def get_plots(self, metadata: dict, mode: str):
        start = time.time()

        self.update(metadata["labels"], metadata["preds"])

        climatological_rate = self.get_climatological_rate()
        thresholds = np.linspace(0, 1, 100)

        tss_values = vectorized_true_skill_statistic(
            self.targets, self.probs, thresholds
        )

        best_tss = np.max(tss_values)
        best_threshold = thresholds[np.argmax(tss_values)]

        fig, ax = plt.subplots()
        ax.plot(
            thresholds, tss_values, label="TSS", color="blue", linestyle="-", marker="o"
        )
        ax.axvline(
            x=climatological_rate,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Climatological Rate",
        )
        ax.axvline(
            x=best_threshold,
            color="green",
            linestyle="-.",
            linewidth=2,
            label="Best Threshold",
        )

        ax.set_title(f"True Skill Score Profile: {mode}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("True Skill Score")
        ax.legend(title="Legend")

        end = time.time()
        print(f"Plotting TSS took {end - start} seconds.")

        self.reset()

        return fig

    def plot(self, metadata: dict, mode: str):
        return self.get_plots(metadata, mode)
