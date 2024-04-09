import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid
import torch
import pandas as pd
import time

# Set the seed for reproducibility


class PlotBinaryTimeseries:
    def __init__(
        self, n_plots: int, input_logits: bool, prob_threshold: float, regions=None
    ) -> None:
        self.n_plots = n_plots
        self.input_logits = input_logits
        self.prob_threshold = prob_threshold
        self.regions = regions
        self.random_generator = np.random.default_rng(42)

    def reset(self) -> None:
        return

    def select_regions(self, metadata: dict) -> None:
        if self.regions is not None:
            print(f"Regions are already selected: {self.regions}")
            return None

        unique_regions = np.unique(metadata["harpnum"])
        positive_indices = np.where(metadata["labels"] == 1)[0]
        positive_regions = np.unique([metadata["harpnum"][i] for i in positive_indices])

        positive_chosen = self.random_generator.choice(
            positive_regions,
            min(len(positive_regions), self.n_plots // 2),
            replace=False,
        )
        other_regions = np.setdiff1d(
            unique_regions, positive_chosen, assume_unique=True
        )
        other_chosen = self.random_generator.choice(
            other_regions, self.n_plots - len(positive_chosen), replace=False
        )

        self.regions = np.concatenate([positive_chosen, other_chosen]).tolist()

        if len(self.regions) != self.n_plots:
            raise ValueError("Incorrect number of regions selected.")

    def apply_sigmoid(self, preds):
        return 1 / (1 + np.exp(-np.array(preds)))

    def get_plots(self, metadata: dict, mode: str):
        start = time.time()
        if self.regions is None:
            self.select_regions(metadata)

        figures = {}
        for region in self.regions:
            indices = np.where(np.array(metadata["harpnum"]) == region)[0]
            labels = np.array(metadata["labels"])[indices]
            preds = np.array(metadata["preds"])[indices]
            timestamps = np.array(metadata["end_date"])[indices]
            preds = self.apply_sigmoid(preds) if self.input_logits else preds

            data_df = pd.DataFrame(
                {
                    "labels": labels,
                    "preds": preds,
                    "timestamps": timestamps,
                }
            )

            data_df["timestamps"] = pd.to_datetime(data_df["timestamps"])

            data_df = data_df.sort_values(by="timestamps")

            labels = data_df["labels"].values
            preds = data_df["preds"].values
            timestamps = data_df["timestamps"].values

            fig, ax = plt.subplots()
            ax.plot(
                timestamps,
                preds,
                label="Predictions",
                linestyle="-",
                marker="o",
                color="blue",
            )
            ax.plot(timestamps, labels, label="Labels", linestyle="-", color="orange")
            ax.axhline(y=self.prob_threshold, color="red", linestyle="--")

            ax.set_ylabel("Label/Pred. Prob")
            ax.set_title(f"Region: {region}. {mode}")
            ax.legend()

            figures[f"region_{region}"] = fig

        end = time.time()

        print(f"Plotting binary preds took {end - start} seconds.")

        return figures

    def plot(self, metadata, mode):
        return self.get_plots(metadata, mode)
