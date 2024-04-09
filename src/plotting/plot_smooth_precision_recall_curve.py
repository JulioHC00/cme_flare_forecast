import plotly.graph_objects as go
import torch
from torch.nn.functional import sigmoid
import numpy as np
from typing import Union
from sklearn.metrics import precision_recall_curve
import pandas as pd


class GaussianPrecisionRecallCurve:
    def __init__(
        self,
        label_threshold: float = 0.5,
    ) -> None:
        self.label_threshold = label_threshold

    def get_plots(self, metadata: pd.DataFrame, mode: str):
        # Read the data
        raw_targets = torch.tensor(metadata["labels"].astype(float).to_numpy())
        raw_preds = torch.tensor(np.array(metadata["preds"].tolist()))[:, 0]

        binary_targets = (raw_targets > self.label_threshold).int().float()

        climatological_rate = binary_targets.mean()

        precision, recall, thresholds = precision_recall_curve(
            binary_targets.cpu().numpy(), raw_preds.cpu().numpy()
        )

        # Create Plotly figure
        fig = go.Figure()

        # Add No Skill Line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[climatological_rate, climatological_rate],
                mode="lines",
                line=dict(color="grey", dash="dash"),
                name="No skill",
            )
        )

        # Add Precision-Recall curve
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                line=dict(color="blue"),
                name="Model",
            )
        )

        # Update layout
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )

        return {"precission_recall_curve": fig}

    def plot(self, metadata: pd.DataFrame, mode: str):
        return self.get_plots(metadata, mode)
