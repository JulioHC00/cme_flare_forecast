from torch.nn import BCEWithLogitsLoss
import torch
from typing import Union, Optional


class CustomBCEWithLogitsLoss(BCEWithLogitsLoss):
    def __init__(
        self,
        cost_sensitive: bool,
        reduction: str = "mean",
        pos_weight: Optional[Union[float, torch.Tensor]] = None,
        *args,
        **kwargs
    ):
        """
        Custom BCEWithLogitsLoss that supports cost sensitivity and handles tensor shapes.

        Args:
            cost_sensitive (bool): If True, a positive weight must be specified.
            reduction (str): Specifies the reduction to apply to the output.
            pos_weight (Union[None, float, torch.Tensor]): A weight of positive examples.
        """
        if cost_sensitive and pos_weight is None:
            raise ValueError("pos_weight must be specified if cost_sensitive is True")

        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

        super().__init__(reduction=reduction, pos_weight=pos_weight, *args, **kwargs)

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, metadata: dict
    ) -> torch.Tensor:
        """
        Forward pass of the loss function.

        Args:
            preds (torch.Tensor): Predicted probabilities.
            targets (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed loss.
        """
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        if targets.dtype == torch.int64:
            targets = targets.float()

        if preds.numel() == 0 or targets.numel() == 0:
            # Handling empty batches
            return torch.tensor(0.0, device=preds.device)

        return super().forward(preds, targets)
