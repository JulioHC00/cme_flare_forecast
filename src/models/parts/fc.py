import torch
import torch.nn as nn
from typing import List, Optional
import warnings
from torch.nn import init


class FC(nn.Module):
    """
    A customizable classifier module for neural networks.

    This module can be appended to models like CNNs for tasks such as binary classification.
    It supports configurable layer sizes, dropout, batch normalization, and output activation.

    The first layer of the classifier already includes batchnorm and relu activation.

    Attributes:
        layers (nn.Sequential): A sequential container of layers forming the classifier.
    """

    def __init__(
        self,
        input_features: int,
        classifier_layers: List[int],
        dropout_rates: List[float],
        norm: Optional[str] = None,
        output_features: int = 1,
    ) -> None:
        """
        Initializes the Classifier class with the specified configuration.

        Args:
            input_features (int): Number of input features to the classifier.
            classifier_layers (List[int]): List of neuron counts for each layer in the classifier.
            dropout_rates (List[float]): Dropout rates for each classifier layer.
            batch_norm (bool): Whether to use batch normalization in the classifier.
            return_logits (bool): Whether to return logits or apply sigmoid at the output.

        Raises:
            ValueError: If input parameters are invalid.
        """
        super().__init__()

        # If dropout is a float, this is interpreted as the dropout rate for all layers
        if isinstance(dropout_rates, (float, int)):
            dropout_rates = [dropout_rates] * len(classifier_layers)

        # Add a warning if using batch norm with dropout
        if (norm is not None) and any(dropout_rates):
            warnings.warn(
                "Using batch normalization with dropout can lead to suboptimal results due to the "
                "interaction between the two. Consider using dropout only or batch normalization only."
            )

        if len(dropout_rates) != len(classifier_layers):
            raise ValueError(
                "Classifier layers and dropout rates must be of equal length (or dropout a single value)."
            )

        self.layers = nn.Sequential()

        for i, layer_size in enumerate(classifier_layers):
            n_in = input_features if i == 0 else classifier_layers[i - 1]
            apply_relu = False if i == 0 else True
            layer_batch_norm = False if i == 0 else norm
            self.add_layer(
                n_in, layer_size, i, dropout_rates[i], layer_batch_norm, apply_relu
            )

        # Add the final output layer
        # The final layer doesn't use ReLU nor batch norm
        if len(classifier_layers) > 0:
            self.layers.add_module(
                "final_layer", nn.Linear(classifier_layers[-1], output_features)
            )
        else:
            self.layers.add_module(
                "final_layer", nn.Linear(input_features, output_features)
            )

        # Initialize the weights of the model
        self.__initialize_weights()

    def add_layer(
        self,
        in_features: int,
        out_features: int,
        layer_idx: int,
        dropout_rate: float,
        norm: Optional[str] = None,
        apply_relu: bool = True,
    ) -> None:
        """
        Adds a layer to the classifier.

        Args:
            in_features (int): Number of input features for the layer.
            out_features (int): Number of output features for the layer.
            layer_idx (int): Index of the layer in the classifier.
            dropout_rate (float): Dropout rate for the layer.
            batch_norm (bool): Whether to apply batch normalization.
            apply_relu (bool): Whether to apply ReLU activation after the layer.
        """
        if norm == "batch":
            self.layers.add_module(
                f"batch_norm_{layer_idx}", nn.BatchNorm1d(in_features)
            )
        elif norm == "layer":
            self.layers.add_module(f"layer_norm_{layer_idx}", nn.LayerNorm(in_features))
        if apply_relu:
            self.layers.add_module(f"relu_{layer_idx}", nn.ReLU())
        if dropout_rate > 0:
            self.layers.add_module(f"dropout_{layer_idx}", nn.Dropout(dropout_rate))
        self.layers.add_module(
            f"linear_{layer_idx}", nn.Linear(in_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor to the classifier.

        Returns:
            torch.Tensor: Output tensor of the classifier.
        """
        return self.layers(x)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # He initialization for layers with ReLU activation
                init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
