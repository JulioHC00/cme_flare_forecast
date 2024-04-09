import torch
import torch.nn as nn
from torch.nn.modules import transformer
from .parts.rotary_transformer_block import RotaryTransformerBlock
from .parts.fc import FC


class RotaryTransformerForecastModel(nn.Module):
    """
    LSTM-based Model with Multi-Head Attention for Forecasting.

    This model processes input sequences through an LSTM, followed by multi-head attention,
    and finally through fully connected layers to produce a forecast. It supports optional
    output logits and incorporates layer normalization at various stages for stability.

    Args:
        input_size (tuple): Tuple representing (sequence_length, n_features).
        lstm_args (dict): Arguments for the lstm (LSTM) layer.
        fc_args (dict): Arguments for the fully connected layers.
        multihead_attention_args (dict): Arguments for the multi-head attention layer.
        out_logits (bool): If True, the model outputs logits; otherwise, it applies a sigmoid.
        mha_dropout (float): Dropout rate for multi-head attention.
    """

    def __init__(
        self,
        input_size: tuple,
        transformer_block_args: dict,
        n_transformer_blocks: int,
        fc_args: dict,
        avg_pool: bool,
        out_logits: bool,
        pre_fc_dropout: float = 0.0,
        fine_tuning: bool = False,
        mask_invalid: bool = False,
        shuffle_sequence: bool = False,
    ):
        super().__init__()

        embed_dim = input_size[1]

        transformer_block_args["input_shape"] = (
            input_size[0],
            embed_dim,
        )
        # Adjust input features for fully connected layer
        fc_args["input_features"] = embed_dim

        self.shuffle_sequence = shuffle_sequence

        # Need to add the class token
        self.avg_pool = avg_pool

        if not self.avg_pool:
            self.class_token = nn.Parameter(torch.rand(1, 1, embed_dim))
            self.class_token_norm = nn.LayerNorm(embed_dim)

        self.transformer_encoder = nn.Sequential(
            *[
                RotaryTransformerBlock(
                    **transformer_block_args, apply_rope=(i == 0)
                )  # Apply rope to the first layer
                for i in range(n_transformer_blocks)
            ]
        )

        self.pre_fc_dropout = nn.Dropout(p=pre_fc_dropout)

        # Initialize fully connected layers
        self.fc = FC(**fc_args)

        self.out_logits = out_logits

        # If fine_tuning, freeze all except the class token and the FC

        if fine_tuning:
            # Freeze the transformer encoder
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False

        self.mask_invalid = mask_invalid

        # Print the number of trainable parameters
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of trainable parameters: {n_params}/{total_n_params}")

        print("MASKING INVALID: ", mask_invalid)

    def masked_average_pooling(self, sequence, mask):
        # Sequence is (B, S, F)
        # Mask is (B, S)
        # Need have same shape. So if F is 0 all F are 0
        expanded_mask = mask.unsqueeze(-1).expand_as(sequence)

        # Apply the mask to zero out invalid positions
        sequence_output_masked = sequence * expanded_mask.float()

        # Sum over the sequence dimension and divide by the number of valid positions
        sum_sequence = torch.sum(sequence_output_masked, dim=1)
        sum_mask = mask.sum(dim=1, keepdim=True).float()

        # Avoid division by zero for completely masked sequences
        sum_mask = sum_mask.clamp(min=1e-9)

        average_pooled = sum_sequence / sum_mask

        return average_pooled

    def forward(self, x, metadata=None):
        """
        Forward pass of the LSTMMHAForecastModel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_features).
            metadata (Optional): Additional data for more complex models (not used in this implementation).

        Returns:
            torch.Tensor: Model output.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input of 3 dimensions, got {x.ndim}")

        b = x.shape[0]  # batch_size

        if self.shuffle_sequence:
            # Shuffle the sequence
            x = x[:, torch.randperm(x.size(1)), :]

        if not self.avg_pool:
            class_tokens = self.class_token_norm(self.class_token).repeat(b, 1, 1)

            x = torch.cat((class_tokens, x), dim=1)

        # Generate mask from metadata if provided
        if (metadata is not None) and self.mask_invalid:
            src_key_padding_mask = metadata[
                "IS_VALID"
            ].bool()  # Given our implementation, True means you use the attention score.

            # If we've added a class token, need to add a True to the start

            if not self.avg_pool:
                src_key_padding_mask = torch.cat(
                    (
                        torch.ones(b, 1, dtype=torch.bool).to(
                            src_key_padding_mask.device
                        ),
                        src_key_padding_mask,
                    ),
                    dim=1,
                )
        else:
            # All true
            src_key_padding_mask = torch.ones(
                (b, x.size(1)), dtype=torch.bool, device=x.device
            )

        # Pass through transformer encoder
        for layer in self.transformer_encoder:
            x = layer(x, src_mask=src_key_padding_mask)

        transformer_encoder_out = x

        # transformer_encoder_out = self.transformer_encoder(x)

        if self.avg_pool:
            # Average pooling over sequence sequence_length
            fc_in = self.masked_average_pooling(
                transformer_encoder_out, src_key_padding_mask
            )

        else:
            fc_in = transformer_encoder_out[:, 0, :]
            # Get the class token

        # Apply dropout
        fc_in = self.pre_fc_dropout(fc_in)

        # Pass through fully connected layers
        pred = self.fc(fc_in)

        return pred
