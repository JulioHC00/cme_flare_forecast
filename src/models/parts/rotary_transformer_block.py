import torch.nn as nn
import torch.nn.functional as F
from .rotary_mha import RotaryMHA


class RotaryTransformerBlock(nn.Module):
    def __init__(
        self, input_shape, num_heads, ff_dim, dropout=0.0, apply_rope=True, norm="layer"
    ):
        super().__init__()
        # Input shape is (seq_len, input_dim)
        n_features = input_shape[1]
        seq_len = input_shape[0]
        self.norm_type = norm

        # LayerNorm expects shape (batch_size, seq_len, input_dim)
        if self.norm_type == "layer":
            self.norm1 = nn.LayerNorm(n_features)
        elif self.norm_type == "batch":
            self.norm1 = nn.BatchNorm1d(n_features)
        elif self.norm_type == "none":
            self.norm1 = nn.Identity()

        self.attn = RotaryMHA(
            embed_dim=n_features,
            rot_dims=n_features // num_heads,
            num_heads=num_heads,
            dropout=dropout,
            apply_rope=apply_rope,
        )

        if self.norm_type == "layer":
            self.norm2 = nn.LayerNorm(n_features)
        elif self.norm_type == "batch":
            self.norm2 = nn.BatchNorm1d(n_features)
        elif self.norm_type == "none":
            self.norm2 = nn.Identity()

        # Conv1D needs shape (batch_size, channels, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=n_features, out_channels=ff_dim, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=ff_dim, out_channels=ff_dim, kernel_size=1, bias=False
        )
        self.conv3 = nn.Conv1d(
            in_channels=ff_dim, out_channels=n_features, kernel_size=1, bias=False
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # x shape is expected to be (batch_size, seq_len, input_dim)
        res = x
        attn_output = self.attn(x, input_mask=src_mask)
        if self.norm_type == "batch":
            x = res + attn_output
            x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm1(res + attn_output)

        res = x  # Save residual for later addition

        # Conv1D expects (batch_size, channels, seq_len)
        x = F.relu(self.conv1(x.permute(0, 2, 1)))
        x = F.relu(self.conv2(x))
        x = self.conv3(x).permute(0, 2, 1)
        if self.norm_type == "batch":
            x = res + x
            x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm2(res + x)
        x = self.dropout2(x)

        return x  # Add the residual
