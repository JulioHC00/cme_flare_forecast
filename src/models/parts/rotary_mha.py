import torch.nn as nn
from math import sqrt
import torch
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from typing import Optional


class RotaryMHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        rot_dims: int,
        num_heads: int,
        max_seq_length: int = 200,
        dropout: Optional[float] = 0,
        theta: Optional[float] = 10000,
        is_causal: Optional[bool] = False,
        apply_rope: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.rot_dims = rot_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.max_seq_length = max_seq_length
        self.apply_rope = apply_rope

        self.softmax = nn.Softmax(dim=-1)

        # Check rot_dims is even

        assert self.rot_dims % 2 == 0, "rot_dims must be even"

        # Assert embed_dim is divisible by num_heads
        assert (
            self.embed_dim % self.num_heads == 0
        ), "embed_dim must be divisible by num_heads"

        # Assert rot_dims is < embed_dim
        assert self.rot_dims < self.embed_dim, "rot_dims must be less than embed_dim"

        if self.apply_rope:
            self.rotary_emb = RotaryEmbedding(dim=self.rot_dims, theta=theta)
        else:
            self.rotary_emb = None

        self.head_dim = self.embed_dim // self.num_heads

        # Input projection

        self.input_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # Generate the base mask
        # Shape is (1, )
        if self.is_causal:
            self.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(
                        1,
                        self.max_seq_length,
                        self.max_seq_length,
                        dtype=torch.bool,
                    )
                ),
            )
        else:
            self.register_buffer(
                "mask",
                torch.ones(
                    1,
                    self.max_seq_length,
                    self.max_seq_length,
                    dtype=torch.bool,
                ),
            )

    def forward(self, x, input_mask=None):
        # x shape is expected to be (batch_size, seq_len, input_dim)
        # Let's check

        assert x.shape[-1] == self.embed_dim, "Input shape does not match embed_dim"
        assert x.ndim == 3, "Input shape must be 3D"
        assert (
            x.shape[1] <= self.max_seq_length
        ), "Sequence length exceeds max_seq_length"

        x_proj = (
            self.input_proj(x)
            .reshape(x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )  # (3, N, H, S, He)

        q, k, v = x_proj  # (N, H, S, He)

        # Now need to apply the rotations

        if self.apply_rope:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # q, k = self.rotary_emb.rotate_queries_and_keys(q, k)

        # The mask must be of shape (N, seq_len, seq_len) indicating for
        # element (i, j) whether the value at position i is allowed to attend
        # to position j. A value of 0 indicates that position i is not allowed
        # to attend to position j, while a value of 1 indicates that position i
        # is allowed to attend to position j.
        # So, a casual mask would have a triangular shape, with the upper right
        # part of the matrix being all zeros.
        # Attending to everything would mean a mask of all ones.
        # Meanwhile, invalid entries would be represented by a cross
        # centred in the diagonal at position (k, k) where k is the position
        # being masked. (k,k) would be 1 to allow the diagonal to attend to
        # itself, while the rest of the row and column would be 0.
        # The input mask is simple, it has shape (N, S) saying which elements
        # in the sequence are not valid. We need to take the base mask and adapt it

        full_mask = self.mask[:, : x.shape[1], : x.shape[1]]

        # Add the input mask
        if input_mask is not None:
            # Ensure mask is a boolean tensor
            input_mask = input_mask.to(torch.bool)

            # Invert the mask for easier manipulation: True for invalid positions
            inv_mask = ~input_mask

            # Create a base square mask with all True (allowing all to attend)
            base_square_mask = torch.ones(
                (input_mask.size(0), input_mask.size(1), input_mask.size(1)),
                dtype=torch.bool,
                device=input_mask.device,
            )

            # Use broadcasting to create the cross pattern: block attention to/from invalid positions
            cross_mask = inv_mask.unsqueeze(1) | inv_mask.unsqueeze(2)

            # Apply cross mask to the base square mask, turning off attention where needed
            modified_square_mask = base_square_mask & ~cross_mask

            # Ensure self-attention is allowed by setting diagonal to True
            diag_indices = torch.arange(input_mask.size(1), device=input_mask.device)
            modified_square_mask[:, diag_indices, diag_indices] = True

            # Combine with the base (causal) mask if it exists
            full_mask = full_mask & modified_square_mask

        # And add the head dimension, and repeat
        full_mask = full_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # Now we can do the attention

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.head_dim)

        # Ensure mask is a boolean tensor and expand it to cover the head dimension
        attn_scores = torch.where(
            full_mask.to(attn_scores.device),
            attn_scores,
            torch.tensor(float("-inf")).to(attn_scores.device),
        )

        # Apply softmax to get attention probabilities
        attn_probs = self.softmax(attn_scores)

        # Optionally apply dropout
        if self.dropout is not None:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Multiply by values to get the final output
        attn_out = torch.matmul(attn_probs, v)

        # Concatenate the heads and project back to the input dimension
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[1], self.embed_dim
        )

        out = self.out_proj(attn_out)

        return out
