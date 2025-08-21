import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from celltreeqm.layers.feature_gates import FeatureGates


class CellTreeQMAttention(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        output_dim,
        dropout_data=0.2,
        dropout_metric=0.2,
        norm_method=None,
        proj_dim=1024,
        gate_mode="none",
        gate_embed_dim=32,
        gate_hidden_dim=64,
        tau=1.0,
        device="cpu",
        init_ones=False,
    ):
        """
        Encoder with optional feature gating for CellTree Quartet Matching.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension of the transformer.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            output_dim (int): Output feature dimension.
            dropout_data (float): Dropout for data projection.
            dropout_metric (float): Dropout for output.
            norm_method (str): Normalization method ('batch_norm', 'layer_norm', or None).
            proj_dim (int): Projection dimension for transformer input.
            gate_mode (str): Feature gating mode ('none', 'sigmoid', 'softmax', 'gumbel').
            gate_embed_dim (int): Embedding dimension for feature gating.
            gate_hidden_dim (int): Hidden dimension for feature gating.
            tau (float): Temperature parameter for Gumbel softmax.
            device (str): Device for computation.
            init_ones (bool): Whether to initialize gates to favor "on" state.
        """
        super().__init__()
        self.norm_method = norm_method

        logging.info(f"Input dim: {input_dim}")
        logging.info(f"Projection dim: {proj_dim}")
        logging.info(f"Gating mode: {gate_mode}")

        # Feature gates
        if gate_mode == "none":
            self.feature_gate = None
        elif gate_mode == "linear":
            self.feature_gate = None
            self.G = nn.Parameter(torch.eye(input_dim))
        else:
            self.feature_gate = FeatureGates(
                input_dim,
                embed_dim=gate_embed_dim,
                hidden_dim=gate_hidden_dim,
                tau=tau,
                mode=gate_mode,
                device=device,
                init_ones=init_ones,
            )

        # Linear projection to transformer dimension
        self.projection = nn.Linear(input_dim, proj_dim)
        nn.init.xavier_normal_(self.projection.weight)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_data,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output layer
        self.output_layer = nn.Linear(proj_dim, output_dim)

        # Normalization
        if norm_method == "batch_norm":
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm_method == "layer_norm":
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = None

        # Dropout
        self.dropout1 = nn.Dropout(dropout_data)
        self.dropout2 = nn.Dropout(dropout_metric)

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C) or (N, C).

        Returns:
            torch.Tensor: Output tensor.
        """
        if len(x.shape) == 3:
            batched = True
            B, N, C = x.shape
            x = x.view(B * N, C)
        else:
            batched = False

        # Apply feature gates
        if self.feature_gate is not None:
            x = self.feature_gate(x)
        elif hasattr(self, 'G'):
            # Linear gating with learnable matrix G
            x = x @ self.G

        # Apply the linear projection
        x = self.projection(x)
        x = self.dropout1(x)

        # Transformer encoder
        x = x.unsqueeze(1)  # Add a sequence dimension
        x = self.transformer_encoder(x)
        x = x.squeeze(1)

        # Output layer
        x = self.output_layer(x)

        # Apply normalization
        if self.norm is not None:
            x = self.norm(x)

        # Apply dropout
        x = self.dropout2(x)

        if batched:
            x = x.view(B, N, -1)

        return x 