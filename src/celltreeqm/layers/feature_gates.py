import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class FeatureGates(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=32,
        hidden_dim=64,
        tau=1.0,
        mode="gumbel",
        device="cpu",
        init_ones=False,
    ):
        """
        Feature gating module.

        Args:
            input_dim (int): Number of input features.
            embed_dim (int): Embedding dimension for gating.
            hidden_dim (int): Hidden dimension for the MLP in gating.
            tau (float): Temperature parameter for Gumbel softmax.
            mode (str): Feature gating mode. One of ['sigmoid', 'softmax', 'gumbel', 'none'].
        """
        super().__init__()

        assert mode in ["sigmoid", "softmax", "gumbel", "none"]

        self.tau = tau
        self.mode = mode
        self.input_dim = input_dim
        self.device = device

        self.frozen_mask = None
        # Keep track of gating
        self.gated_weights = None
        self.prev_gated_weights = None

        if mode in ["softmax", "gumbel"]:
            self.feature_embed = nn.Parameter(torch.randn(input_dim, embed_dim) * 0.01)
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            # If requested, initialize so that "on" logit >> "off" logit
            if init_ones and mode == "gumbel":
                with torch.no_grad():
                    final_linear = self.mlp[-1]  # nn.Linear(hidden_dim, 2)
                    # Optionally zero out weights so the final result relies mainly on bias
                    final_linear.weight.fill_(0.0)
                    # Make bias for "on" significantly higher than "off"
                    final_linear.bias[0] = 0.0  # off logit
                    final_linear.bias[1] = (
                        5.0  # on logit (could go even higher if needed)
                    )

        elif mode == "sigmoid":
            self.gates = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        """
        Apply gating to input features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C) or (N, C).

        Returns:
            torch.Tensor: Gated tensor of the same shape as input.
        """
        if self.mode == "none":
            # No gating - just pass through
            return x
            
        if self.frozen_mask is not None:
            return x * self.frozen_mask  # shape broadcast OK

        if self.mode == "sigmoid":
            gated_weights = torch.sigmoid(self.gates)  # Constrain weights to (0, 1)

            self.gated_weights = gated_weights  # Save for penalty
            return x * gated_weights

        if self.mode in ["softmax", "gumbel"]:
            logits = self.mlp(self.feature_embed)  # Shape: (C, 2)

            if self.mode == "softmax":
                softmax_output = F.softmax(logits, dim=-1)
                gated_weights = softmax_output[..., 1]  # Shape: (C,)
            elif self.mode == "gumbel":
                gumbel_output = F.gumbel_softmax(
                    logits, tau=self.tau, hard=True, dim=-1
                )
                gated_weights = gumbel_output[..., 1]  # Shape: (C,)

            self.gated_weights = gated_weights  # Save for penalty
            return x * gated_weights

        raise ValueError(f"Invalid gating mode: {self.mode}")

    def effective_gates(self):
        """
        Return the effective gating weights for analysis.

        Returns:
            torch.Tensor: Effective gating weights for the features.
        """
        if self.mode == "none":
            return torch.ones(self.input_dim).to(self.device)
        return self.gated_weights

    def set_frozen_mask(self, mask: torch.Tensor):
        """
        Set a single gating mask to freeze gating.
        Usually you'd pass a 0/1 mask or any fixed real-value mask.

        Args:
            mask (torch.Tensor): shape (C,) with values in [0,1] or exactly {0,1}.
        """
        self.frozen_mask = mask.detach().to(self.device)

    def compute_penalty(
        self, penalty_type="retain", lambda_penalty=1.0, lambda_flips=0.0
    ):
        """
        Compute the penalty term for gating.

        Args:
            penalty_type (str): Type of penalty ('retain' or 'sparsity').
            lambda_penalty (float): Coefficient for the penalty term.

        Returns:
            torch.Tensor: The penalty term to be added to the loss.
        """
        if self.mode == "none":
            return torch.tensor(0.0, device=self.device)
            
        if lambda_penalty < 0:
            return torch.tensor(0.0, device=self.device)

        if penalty_type == "retain":
            # Encourage retaining more features
            base_penalty = (
                lambda_penalty * torch.sum(1 - self.gated_weights) / self.input_dim
            )
        elif penalty_type == "sparsity":
            # Encourage sparsity (if needed instead)
            base_penalty = (
                lambda_penalty * torch.sum(self.gated_weights) / self.input_dim
            )
        else:
            raise ValueError(f"Invalid penalty_type: {penalty_type}")

        # Flip penalty
        flip_penalty = torch.tensor(0.0, device=self.device)
        if (self.prev_gated_weights is not None) and (lambda_flips > 0.0):
            old_mask = self.prev_gated_weights > 0.5
            new_mask = self.gated_weights > 0.5
            flips = torch.sum(old_mask ^ new_mask).float()
            flip_penalty = flips * lambda_flips

        # Update prev_gated_weights
        if self.gated_weights is not None:
            self.prev_gated_weights = self.gated_weights.detach()

        return base_penalty + flip_penalty 