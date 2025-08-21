import torch
import numpy as np
from itertools import combinations
import random


def generate_quartets_tensor(batch_size, dm, dm_ref, device, seed=None):
    """
    Generate quartets of distance matrices from the input dm and ref_dm.

    Args:
        batch_size (int): The number of quartets to generate in a batch.
        dm (torch.Tensor): The input distance matrix of shape (N, N).
        ref_dm (torch.Tensor): The reference distance matrix of shape (N, N).
        device (torch.device): The device to perform computations on.
        seed (int, optional): Seed for random number generator.

    Returns:
        tuple: A tuple containing:
            - dm_quartets (torch.Tensor): Quartet submatrices from dm with shape (batch_size, 4, 4).
            - dm_ref_quartets (torch.Tensor): Quartet submatrices from ref_dm with shape (batch_size, 4, 4).
    """
    # dm and ref_dm must be tensors
    if not isinstance(dm, torch.Tensor) or not isinstance(dm_ref, torch.Tensor):
        raise ValueError("dm and ref_dm must be PyTorch tensors.")

    # Squeeze batch dimension if it exists
    if dm.dim() == 3 and dm.size(0) == 1:
        dm = dm.squeeze(0)  # (N, N)
    if dm_ref.dim() == 3 and dm_ref.size(0) == 1:
        dm_ref = dm_ref.squeeze(0)  # (N, N)

    dm = dm.to(device)
    dm_ref = dm_ref.to(device)
    N = dm.size(0)

    # Create a generator if seed is provided
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    else:
        gen = None

    if batch_size > 1:
        # Sample indices for quartets
        quartet_idx = torch.multinomial(
            torch.ones(batch_size, N, device=device),
            4,
            replacement=False,
            generator=gen,
        )  # Shape: (batch_size, 4)
    else:
        # we take all the quartets idxs
        quartet_idx = torch.tensor(list(combinations(range(N), 4)), device=device)

    # Extract quartet submatrices without a loop
    idx_rows = quartet_idx.unsqueeze(2).expand(-1, -1, 4)  # Shape: (batch_size, 4, 4)
    idx_cols = quartet_idx.unsqueeze(1).expand(-1, 4, -1)  # Shape: (batch_size, 4, 4)
    dm_quartets = dm[idx_rows, idx_cols]  # Shape: (batch_size, 4, 4)
    dm_ref_quartets = dm_ref[idx_rows, idx_cols]  # Shape: (batch_size, 4, 4)

    return dm_quartets, dm_ref_quartets


def generate_quartets_tensor_from_tensor_vectorized(
    batch_size, dm, quartets_tensor, codes_tensor, device
):
    """
    Generate quartet distance matrices from pre-defined quartet indices.

    Args:
        batch_size (int): The number of quartets to use (-1 for all).
        dm (torch.Tensor): The input distance matrix of shape (B, N, N) or (N, N).
        quartets_tensor (torch.Tensor): Pre-defined quartet indices of shape (n_quartets, 4).
        codes_tensor (torch.Tensor): Codes for the quartets (unused in this function).
        device (torch.device): The device to perform computations on.

    Returns:
        tuple: A tuple containing:
            - dm_quartets (torch.Tensor): Quartet submatrices from dm.
            - dm_ref_quartets (torch.Tensor): Same as dm_quartets (for compatibility).
    """
    # Handle different input dimensions
    if dm.dim() == 3 and dm.size(0) == 1:
        dm = dm.squeeze(0)  # (N, N)
    
    dm = dm.to(device)
    quartets_tensor = quartets_tensor.to(device)
    
    n_available_quartets = quartets_tensor.size(0)
    
    if batch_size == -1 or batch_size >= n_available_quartets:
        # Use all available quartets
        selected_quartets = quartets_tensor
    else:
        # Sample random quartets
        indices = torch.randperm(n_available_quartets, device=device)[:batch_size]
        selected_quartets = quartets_tensor[indices]
    
    # Extract quartet submatrices
    quartet_idx = selected_quartets  # Shape: (actual_batch_size, 4)
    actual_batch_size = quartet_idx.size(0)
    
    idx_rows = quartet_idx.unsqueeze(2).expand(-1, -1, 4)  # Shape: (actual_batch_size, 4, 4)
    idx_cols = quartet_idx.unsqueeze(1).expand(-1, 4, -1)  # Shape: (actual_batch_size, 4, 4)
    dm_quartets = dm[idx_rows, idx_cols]  # Shape: (actual_batch_size, 4, 4)
    
    # For this function, we return the same quartets as both dm and ref
    # (this is used when we already have reference quartets stored separately)
    dm_ref_quartets = dm_quartets
    
    return dm_quartets, dm_ref_quartets


def get_quartet_dist(dm1, dm2):
    """
    Calculate the quartet distance between two sets of distance matrices.
    
    This measures how often the quartet topology (which two pairs are closest)
    differs between dm1 and dm2.

    Args:
        dm1 (torch.Tensor): First set of quartet distance matrices, shape (B, 4, 4).
        dm2 (torch.Tensor): Second set of quartet distance matrices, shape (B, 4, 4).

    Returns:
        torch.Tensor: Fraction of quartets with different topologies (scalar).
    """
    # Compute the three possible pairwise distance sums for each quartet
    dm1_sums = torch.stack(
        [
            dm1[:, 0, 1] + dm1[:, 2, 3],  # (0,1) + (2,3)
            dm1[:, 0, 2] + dm1[:, 1, 3],  # (0,2) + (1,3)
            dm1[:, 0, 3] + dm1[:, 1, 2],  # (0,3) + (1,2)
        ],
        dim=1,
    )
    dm2_sums = torch.stack(
        [
            dm2[:, 0, 1] + dm2[:, 2, 3],
            dm2[:, 0, 2] + dm2[:, 1, 3],
            dm2[:, 0, 3] + dm2[:, 1, 2],
        ],
        dim=1,
    )
    
    # Find which pairing has the top 2 largest sums (this defines the quartet topology)
    _, true_idx = torch.topk(dm1_sums, 2, dim=1)  # Shape: (B, 2)
    _, est_idx = torch.topk(dm2_sums, 2, dim=1)   # Shape: (B, 2)

    # The quartet topology is defined by which pairing has the smallest sum
    # (i.e., which pairing is NOT in the top 2)
    lowest_value_idx_true = (3 - true_idx[:, 0] - true_idx[:, 1]).to(torch.int64)
    lowest_value_idx_est = (3 - est_idx[:, 0] - est_idx[:, 1]).to(torch.int64)
    
    # Return the fraction of quartets where the topologies differ
    return (lowest_value_idx_true != lowest_value_idx_est).float().mean()


def quartet_dict_to_tensors(quartet_dict):
    """
    Convert a dictionary {(i,j,k,l) -> code} into two tensors:
      - quartets_tensor of shape (n_quartets, 4)
      - codes_tensor of shape (n_quartets,)
    """
    # Extract keys and values in a consistent order
    quartet_keys = list(quartet_dict.keys())  # list of (i,j,k,l)
    quartet_codes = [quartet_dict[k] for k in quartet_keys]

    # Convert to tensors
    quartets_tensor = torch.tensor(quartet_keys, dtype=torch.int)  # (n_quartets, 4)
    codes_tensor = torch.tensor(quartet_codes, dtype=torch.int)  # (n_quartets,)

    return quartets_tensor, codes_tensor 