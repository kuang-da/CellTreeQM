import torch
import logging
from celltreeqm.utils.utils import pairwise_distances


def compute_pairwise_distance_sums(dm_quartets):
    """
    Compute sums of selected pairwise distances for a batch of quartets.

    Args:
        dm_quartets (torch.Tensor): Distance matrix quartets of shape (B, 4, 4).

    Returns:
        torch.Tensor: Pairwise distance sums for each quartet with shape (B, 3).
    """
    return torch.stack(
        [
            dm_quartets[:, 0, 1] + dm_quartets[:, 2, 3],
            dm_quartets[:, 0, 2] + dm_quartets[:, 1, 3],
            dm_quartets[:, 0, 3] + dm_quartets[:, 1, 2],
        ],
        dim=1,
    )  # Shape: (B, 3)


def additivity_error_quartet_tensor(
    dm_quartets,
    dm_ref_quartets,
    weight_close=1,
    weight_push=10,
    push_margin=0.5,
    matching_mode="mismatched",
    device="cpu",
):
    """
    Compute the additivity error based on the top two summed pairwise distances.
    The matching_mode argument controls which quartets to consider:
    - 'mismatched': Only quartets where the estimated top 2 indices do not match the reference.
    - 'matched': Only quartets where the estimated top 2 indices match the reference.
    - 'all': All quartets.
    """
    B = dm_quartets.size(0)
    # Compute sums of selected pairwise distances
    dis_sums_ref = compute_pairwise_distance_sums(dm_ref_quartets)  # Shape: (B, 3)
    dis_sums = compute_pairwise_distance_sums(dm_quartets)  # Shape: (B, 3)

    # Find the top 2 largest sums of reference distances-
    _, top2_idx_ref = torch.topk(dis_sums_ref, 2, dim=1)  # Shape: (B, 2)
    # Find the top 2 largest sums in the estimated distances
    _, top2_idx_est = torch.topk(dis_sums, 2, dim=1)  # Shape: (B, 2)

    # Identify samples where the estimated top 2 indices do not match the reference
    mismatched = torch.any(top2_idx_ref != top2_idx_est, dim=1)  # Shape: (B,)
    matched = torch.all(top2_idx_ref == top2_idx_est, dim=1)  # Shape: (B,)

    # Initialize losses
    loss = torch.tensor(0.0, device=device)
    loss_close = torch.tensor(0.0, device=device)
    loss_push = torch.tensor(0.0, device=device)

    # Determine which quartets to consider based on matching_mode
    if matching_mode == "mismatched":
        mask = mismatched
    elif matching_mode == "matched":
        mask = matched
    elif matching_mode == "all":
        mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        raise ValueError(
            "Invalid matching_mode. Must be 'mismatched', 'matched', or 'all'."
        )

    # If there are any non-matching samples, compute the loss
    if mask.any():  # Select only the non-matching samples
        dis_sums_nm = dis_sums[mask]  # Shape: (B_nm, 3)
        top2_idx_ref_nm = top2_idx_ref[mask]  # Shape: (B_nm, 2)
        B_nm = dis_sums_nm.size(0)

        # Extract the top two values based on the reference indices
        top_2_values_est = dis_sums_nm.gather(1, top2_idx_ref_nm)

        # Identify the lowest value not in the top two
        lowest_value_idx = 3 - top2_idx_ref_nm.sum(dim=1)  # Shape: (B_nm,)
        lowest_values = dis_sums_nm[torch.arange(B_nm), lowest_value_idx]

        # Loss to make the top two values closer (using absolute difference)
        loss_close = torch.mean(
            torch.abs(top_2_values_est[:, 0] - top_2_values_est[:, 1])
        )

        # Loss to push the lowest value further away with a margin
        avg_top2 = top_2_values_est.mean(dim=1)
        diff = (avg_top2 + push_margin) - lowest_values
        loss_push = torch.mean(torch.relu(-diff))

        # Total loss
        loss = weight_close * loss_close + weight_push * loss_push

    return loss, loss_close, loss_push, top2_idx_ref


def triplet_loss_quartet_tensor_vectorized(
    dm_quartets: torch.Tensor,
    dm_ref_quartets: torch.Tensor,
    margin: float = 1.0,
    device: str = "cpu",
):
    """
    Vectorized triplet loss for quartets of shape (B,4,4).

    1) We flatten the upper triangle (i<j) of each (4x4) block into shape (B,6).
    2) Identify the pair with the smallest reference distance -> (anchor, positive).
    3) Among the leftover 2 leaves, pick the leaf farthest from anchor as negative.
    4) Compute the triplet hinge: [d(A,P) - d(A,N) + margin]_+ in the *learned* distances.

    Args:
        dm_quartets:     (B,4,4) learned distances
        dm_ref_quartets: (B,4,4) reference distances
        margin:          margin for the hinge
        device:          "cpu" or "cuda"
    Returns:
        loss (scalar)
    """

    B = dm_quartets.size(0)

    # ----------------------------------------------------------------------
    # 1) Flatten the upper triangle (i<j) of each (4x4) into (B,6).
    #    The order of pairs we choose is crucial, and must be consistent
    #    between reference and learned. For i<j with i,j in [0..3], we get pairs:
    #      (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    # ----------------------------------------------------------------------
    # Create a quick list of upper-triangle pairs
    upper_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    # We'll build (B,6) by gathering dm_quartets[b, i, j] for i<j
    ref_upper = []
    learned_upper = []
    for i, j in upper_pairs:
        ref_upper.append(dm_ref_quartets[:, i, j])
        learned_upper.append(dm_quartets[:, i, j])

    # Now stack them along dim=1, shape = (B,6)
    ref_upper = torch.stack(ref_upper, dim=1)  # (B,6)
    learned_upper = torch.stack(learned_upper, dim=1)  # (B,6)

    # We'll also define a lookup table that maps (i,j) -> index in [0..5]:
    # (since i<j, it's a simple triangular indexing)
    pair_map = torch.full((4, 4), -1, dtype=torch.long, device=device)
    for idx, (i, j) in enumerate(upper_pairs):
        pair_map[i, j] = idx
        pair_map[j, i] = idx  # for quick symmetrical access

    # ----------------------------------------------------------------------
    # 2) Identify the pair with the smallest reference distance -> anchor, positive
    #    ap_idx[b] in [0..5] tells us which pair is anchor-positive for that quartet.
    # ----------------------------------------------------------------------
    ap_idx = torch.argmin(ref_upper, dim=1)  # (B,)

    # We'll build a small tensor of shape (6,2) that, for each pair index,
    # gives the (anchor, positive) leaf indices. That way we can gather
    # anchor/positive in one step.
    idx2pair = torch.tensor(upper_pairs, device=device)  # shape (6,2)

    # anchor_arr[b], positive_arr[b] are the leaves for quartet b.
    anchor_arr = idx2pair[ap_idx, 0]  # shape (B,)
    positive_arr = idx2pair[ap_idx, 1]  # shape (B,)

    # ----------------------------------------------------------------------
    # 3) For each (anchor, positive), pick negative among leftover leaves:
    #    leftover = {0,1,2,3} \ {A,P}. We choose the leaf farthest from anchor
    #    in the reference.
    # ----------------------------------------------------------------------
    # We can do partial vectorization by constructing all leaves [0,1,2,3],
    # masking out A, P, then picking the one w/ maximum reference distance from A.
    all_leaves = torch.arange(4, device=device).unsqueeze(0).expand(B, 4)
    # shape (B,4), where each row is [0,1,2,3]

    # Create masks for "not the anchor" and "not the positive"
    # We'll compare all_leaves with anchor_arr and positive_arr
    mask_anchor = all_leaves != anchor_arr.unsqueeze(1)
    mask_positive = all_leaves != positive_arr.unsqueeze(1)
    combined_mask = mask_anchor & mask_positive  # shape (B,4)

    # leftover[b] -> the 2 leaves that are not anchor_arr[b], positive_arr[b].
    # leftover has shape (2B,) once we do all_leaves[combined_mask].
    leftover = all_leaves[combined_mask]  # shape = (2*B,)
    # Reshape to (B,2)
    leftover = leftover.view(B, 2)

    # Now, for each b, leftover[b,0] and leftover[b,1] are the 2 candidate negatives.
    # We'll pick whichever is farthest from anchor[b] in the REFERENCE distances.
    # We can do this by quickly comparing reference distances in ref_upper.

    # ref_dist_anchor[b,0] = ref distance between anchor[b] and leftover[b,0]
    # We'll use pair_map to get the pair index for (anchor, leftover).
    # Then gather from ref_upper(b, that_index).
    idx_left0 = pair_map[anchor_arr.unsqueeze(1), leftover[:, 0].unsqueeze(1)].squeeze(
        1
    )
    idx_left1 = pair_map[anchor_arr.unsqueeze(1), leftover[:, 1].unsqueeze(1)].squeeze(
        1
    )
    dist_left0 = ref_upper[torch.arange(B, device=device), idx_left0]
    dist_left1 = ref_upper[torch.arange(B, device=device), idx_left1]

    # Compare which leftover is bigger
    # negative_arr[b] will be leftover[b, 0] if dist_left0 > dist_left1,
    # otherwise leftover[b, 1].
    bigger_mask = dist_left0 > dist_left1  # shape (B,)
    negative_arr = torch.where(
        bigger_mask, leftover[:, 0], leftover[:, 1]
    )  # shape (B,)

    # ----------------------------------------------------------------------
    # 4) Compute the triplet hinge: [d(A,P) - d(A,N) + margin]_+ in the *learned* distances.
    #    We'll access learned_upper(b, pair_map[A,N]) for d(A,N), etc.
    # ----------------------------------------------------------------------
    ap_pair_idx = pair_map[anchor_arr, positive_arr]  # (B,)
    an_pair_idx = pair_map[anchor_arr, negative_arr]  # (B,)

    dist_ap = learned_upper[torch.arange(B, device=device), ap_pair_idx]
    dist_an = learned_upper[torch.arange(B, device=device), an_pair_idx]

    triplet_loss_vals = dist_ap - dist_an + margin
    triplet_loss_vals = torch.relu(triplet_loss_vals)  # (B,)
    loss = triplet_loss_vals.mean()

    return loss


def quadruplet_loss_quartet_tensor_vectorized(
    dm_quartets: torch.Tensor,
    dm_ref_quartets: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    device: str = "cpu",
):
    """
    Vectorized quadruplet loss for quartets of shape (B,4,4).

    1) We flatten the upper triangle (i<j) of each (4x4) block into shape (B,6).
    2) Identify the pair with the smallest reference distance -> (A, P).
    3) Among the leftover 2 leaves, pick:
       - N = the leaf farthest from A (in the reference).
       - N' = the other leftover leaf.
    4) Compute the two hinge terms:
       [ d(A,P) - d(A,N) + alpha ]_+
       + [ d(A,P) - d(N',N) + beta ]_+
    5) Sum over all quartets (batch), then take mean.

    Args:
        dm_quartets:     (B,4,4) learned distances
        dm_ref_quartets: (B,4,4) reference distances
        alpha:           margin for the first hinge term.
        beta:            margin for the second hinge term.
        device:          "cpu" or "cuda".

    Returns:
        loss (scalar), anchor_arr, positive_arr, negative_arr, negativeprime_arr
    """

    B = dm_quartets.size(0)

    # ---------------------
    # 1) Flatten the upper triangle (i<j) into shape (B,6)
    # ---------------------
    upper_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    ref_list = []
    learned_list = []
    for i, j in upper_pairs:
        ref_list.append(dm_ref_quartets[:, i, j])  # shape (B,)
        learned_list.append(dm_quartets[:, i, j])  # shape (B,)

    # Stack into (B,6)
    ref_upper = torch.stack(ref_list, dim=1)  # (B,6)
    learned_upper = torch.stack(learned_list, dim=1)  # (B,6)

    # We'll also define a lookup table so we can get pair indices easily
    pair_map = torch.full((4, 4), -1, dtype=torch.long, device=device)
    for idx, (i, j) in enumerate(upper_pairs):
        pair_map[i, j] = idx
        pair_map[j, i] = idx  # symmetrical

    # ---------------------
    # 2) Identify (A,P) = the pair with smallest reference distance
    # ---------------------
    ap_idx = torch.argmin(ref_upper, dim=1)  # shape (B,)

    # anchor_arr[b], positive_arr[b] are leaves in [0..3]
    idx2pair = torch.tensor(upper_pairs, device=device)  # shape (6,2)
    anchor_arr = idx2pair[ap_idx, 0]  # shape (B,)
    positive_arr = idx2pair[ap_idx, 1]  # shape (B,)

    # ---------------------
    # 3) Among leftover 2 leaves, pick:
    #    - N = farthest from A
    #    - N' = the other leftover
    # ---------------------
    all_leaves = torch.arange(4, device=device).unsqueeze(0).expand(B, 4)
    # shape (B,4), each row is [0,1,2,3]

    mask_anchor = all_leaves != anchor_arr.unsqueeze(1)
    mask_positive = all_leaves != positive_arr.unsqueeze(1)
    combined_mask = mask_anchor & mask_positive  # shape (B,4)

    # leftover[b] has the two leaves not in (A,P)
    leftover = all_leaves[combined_mask].view(B, 2)  # shape (B,2)

    # Distances from A to leftover
    # leftover[b,0], leftover[b,1]
    idx_left0 = pair_map[anchor_arr.unsqueeze(1), leftover[:, 0].unsqueeze(1)].squeeze(
        1
    )
    idx_left1 = pair_map[anchor_arr.unsqueeze(1), leftover[:, 1].unsqueeze(1)].squeeze(
        1
    )

    # reference distances from A -> leftover
    dist_left0 = ref_upper[torch.arange(B, device=device), idx_left0]
    dist_left1 = ref_upper[torch.arange(B, device=device), idx_left1]

    # negative = the one that is *farthest* from A
    bigger_mask = dist_left0 > dist_left1  # shape (B,)
    negative_arr = torch.where(
        bigger_mask, leftover[:, 0], leftover[:, 1]
    )  # shape (B,)
    negativeprime_arr = torch.where(
        ~bigger_mask, leftover[:, 0], leftover[:, 1]
    )  # shape (B,)

    # ---------------------
    # 4) Compute the two hinge terms in the *learned* distances
    #    1) [ d(A,P) - d(A,N) + alpha ]_+
    #    2) [ d(A,P) - d(N',N) + beta ]_+
    # ---------------------
    ap_pair_idx = pair_map[anchor_arr, positive_arr]
    an_pair_idx = pair_map[anchor_arr, negative_arr]
    nnprime_pair_idx = pair_map[negative_arr, negativeprime_arr]

    dist_ap = learned_upper[torch.arange(B, device=device), ap_pair_idx]
    dist_an = learned_upper[torch.arange(B, device=device), an_pair_idx]
    dist_nnprime = learned_upper[torch.arange(B, device=device), nnprime_pair_idx]

    bracket1 = dist_ap - dist_an + alpha
    bracket2 = dist_ap - dist_nnprime + beta

    # Hinge
    bracket1 = torch.relu(bracket1)  # shape (B,)
    bracket2 = torch.relu(bracket2)  # shape (B,)

    quadruplet_loss_vals = bracket1 + bracket2
    loss = quadruplet_loss_vals.mean()

    return loss, anchor_arr, positive_arr, negative_arr, negativeprime_arr
