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


def generate_quartets_from_level(tree, level, clade_relations="known"):
    """
    Generate quartets from tree clades at specified level.
    This is the core function from research codebase.
    
    Args:
        tree: ete3 Tree object
        level: int, level in tree hierarchy 
        clade_relations: str, "known" or "unknown"
    
    Returns:
        dict: quartet_dict with (i,j,k,l) -> code mapping
    """
    import logging
    import math
    from itertools import combinations
    from ete3 import Tree
    import numpy as np
    
    # Collect leaf names and map to indices
    leaf_names = [leaf.name for leaf in tree.get_leaves()]
    n_leaves = len(leaf_names)
    name_to_idx = {nm: i for i, nm in enumerate(leaf_names)}
    
    # Get clades at specified level
    clade_roots = _collect_clades_at_level(tree, level)
    clade_info = _get_clade_list_and_roots(clade_roots, name_to_idx)
    
    logging.info(f"Found {len(clade_info)} clades at level={level}")
    
    # Generate 2+2 quartets between clades
    quartets_2by2 = _generate_between_clade_quartets(clade_info)
    
    # Generate 1+1+1+1 quartets if enough clades and relations known
    quartets_1each = {}
    if clade_relations == "known" and len(clade_info) >= 4:
        clade_dist_mat = _compute_clade_distance_by_root(clade_info)
        quartets_1each = _generate_one_leaf_per_clade_quartets(clade_info, clade_dist_mat)
    
    # Generate 2+1+1 quartets
    quartets_2by1by1 = {}
    if len(clade_info) >= 3:
        quartets_2by1by1 = _generate_two_plus_one_plus_one_quartets(clade_info)
    
    # Combine all quartets
    total_quartets = {**quartets_2by2, **quartets_1each, **quartets_2by1by1}
    
    logging.info(f"Generated {len(quartets_2by2)} 2+2, {len(quartets_1each)} 1+1+1+1, {len(quartets_2by1by1)} 2+1+1 quartets")
    logging.info(f"Total known quartets: {len(total_quartets)}")
    
    return total_quartets


def generate_quartets_from_level_sampled(tree, level, cap, seed=123):
    """
    Sample a bounded number of "known" quartets at a given tree level without
    enumerating all possibilities. Returns a dict {(i,j,k,l)->code} with up to cap items.

    Strategy:
      - Collect clades at the given level and their leaf index lists
      - Randomly sample quartets across patterns:
        * 2+2: pick two clades, then 2 leaves from each
        * 1+1+1+1: pick four distinct clades, one leaf each; compute code by root distances
        * 2+1+1: pick three clades; from one clade sample 2 leaves, from the others 1 each
      - Use a set to avoid duplicates; stop when cap is reached or attempts exhausted
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    leaf_names = [leaf.name for leaf in tree.get_leaves()]
    n_leaves = len(leaf_names)
    name_to_idx = {nm: i for i, nm in enumerate(leaf_names)}

    clade_roots = _collect_clades_at_level(tree, level)
    clade_info = _get_clade_list_and_roots(clade_roots, name_to_idx)
    n_clades = len(clade_info)
    if n_clades < 2:
        return {}

    # Precompute clade root distance matrix for 1+1+1+1 code computation
    clade_dist = _compute_clade_distance_by_root(clade_info) if n_clades >= 4 else None

    result = {}
    attempts = 0
    max_attempts = max(20000, cap * 50)

    def _code_for_indices(a, b, c, d):
        # compute code by leaf distances (4-point)
        S1 = tree.get_distance(leaf_names[a], leaf_names[b]) + tree.get_distance(leaf_names[c], leaf_names[d])
        S2 = tree.get_distance(leaf_names[a], leaf_names[c]) + tree.get_distance(leaf_names[b], leaf_names[d])
        S3 = tree.get_distance(leaf_names[a], leaf_names[d]) + tree.get_distance(leaf_names[b], leaf_names[c])
        sums = [S1, S2, S3]
        return sums.index(min(sums))

    while len(result) < cap and attempts < max_attempts:
        attempts += 1
        if n_clades >= 4 and rng.random() < 0.3:
            # 1+1+1+1
            cids = rng.choice(n_clades, size=4, replace=False)
            sel = []
            for cid in cids:
                leaves, _ = clade_info[cid]
                if len(leaves) == 0:
                    sel = []
                    break
                sel.append(rng.choice(leaves))
            if len(sel) != 4:
                continue
            i, j, k, l = sorted(sel)
            key = (i, j, k, l)
            code = _code_for_indices(i, j, k, l)
            result[key] = code
        elif n_clades >= 3 and rng.random() < 0.5:
            # 2+1+1
            cids = rng.choice(n_clades, size=3, replace=False)
            # choose which clade yields the pair
            pair_idx = rng.integers(0, 3)
            leaves_pair, _ = clade_info[cids[pair_idx]]
            if len(leaves_pair) < 2:
                continue
            pair = rng.choice(leaves_pair, size=2, replace=False)
            ones = []
            for idx, cid in enumerate(cids):
                if idx == pair_idx:
                    continue
                leaves1, _ = clade_info[cid]
                if len(leaves1) == 0:
                    ones = []
                    break
                ones.append(rng.choice(leaves1))
            if len(ones) != 2:
                continue
            sel = sorted([pair[0], pair[1], ones[0], ones[1]])
            key = tuple(sel)
            code = _code_for_indices(*sel)
            result[key] = code
        else:
            # 2+2
            c1, c2 = rng.choice(n_clades, size=2, replace=False)
            leaves1, _ = clade_info[c1]
            leaves2, _ = clade_info[c2]
            if len(leaves1) < 2 or len(leaves2) < 2:
                continue
            i, j = rng.choice(leaves1, size=2, replace=False)
            k, l = rng.choice(leaves2, size=2, replace=False)
            sel = sorted([int(i), int(j), int(k), int(l)])
            key = tuple(sel)
            code = 0  # by construction corresponds to (i,j)-(k,l)
            result[key] = code

    return result


def _collect_clades_at_level(root_node, target_level=1, current_level=0):
    """Collect clade roots at specified level."""
    clade_roots = []
    if current_level == target_level:
        clade_roots.append(root_node)
    else:
        for child in root_node.children:
            clade_roots.extend(_collect_clades_at_level(child, target_level, current_level + 1))
    return clade_roots


def _get_clade_list_and_roots(clade_roots, name_to_idx):
    """Convert clade roots to (leaf_indices, root_node) tuples."""
    clade_info = []
    for croot in clade_roots:
        leaf_names = croot.get_leaf_names()
        leaf_indices = [name_to_idx[nm] for nm in leaf_names]
        clade_info.append((leaf_indices, croot))
    return clade_info


def _generate_between_clade_quartets(clade_info):
    """Generate 2+2 quartets between pairs of clades."""
    from itertools import combinations
    quartet_dict = {}
    n_clades = len(clade_info)
    
    for c1_idx in range(n_clades):
        for c2_idx in range(c1_idx + 1, n_clades):
            leafset1, _ = clade_info[c1_idx]
            leafset2, _ = clade_info[c2_idx]
            
            for i, j in combinations(leafset1, 2):
                for k, l in combinations(leafset2, 2):
                    quartet_key = tuple(sorted((i, j, k, l)))
                    quartet_dict[quartet_key] = 0  # code=0 => (i,j)-(k,l)
    
    return quartet_dict


def _compute_clade_distance_by_root(clade_info):
    """Compute distance matrix between clade roots."""
    import numpy as np
    n_clades = len(clade_info)
    distmat = np.zeros((n_clades, n_clades))
    
    for i in range(n_clades):
        for j in range(i + 1, n_clades):
            _, root_i = clade_info[i]
            _, root_j = clade_info[j]
            d = root_i.get_distance(root_j)
            distmat[i][j] = d
            distmat[j][i] = d
    
    return distmat


def _generate_one_leaf_per_clade_quartets(clade_info, clade_dist):
    """Generate 1+1+1+1 quartets using root distances."""
    from itertools import combinations
    quartet_dict = {}
    n_clades = len(clade_info)
    
    for c1, c2, c3, c4 in combinations(range(n_clades), 4):
        # Compute partition code from root distances
        d_c1c2 = clade_dist[c1][c2]
        d_c3c4 = clade_dist[c3][c4]
        d_c1c3 = clade_dist[c1][c3]
        d_c2c4 = clade_dist[c2][c4]
        d_c1c4 = clade_dist[c1][c4]
        d_c2c3 = clade_dist[c2][c3]
        
        S1 = d_c1c2 + d_c3c4  # code=0
        S2 = d_c1c3 + d_c2c4  # code=1
        S3 = d_c1c4 + d_c2c3  # code=2
        sums = [S1, S2, S3]
        best_code = sums.index(min(sums))
        
        # Generate all combinations
        leafset1, _ = clade_info[c1]
        leafset2, _ = clade_info[c2]
        leafset3, _ = clade_info[c3]
        leafset4, _ = clade_info[c4]
        
        for a in leafset1:
            for b in leafset2:
                for c in leafset3:
                    for d in leafset4:
                        quartet_key = tuple(sorted((a, b, c, d)))
                        quartet_dict[quartet_key] = best_code
    
    return quartet_dict


def _generate_two_plus_one_plus_one_quartets(clade_info):
    """Generate 2+1+1 quartets."""
    from itertools import combinations
    quartet_dict = {}
    n_clades = len(clade_info)
    
    for c1, c2, c3 in combinations(range(n_clades), 3):
        def _gen_2plus1plus1(singled, otherA, otherB):
            singled_leafset, _ = clade_info[singled]
            leafsetA, _ = clade_info[otherA]
            leafsetB, _ = clade_info[otherB]
            for i, j in combinations(singled_leafset, 2):
                for k in leafsetA:
                    for l in leafsetB:
                        quartet_key = tuple(sorted((i, j, k, l)))
                        quartet_dict[quartet_key] = 0
        
        _gen_2plus1plus1(c1, c2, c3)
        _gen_2plus1plus1(c2, c1, c3)
        _gen_2plus1plus1(c3, c1, c2)
    
    return quartet_dict


def generate_all_quartets(tree):
    """
    Generate all quartets from a tree using 4-point condition.
    This matches the research codebase implementation.
    
    Args:
        tree: ete3 Tree object
        
    Returns:
        dict: quartet_dict with (i,j,k,l) -> code mapping for ALL quartets
    """
    import logging
    import time
    import numpy as np
    from itertools import combinations
    from multiprocessing import Pool, cpu_count
    
    start_time = time.time()
    
    # Collect leaves and create distance matrix
    leaves = tree.get_leaves()
    n_leaves = len(leaves)
    if n_leaves < 4:
        return {}
    
    logging.info(f"Generating all quartets for {n_leaves} leaves...")
    
    # Precompute pairwise distances
    distmat = np.zeros((n_leaves, n_leaves))
    for i in range(n_leaves):
        for j in range(i + 1, n_leaves):
            d = leaves[i].get_distance(leaves[j])
            distmat[i][j] = d
            distmat[j][i] = d
    
    # Generate all 4-combinations and compute codes
    all_combinations = list(combinations(range(n_leaves), 4))
    n_quartets = len(all_combinations)
    
    # Use parallel processing for large datasets
    n_processes = min(cpu_count(), 8)
    chunk_size = max(1, n_quartets // (n_processes * 10))
    chunks = [all_combinations[i:i + chunk_size] for i in range(0, n_quartets, chunk_size)]
    
    args_list = [(chunk, distmat) for chunk in chunks]
    
    quartet_dict = {}
    with Pool(n_processes) as pool:
        results = pool.map(_process_quartet_batch, args_list)
        for batch_dict in results:
            quartet_dict.update(batch_dict)
    
    elapsed = time.time() - start_time
    logging.info(f"Generated {len(quartet_dict)} quartets in {elapsed:.2f}s")
    
    return quartet_dict


def _process_quartet_batch(args):
    """Helper for parallel quartet processing."""
    quartet_indices, distmat = args
    batch_dict = {}
    
    for i, j, k, l in quartet_indices:
        # 4-point condition
        S1 = distmat[i][j] + distmat[k][l]  # code=0
        S2 = distmat[i][k] + distmat[j][l]  # code=1  
        S3 = distmat[i][l] + distmat[j][k]  # code=2
        sums = [S1, S2, S3]
        best_code = sums.index(min(sums))
        
        quartet_key = tuple(sorted((i, j, k, l)))
        batch_dict[quartet_key] = best_code
    
    return batch_dict


def generate_quartets_tensor_with_ref(batch_size, dm, quartets_tensor, ref_quartets_tensor, device):
    """
    Sample quartets from pre-defined indices and return learned dm quartets and provided ref quartets.

    Args:
      batch_size: number of quartets to sample (-1 or >= n selects all)
      dm: (B,N,N) or (N,N) tensor of distances (learned)
      quartets_tensor: (Q,4) long
      ref_quartets_tensor: (Q,4,4) float
      device: torch.device
    Returns:
      dm_quartets: (Bq,4,4)
      dm_ref_quartets: (Bq,4,4)
    """
    import torch

    if dm.dim() == 3 and dm.size(0) == 1:
        dm = dm.squeeze(0)

    Q = quartets_tensor.size(0)
    if batch_size == -1 or batch_size >= Q:
        sel_idx = torch.arange(Q, device=device)
    else:
        sel_idx = torch.randperm(Q, device=device)[:batch_size]

    sel_quartets = quartets_tensor.to(device)[sel_idx]
    sel_ref = ref_quartets_tensor.to(device)[sel_idx]

    idx_rows = sel_quartets.unsqueeze(2).expand(-1, -1, 4)
    idx_cols = sel_quartets.unsqueeze(1).expand(-1, 4, -1)
    dm_quartets = dm[idx_rows, idx_cols]
    return dm_quartets, sel_ref