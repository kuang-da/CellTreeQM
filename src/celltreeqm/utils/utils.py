import torch
import torch.nn.functional as F
import numpy as np
import sys
import logging
from ete3 import Tree
import tqdist
from scipy.cluster import hierarchy as hcluster
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
import Bio.Phylo as Phylo
from io import StringIO


def dm_to_etree(dm, node_names=None, method="nj"):
    """
    Construct ete3 tree from the distance matrix of the leaves.

    Args:
        dm: distance matrix of the leaves
        node_names: names of the leaves
        method: reconstruction method ("nj", "upgma", "ward", "single")

    Returns:
        ete3.Tree: Reconstructed phylogenetic tree
    """
    # If dm is torch.Tensor, transform it to numpy ndarray
    if hasattr(dm, "numpy"):
        # if dm is in GPU, transform it to CPU
        if dm.is_cuda:
            dm = dm.cpu()
        dm = dm.numpy()

    # If node_names is dataframe.index, transform it to list
    if hasattr(node_names, "to_list"):
        node_names = node_names.to_list()

    method_name_map = {
        "ward": "ward",
        "upgma": "average",
        "single": "single",
        "nj": "nj",
    }
    method = method_name_map[method]

    if method == "nj":
        return _nj_reconstruct(dm, node_names)
    else:
        # Hierarchical clustering methods
        n = dm.shape[0]
        X = _full_to_condensed(dm)
        Z = hcluster.linkage(X, method)
        T = hcluster.to_tree(Z, rd=True)

        scipy_tree_root = T[0]
        scipy_tree_node_list = T[1]

        for node in scipy_tree_node_list:
            node.name = str(node.id)

        if node_names is not None:
            for i in range(len(node_names)):
                scipy_tree_node_list[i].name = node_names[i]

        # Create the root for ete3 tree and initialize mapping with node ids
        ete3_root = Tree()
        ete3_root.name = str(scipy_tree_root.id)
        ete3_root.dist = 0
        node_map = {scipy_tree_root.id: ete3_root}

        # BFS to copy from scipy tree to ete3 tree
        to_visit = [scipy_tree_root]
        while to_visit:
            current_scipy_node = to_visit.pop(0)
            cl_dist = current_scipy_node.dist / 2.0
            current_ete3_node = node_map[current_scipy_node.id]

            # Add children nodes
            for child in [current_scipy_node.left, current_scipy_node.right]:
                if child:
                    new_ete3_node = Tree()
                    new_ete3_node.add_features(name=child.name)
                    new_ete3_node.add_features(dist=cl_dist)
                    new_ete3_node.add_features(dist_format="{:.3f}".format(cl_dist))
                    current_ete3_node.add_child(new_ete3_node)
                    node_map[child.id] = new_ete3_node
                    to_visit.append(child)

        return ete3_root


def _nj_reconstruct(dm, names):
    """Reconstruct tree using Neighbor Joining algorithm."""
    lower_matrix = _lower_triangle_list(dm)
    dm_bio = DistanceMatrix(names=names, matrix=lower_matrix)
    constructor = DistanceTreeConstructor()
    nj_tree = constructor.nj(dm_bio)
    newick_str = StringIO()
    Phylo.write(nj_tree, newick_str, "newick")
    newick_str.seek(0)
    ete_tree = Tree(newick_str.getvalue(), format=1)
    return ete_tree


def _lower_triangle_list(matrix):
    """Convert distance matrix to lower triangle list format for Bio.Phylo."""
    n = matrix.shape[0]
    lower_triangle = []

    # Iterate over each row and gather the elements below the diagonal
    for i in range(0, n):
        row_values = []
        for j in range(i + 1):  # j goes from 0 to i
            row_values.append(matrix[i, j])
        lower_triangle.append(row_values)

    return lower_triangle


def _full_to_condensed(distance_matrix):
    """Convert full distance matrix to condensed format for scipy hierarchical clustering."""
    n = distance_matrix.shape[0]
    # Get the indices for the upper triangle, excluding the diagonal
    upper_triangle_indices = np.triu_indices(n, k=1)
    # Extract the distances using these indices
    condensed_matrix = distance_matrix[upper_triangle_indices]
    return condensed_matrix


def distance_error(
    orig_point_matrix,
    transformed_point_matrix,
    diff_norm="fro",
    dist_metric="euclidean",
):
    """
    Compute the error between pairwise distance matrices of the original and transformed points.

    Args:
        orig_point_matrix (Tensor): Tensor of shape (B, M, N), representing the original points.
        transformed_point_matrix (Tensor): Tensor of shape (B, M, K), representing the transformed points.
        diff_norm (str or int): Norm type to use for computing the error. Options are 'fro' (Frobenius norm) or any valid p-norm (e.g., 1, 2).
        dist_metric (str): Distance metric to use when computing pairwise distances. Options are 'cosine', 'euclidean', 'manhattan'.

    Returns:
        Tensor: A scalar tensor representing the mean distance error between the original and transformed points.
    """
    valid_norms = ["fro", 1, 2, "inf"]
    if diff_norm not in valid_norms:
        raise ValueError(
            f"Unsupported norm type '{diff_norm}'. Supported options are 'fro', 1, or 2."
        )
    elif diff_norm == "inf":
        diff_norm = np.inf

    if len(orig_point_matrix.size()) != 3 or len(transformed_point_matrix.size()) != 3:
        raise ValueError("The shape must be (B, M, N).")
    else:
        if orig_point_matrix.size(1) != transformed_point_matrix.size(1):
            raise ValueError(
                "The second dimension (number of points) must be the same for both input matrices."
            )

    dis_orig = pairwise_distances(orig_point_matrix, metric=dist_metric)
    dis_transformed = pairwise_distances(transformed_point_matrix, metric=dist_metric)

    # Get the number of features (dimensions) for normalization
    orig_num_features = orig_point_matrix.size(2)
    if not isinstance(orig_num_features, torch.Tensor):
        orig_num_features = torch.tensor(orig_num_features, dtype=torch.float32)

    transformed_num_features = transformed_point_matrix.size(2)

    # Normalize the pairwise distances by the square root of the number of features
    dis_orig = dis_orig / torch.sqrt(orig_num_features)
    dis_transformed = dis_transformed / torch.sqrt(
        torch.tensor(transformed_num_features, dtype=torch.float32)
    )
    error = torch.linalg.matrix_norm(dis_orig - dis_transformed, ord=diff_norm)

    # Average the error across all samples in the batch
    error = torch.mean(error)
    return error


def pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distances.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, D) or (N, D).

    Returns:
        torch.Tensor: Pairwise cosine distance matrix.
    """
    # Normalize the vectors to unit length
    x_norm = F.normalize(x, p=2, dim=-1)

    # Compute cosine similarity
    if x.dim() == 3:  # Batched case (B, N, D)
        cosine_sim = torch.bmm(x_norm, x_norm.transpose(-2, -1))
    else:  # Single case (N, D)
        cosine_sim = torch.mm(x_norm, x_norm.t())

    # Convert similarity to distance
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def pairwise_distances(embeddings, metric="euclidean", epsilon=1e-6):
    """
    Compute pairwise distances between embeddings using various distance metrics.

    Args:
        embeddings (Tensor): Input tensor of shape (B, N, n_features) representing the embeddings.
        metric (str): Distance metric to use. Options are "euclidean", "manhattan", "inf", "cosine", "poincare".
        epsilon (float): Values below this threshold will be clamped to zero.

    Returns:
        Tensor: Pairwise distance matrix of shape (B, N, N).

    Raises:
        ValueError: If the metric is unsupported.
    """
    p_norm_dict = {"euclidean": 2, "manhattan": 1, "inf": float("inf")}

    # If embeddings is numpy array, transform it to tensor
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    if metric in p_norm_dict:
        distance_matrix = torch.cdist(embeddings, embeddings, p=p_norm_dict[metric])

    elif metric == "cosine":
        return pairwise_cosine_distance(embeddings)

    # elif metric == "poincare":
    #     return pairwise_poincare_distance(embeddings, eps=epsilon)
    else:
        raise ValueError(
            "Unsupported metric. Choose 'cosine', 'euclidean', 'manhattan', or 'poincare'."
        )

    return distance_matrix


# def pairwise_distances(embeddings, metric="euclidean", epsilon=1e-6):
#     """
#     Compute pairwise distances between embeddings using the specified metric.
#     Memory-efficient implementation that avoids creating large intermediate tensors.

#     Args:
#         embeddings (torch.Tensor): Tensor of shape (B, N, D) or (N, D) representing the embeddings.
#         metric (str): Distance metric ('euclidean', 'cosine', 'manhattan').
#         epsilon (float): Small value for numerical stability.

#     Returns:
#         torch.Tensor: Pairwise distance matrix of shape (B, N, N) or (N, N).
#     """
#     if metric == "euclidean":
#         if embeddings.dim() == 3:  # Batched case (B, N, D)
#             B, N, D = embeddings.shape
#             # Use the mathematical identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
#             # This avoids creating the large (B, N, D, N) tensor

#             # Compute squared norms: (B, N)
#             norms_sq = torch.sum(embeddings**2, dim=2)

#             # Compute dot products: (B, N, N)
#             dot_products = torch.bmm(embeddings, embeddings.transpose(1, 2))

#             # Use broadcasting to compute ||a||^2 + ||b||^2 - 2<a,b>
#             distances_sq = (
#                 norms_sq.unsqueeze(2) + norms_sq.unsqueeze(1) - 2 * dot_products
#             )

#             # Clamp to avoid negative values due to numerical errors
#             distances_sq = torch.clamp(distances_sq, min=0)
#             distances = torch.sqrt(distances_sq + epsilon)

#         else:  # Single case (N, D)
#             N, D = embeddings.shape
#             # Same approach for non-batched case
#             norms_sq = torch.sum(embeddings**2, dim=1)  # (N,)
#             dot_products = torch.mm(embeddings, embeddings.t())  # (N, N)
#             distances_sq = (
#                 norms_sq.unsqueeze(1) + norms_sq.unsqueeze(0) - 2 * dot_products
#             )
#             distances_sq = torch.clamp(distances_sq, min=0)
#             distances = torch.sqrt(distances_sq + epsilon)

#     elif metric == "cosine":
#         distances = pairwise_cosine_distance(embeddings)

#     elif metric == "manhattan":
#         if embeddings.dim() == 3:  # Batched case (B, N, D)
#             # For Manhattan distance, we still need to expand, but we can do it more efficiently
#             # by processing in chunks if needed
#             B, N, D = embeddings.shape
#             if D > 1000:  # For high-dimensional data, use chunked computation
#                 distances = torch.zeros(B, N, N, device=embeddings.device)
#                 chunk_size = 1000
#                 for i in range(0, D, chunk_size):
#                     end_idx = min(i + chunk_size, D)
#                     chunk = embeddings[:, :, i:end_idx]  # (B, N, chunk_size)
#                     x1 = chunk.unsqueeze(3)  # (B, N, chunk_size, 1)
#                     x2 = chunk.unsqueeze(2)  # (B, N, 1, chunk_size)
#                     distances += torch.sum(torch.abs(x1 - x2), dim=2)
#             else:
#                 x1 = embeddings.unsqueeze(3)  # (B, N, D, 1)
#                 x2 = embeddings.unsqueeze(2)  # (B, N, 1, D)
#                 distances = torch.sum(torch.abs(x1 - x2), dim=2)
#         else:  # Single case (N, D)
#             N, D = embeddings.shape
#             if D > 1000:  # For high-dimensional data, use chunked computation
#                 distances = torch.zeros(N, N, device=embeddings.device)
#                 chunk_size = 1000
#                 for i in range(0, D, chunk_size):
#                     end_idx = min(i + chunk_size, D)
#                     chunk = embeddings[:, i:end_idx]  # (N, chunk_size)
#                     x1 = chunk.unsqueeze(1)  # (N, 1, chunk_size)
#                     x2 = chunk.unsqueeze(0)  # (1, N, chunk_size)
#                     distances += torch.sum(torch.abs(x1 - x2), dim=2)
#             else:
#                 x1 = embeddings.unsqueeze(1)  # (N, 1, D)
#                 x2 = embeddings.unsqueeze(0)  # (1, N, D)
#                 distances = torch.sum(torch.abs(x1 - x2), dim=2)

#     else:
#         raise ValueError(f"Unsupported distance metric: {metric}")

#     return distances


def reconstruct_from_dm(dm, node_names, method, unrooted=True):
    """
    Reconstruct a tree from a distance matrix using the specified method.

    Args:
        dm (numpy.ndarray): Distance matrix.
        node_names (list): List of node names corresponding to the distance matrix.
        method (str): Reconstruction method ('nj' for neighbor joining).
        unrooted (bool): Whether to return an unrooted tree.

    Returns:
        ete3.Tree: Reconstructed tree.
    """
    # The dm_to_etree function may not support the unrooted parameter
    # Let's call it without that parameter for compatibility
    return dm_to_etree(dm, node_names, method=method)


def compare_trees(tree1, tree2, unrooted_trees=False):
    """
    Compare two trees using Robinson-Foulds distance.

    Args:
        tree1, tree2: ete3.Tree objects to compare.
        unrooted_trees (bool): Whether trees are unrooted.

    Returns:
        dict: Dictionary containing RF distance and related metrics.
    """
    if unrooted_trees:
        rf_distance = tree1.robinson_foulds(tree2, unrooted_trees=True)[0]
        # For unrooted trees, the maximum RF distance is 2 * (n - 3)
        # where n is the number of leaves
        max_rf = 2 * (len(tree1.get_leaves()) - 3)
    else:
        rf_distance = tree1.robinson_foulds(tree2)[0]
        # For rooted trees, the maximum RF distance is 2 * (n - 2)
        max_rf = 2 * (len(tree1.get_leaves()) - 2)

    if max_rf == 0:
        relative_rf = 0.0
    else:
        relative_rf = rf_distance / max_rf

    return {
        "rf_distance": rf_distance,
        "max_rf": max_rf,
        "relative_rf": relative_rf,
    }


def check_embedding(dataset, model, dist_metric, device):
    """
    Check the embedding produced by the model for a given dataset.

    Args:
        dataset: Dataset object containing the data.
        model: Neural network model.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.

    Returns:
        tuple: (embeddings, distance_matrix, node_names)
    """
    model.eval()
    with torch.no_grad():
        # Get the node matrix data
        node_mtx_dict = dataset.get_node_mtx()
        pts_mtx = (
            torch.tensor(node_mtx_dict["node_mtx"], dtype=torch.float)
            .unsqueeze(0)
            .to(device)
        )
        node_names = node_mtx_dict["node_names"]

        # Get embeddings
        embeddings = model(pts_mtx)  # Shape: (1, N, D)

        # Compute distance matrix
        dm = pairwise_distances(embeddings, metric=dist_metric)
        dm = dm.squeeze(0).cpu().numpy()  # Shape: (N, N)

        return embeddings, dm, node_names


def train_reconstruct_eval(
    dataset, model, res_dict, dist_metric="euclidean", device="cpu", method="nj"
):
    """
    Evaluate reconstruction performance on training data.

    Args:
        dataset: Training dataset.
        model: Neural network model.
        res_dict: Results dictionary to update.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.
        method (str): Tree reconstruction method.
    """
    model.eval()
    with torch.no_grad():
        _, emb_dm, node_names = check_embedding(dataset, model, dist_metric, device)

        # Reconstruct tree from embedding
        emb_tree = reconstruct_from_dm(emb_dm, node_names, method=method)

        # Compare with reference tree
        emb_topo_res = dataset.compare_trees(emb_tree, ref_tree="topology_tree")

        # Store result
        res_key = f"rf_emb_topo_train"
        res_dict[res_key][method].append(emb_topo_res["relative_rf"])


def test_reconstruct_eval(
    dataset, model, res_dict, dist_metric="euclidean", device="cpu", method="nj"
):
    """
    Evaluate reconstruction performance on test data.

    Args:
        dataset: Test dataset.
        model: Neural network model.
        res_dict: Results dictionary to update.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.
        method (str): Tree reconstruction method.
    """
    model.eval()
    with torch.no_grad():
        _, emb_dm, node_names = check_embedding(dataset, model, dist_metric, device)

        # Reconstruct tree from embedding
        emb_tree = reconstruct_from_dm(emb_dm, node_names, method=method)

        # Compare with reference tree
        emb_topo_res = dataset.compare_trees(emb_tree, ref_tree="topology_tree")

        # Store result
        res_key = f"rf_emb_topo_test"
        res_dict[res_key][method].append(emb_topo_res["relative_rf"])


def test_unknown_reconstruct_eval(
    dataset, model, res_dict, dist_metric="euclidean", device="cpu", method="nj"
):
    """
    Evaluate reconstruction performance on unknown test data.

    Args:
        dataset: Unknown test dataset.
        model: Neural network model.
        res_dict: Results dictionary to update.
        dist_metric (str): Distance metric to use.
        device: Device to run computation on.
        method (str): Tree reconstruction method.
    """
    model.eval()
    with torch.no_grad():
        _, emb_dm, node_names = check_embedding(dataset, model, dist_metric, device)

        # Reconstruct tree from embedding
        emb_tree = reconstruct_from_dm(emb_dm, node_names, method=method)

        # Compare with reference tree
        emb_topo_res = dataset.compare_trees(emb_tree, ref_tree="topology_tree")

        # Store result
        res_key = f"rf_emb_topo_test_unknown"
        res_dict[res_key][method].append(emb_topo_res["relative_rf"])


def prune_dataset_to_leaves(dataset, keep_leaves):
    """
    Create a pruned copy of dataset containing only specified leaves.
    
    Args:
        dataset: Dataset object with topology_tree, data_normalized, etc.
        keep_leaves: List of leaf names to retain
        
    Returns:
        Pruned dataset copy
    """
    import copy
    from ete3 import Tree
    
    pruned = copy.deepcopy(dataset)
    
    # Prune the topology tree
    pruned_tree = dataset.topology_tree.copy()
    pruned_tree.prune(keep_leaves, preserve_branch_length=True)
    pruned.topology_tree = pruned_tree
    
    # Update leaf names and count
    pruned.leave_names = [leaf.name for leaf in pruned_tree.get_leaves()]
    pruned.n_leaves = len(pruned.leave_names)
    
    # Prune the data matrices
    if hasattr(pruned, 'data_normalized') and pruned.data_normalized is not None:
        pruned.data_normalized = pruned.data_normalized.loc[pruned.leave_names]
    if hasattr(pruned, 'data') and pruned.data is not None:
        pruned.data = pruned.data.loc[pruned.leave_names]
    
    # Recompute reference distance matrix
    from .utils import _get_path_distance_matrix
    try:
        pruned.ref_dm = _get_path_distance_matrix(pruned_tree, pruned.leave_names)
    except:
        # Fallback: recompute using tree distances
        import torch
        n = len(pruned.leave_names)
        ref_dm = torch.zeros(n, n)
        leaves = list(pruned_tree.get_leaves())
        for i in range(n):
            for j in range(i+1, n):
                dist = leaves[i].get_distance(leaves[j])
                ref_dm[i, j] = dist
                ref_dm[j, i] = dist
        pruned.ref_dm = ref_dm
    
    return pruned


def _get_path_distance_matrix(tree, leaf_names):
    """Compute path distance matrix from tree."""
    import torch
    n = len(leaf_names)
    dm = torch.zeros(n, n)
    leaves = {leaf.name: leaf for leaf in tree.get_leaves()}
    
    for i, name_i in enumerate(leaf_names):
        for j, name_j in enumerate(leaf_names):
            if i != j:
                dm[i, j] = leaves[name_i].get_distance(leaves[name_j])
    
    return dm
