import os
import json
import time
import logging
import random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.optim as optim

from . import (
    pairwise_distances,
    distance_error,
    generate_quartets_tensor,
    additivity_error_quartet_tensor,
    triplet_loss_quartet_tensor_vectorized,
    quadruplet_loss_quartet_tensor_vectorized,
    check_embedding,
    reconstruct_from_dm,
)
from .utils.utils import prune_dataset_to_leaves
from .utils.quartet_utils import (
    get_quartet_dist,
    quartet_dict_to_tensors,
)


@dataclass
class TrainConfig:
    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 5
    steps_per_epoch: int = 300
    quartets_per_step: int = 2048

    # Loss
    metric: str = "euclidean"
    metric_loss: str = "additivity"  # additivity|triplet|quadruplet
    weight_D: float = 0.1
    weight_P: float = 20.0
    weight_close: float = 1.0
    weight_push: float = 30.0
    margin: float = 0.1

    # Eval and reproducibility
    eval_interval: int = 50
    eval_quartets_cap: int = 100000
    dm_quartet_seed: Optional[int] = None
    seed: int = 42
    recon_method: str = "nj"  # nj|upgma|ward|single
    device: str = "cpu"


@dataclass
class TrainInputs:
    setting: str  # fully_supervised|high_level_partition|partially_labeled_leaves
    train_dataset: Any
    test_dataset: Optional[Any] = None

    # HLP known quartets (indices, codes)
    hlp_quartets: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # PLL known/unknown leaves
    pll_known_leaves: Optional[List[str]] = None
    pll_unknown_leaves: Optional[List[str]] = None


@dataclass
class TrainResult:
    metrics: Dict[str, Any]
    best_rf: float
    best_model_path: str


def _periodic_train_eval(model, dataset, metric: str, device: str, recon_method: str) -> float:
    _, emb_dm_tmp, node_names_tmp = check_embedding(dataset, model, metric, device)
    emb_tree_tmp = reconstruct_from_dm(emb_dm_tmp, node_names_tmp, method=recon_method)
    rf_train_tmp = dataset.compare_trees(emb_tree_tmp, ref_tree="topology_tree")["relative_rf"]
    return rf_train_tmp


def _final_eval(
    model,
    cfg: TrainConfig,
    out_dir: str,
    inputs: TrainInputs,
) -> Dict[str, Any]:
    device = cfg.device

    metrics: Dict[str, Any] = {}
    # Train RF
    _, emb_dm_train, node_names_train = check_embedding(
        inputs.train_dataset, model, cfg.metric, device
    )
    emb_tree_train = reconstruct_from_dm(
        emb_dm_train, node_names_train, method=cfg.recon_method
    )
    rf_train = inputs.train_dataset.compare_trees(
        emb_tree_train, ref_tree="topology_tree"
    )["relative_rf"]
    metrics["rf_train"] = rf_train

    # Test RF
    if inputs.test_dataset is not None:
        _, emb_dm_test, node_names_test = check_embedding(
            inputs.test_dataset, model, cfg.metric, device
        )
        emb_tree_test = reconstruct_from_dm(
            emb_dm_test, node_names_test, method=cfg.recon_method
        )
        rf_test = inputs.test_dataset.compare_trees(
            emb_tree_test, ref_tree="topology_tree"
        )["relative_rf"]
        metrics["rf_test"] = rf_test

    # Build learned and reference DMs for quartet evaluation
    pts_mtx_test = (
        torch.tensor(inputs.test_dataset.get_node_mtx()["node_mtx"], dtype=torch.float)
        .unsqueeze(0)
        .to(device)
    )
    dm_test = pairwise_distances(model(pts_mtx_test), metric=cfg.metric).to(device)
    dm_test = dm_test.squeeze(0)
    dm_ref_test = inputs.test_dataset.ref_dm
    if not isinstance(dm_ref_test, torch.Tensor):
        dm_ref_test = torch.tensor(dm_ref_test, dtype=torch.float32)
    dm_ref_test = dm_ref_test.to(device)

    eval_bs = cfg.eval_quartets_cap if cfg.eval_quartets_cap > 0 else -1
    if eval_bs > 0 and cfg.dm_quartet_seed is not None:
        torch.manual_seed(cfg.dm_quartet_seed)

    if inputs.setting == "high_level_partition" and inputs.hlp_quartets is not None:
        qk_tensor, _qk_codes = inputs.hlp_quartets

        def _slice(dm, idx):
            rows = idx.unsqueeze(2).expand(-1, -1, 4)
            cols = idx.unsqueeze(1).expand(-1, 4, -1)
            return dm[rows, cols]

        def _select_from_tensor(t, k):
            if k <= 0 or k >= t.size(0):
                return t
            perm = torch.randperm(t.size(0), device=device)[:k]
            return t[perm]

        def _sample_random_quartets(n, k):
            # sample k 4-combinations uniformly at random (approx) by rejection if duplicates within a quartet
            if k <= 0:
                return None
            result = []
            tried = 0
            max_trials = max(1000, 10 * k)
            while len(result) < k and tried < max_trials:
                tried += 1
                cand = torch.randperm(n, device=device)[:4].sort()[0]
                tup = tuple(cand.tolist())
                result.append(cand)
            if not result:
                return None
            return torch.stack(result, dim=0)

        n_leaves = dm_test.size(0)
        sel_known = _select_from_tensor(qk_tensor, eval_bs)

        # Unknown: reject if in known set
        known_set = {tuple(sorted(qk_tensor[i].tolist())) for i in range(qk_tensor.size(0))}
        sel_unk = []
        target = eval_bs if eval_bs > 0 else min(50000, qk_tensor.size(0))
        tried = 0
        max_trials = max(2000, 20 * (target if target > 0 else 1000))
        while (len(sel_unk) < target) and (tried < max_trials):
            tried += 1
            cand = torch.randperm(n_leaves, device=device)[:4].sort()[0]
            tup = tuple(cand.tolist())
            if tup not in known_set:
                sel_unk.append(cand)
        if sel_unk:
            sel_unk = torch.stack(sel_unk, dim=0)
        else:
            sel_unk = _sample_random_quartets(n_leaves, min(1000, target if target > 0 else 1000))

        # All: sample independent of known/unknown
        sel_all = _sample_random_quartets(n_leaves, target if target > 0 else 1000)

        if sel_all is not None:
            dmq_all, dmr_all = _slice(dm_test, sel_all), _slice(dm_ref_test, sel_all)
            metrics["quartet_dist_all"] = get_quartet_dist(dmq_all, dmr_all).item()
        if sel_known is not None and sel_known.numel() > 0:
            dmq_known, dmr_known = _slice(dm_test, sel_known), _slice(dm_ref_test, sel_known)
            metrics["quartet_dist_known"] = get_quartet_dist(dmq_known, dmr_known).item()
        if sel_unk is not None and sel_unk.numel() > 0:
            dmq_unk, dmr_unk = _slice(dm_test, sel_unk), _slice(dm_ref_test, sel_unk)
            metrics["quartet_dist_unknown"] = get_quartet_dist(dmq_unk, dmr_unk).item()

    elif inputs.setting == "partially_labeled_leaves" and inputs.pll_known_leaves is not None:
        # Sample quartets and bucket by known/partial/unknown by membership
        leaf_names = inputs.test_dataset.leave_names
        name_to_idx = {name: i for i, name in enumerate(leaf_names)}
        known_set = set(inputs.pll_known_leaves)

        def sample_buckets(n, target):
            known_list, partial_list, unknown_list, all_list = [], [], [], []
            tried = 0
            max_trials = max(5000, 30 * (target if target > 0 else 1000))
            while tried < max_trials and (len(known_list) < target or len(partial_list) < target or len(unknown_list) < target or len(all_list) < target):
                tried += 1
                cand = torch.randperm(n, device=device)[:4].sort()[0]
                all_list.append(cand)
                names = [leaf_names[i] for i in cand.tolist()]
                kcount = sum(1 for nm in names if nm in known_set)
                if kcount == 4 and len(known_list) < target:
                    known_list.append(cand)
                elif kcount == 0 and len(unknown_list) < target:
                    unknown_list.append(cand)
                elif 0 < kcount < 4 and len(partial_list) < target:
                    partial_list.append(cand)
            def _stack(lst):
                return torch.stack(lst, dim=0) if lst else None
            return _stack(all_list[:target] if target > 0 else all_list), _stack(known_list), _stack(partial_list), _stack(unknown_list)

        n_leaves = dm_test.size(0)
        target = eval_bs if eval_bs > 0 else 1000
        sel_all, sel_known, sel_part, sel_unk = sample_buckets(n_leaves, target)

        def _slice(dm, idx):
            rows = idx.unsqueeze(2).expand(-1, -1, 4)
            cols = idx.unsqueeze(1).expand(-1, 4, -1)
            return dm[rows, cols]

        if sel_all is not None:
            dmq_all, dmr_all = _slice(dm_test, sel_all), _slice(dm_ref_test, sel_all)
            metrics["quartet_dist_all"] = get_quartet_dist(dmq_all, dmr_all).item()
        if sel_known is not None:
            dmq_known, dmr_known = _slice(dm_test, sel_known), _slice(dm_ref_test, sel_known)
            metrics["quartet_dist_known"] = get_quartet_dist(dmq_known, dmr_known).item()
        if sel_part is not None:
            dmq_part, dmr_part = _slice(dm_test, sel_part), _slice(dm_ref_test, sel_part)
            metrics["quartet_dist_partial"] = get_quartet_dist(dmq_part, dmr_part).item()
        if sel_unk is not None:
            dmq_unk, dmr_unk = _slice(dm_test, sel_unk), _slice(dm_ref_test, sel_unk)
            metrics["quartet_dist_unknown"] = get_quartet_dist(dmq_unk, dmr_unk).item()

        # Evaluate RF on unknown subset
        test_unknown_dataset = prune_dataset_to_leaves(inputs.test_dataset, inputs.pll_unknown_leaves)
        _, emb_dm_unknown, node_names_unknown = check_embedding(
            test_unknown_dataset, model, cfg.metric, device
        )
        emb_tree_unknown = reconstruct_from_dm(
            emb_dm_unknown, node_names_unknown, method=cfg.recon_method
        )
        rf_test_unknown = test_unknown_dataset.compare_trees(
            emb_tree_unknown, ref_tree="topology_tree"
        )["relative_rf"]
        metrics["rf_test_unknown"] = rf_test_unknown

    else:
        # Fully-supervised: sample quartets for Q-dist on test
        dm_quartets, dm_ref_quartets = generate_quartets_tensor(
            batch_size=eval_bs if eval_bs != -1 else 100000,
            dm=dm_test,
            dm_ref=dm_ref_test,
            device=device,
            seed=cfg.dm_quartet_seed if cfg.dm_quartet_seed is not None else cfg.seed,
        )
        metrics["quartet_dist_test"] = get_quartet_dist(dm_quartets, dm_ref_quartets).item()

    # Save results.json
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)

    return metrics


def train(
    model: torch.nn.Module,
    inputs: TrainInputs,
    cfg: TrainConfig,
    out_dir: str,
    logger: Optional[logging.Logger] = None,
) -> TrainResult:
    os.makedirs(out_dir, exist_ok=True)
    # Global seeding for reproducibility across Python, NumPy, and Torch
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Prepare node matrix and reference DM from train
    node_mtx_dict = inputs.train_dataset.get_node_mtx()
    pts_mtx = (
        torch.tensor(node_mtx_dict["node_mtx"], dtype=torch.float).unsqueeze(0).to(device)
    )
    dm_ref = inputs.train_dataset.ref_dm
    if not isinstance(dm_ref, torch.Tensor):
        dm_ref = torch.tensor(dm_ref, dtype=torch.float32)
    dm_ref = dm_ref.unsqueeze(0).to(device)

    # Best checkpointing: track different metrics per setting
    best_score = float("inf")
    best_model_path = os.path.join(out_dir, "best_model.pth")
    train_seed = cfg.dm_quartet_seed if cfg.dm_quartet_seed is not None else cfg.seed

    for epoch in range(cfg.epochs):
        model.train()
        max_step = max(1, cfg.steps_per_epoch)
        for step in range(max_step):
            optimizer.zero_grad()
            trans_pts_mtx = model(pts_mtx)
            dm = pairwise_distances(trans_pts_mtx, metric=cfg.metric).to(device)

            # Quartet sampling per setting
            if inputs.setting == "high_level_partition" and inputs.hlp_quartets is not None:
                qk_tensor, qk_codes = inputs.hlp_quartets
                # Slice learned/ref DMs using known quartets indices
                idx_rows = qk_tensor.unsqueeze(2).expand(-1, -1, 4)
                idx_cols = qk_tensor.unsqueeze(1).expand(-1, 4, -1)
                if cfg.quartets_per_step > 0 and qk_tensor.size(0) > cfg.quartets_per_step:
                    perm = torch.randperm(qk_tensor.size(0), device=device)[: cfg.quartets_per_step]
                    idx_rows = idx_rows[perm]
                    idx_cols = idx_cols[perm]
                dm_quartets = dm.squeeze(0)[idx_rows, idx_cols]
                dm_ref_quartets = dm_ref.squeeze(0)[idx_rows, idx_cols]
            else:
                dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                    batch_size=cfg.quartets_per_step,
                    dm=dm,
                    dm_ref=dm_ref,
                    device=device,
                    seed=train_seed,
                )

            # Metric loss
            if cfg.metric_loss == "additivity":
                loss_P, _, _, _ = additivity_error_quartet_tensor(
                    dm_quartets,
                    dm_ref_quartets,
                    cfg.weight_close,
                    cfg.weight_push,
                    cfg.margin,
                    device=device,
                )
            elif cfg.metric_loss == "triplet":
                loss_P = triplet_loss_quartet_tensor_vectorized(
                    dm_quartets, dm_ref_quartets, margin=cfg.margin, device=device
                )
            else:
                loss_P, *_ = quadruplet_loss_quartet_tensor_vectorized(
                    dm_quartets, dm_ref_quartets, alpha=0.5, beta=0.5, device=device
                )

            # Distance alignment loss
            loss_D = distance_error(pts_mtx, trans_pts_mtx, diff_norm="fro", dist_metric=cfg.metric)
            total_loss = cfg.weight_D * loss_D + cfg.weight_P * loss_P
            total_loss.backward()
            optimizer.step()

            # Periodic train RF
            if step % max(1, cfg.eval_interval) == 0:
                rf_train_tmp = _periodic_train_eval(
                    model, inputs.train_dataset, cfg.metric, cfg.device, cfg.recon_method
                )
                logging.info(
                    f"[Epoch {epoch+1}/{cfg.epochs} | Step {step}/{max_step}] train RF={rf_train_tmp:.4f} loss={total_loss.item():.4f}"
                )

        logging.info(f"Epoch {epoch+1}/{cfg.epochs} - loss={total_loss.item():.4f}")

        # End-of-epoch lightweight eval to decide best checkpoint
        model.eval()
        try:
            rf_train_tmp = _periodic_train_eval(model, inputs.train_dataset, cfg.metric, cfg.device, cfg.recon_method)
            # For fully_supervised and PLL, prefer test RF when available
            if inputs.test_dataset is not None:
                _, emb_dm_test, node_names_test = check_embedding(inputs.test_dataset, model, cfg.metric, cfg.device)
                emb_tree_test = reconstruct_from_dm(emb_dm_test, node_names_test, method=cfg.recon_method)
                rf_test_tmp = inputs.test_dataset.compare_trees(emb_tree_test, ref_tree="topology_tree")["relative_rf"]
            else:
                rf_test_tmp = float("inf")

            if inputs.setting in ("fully_supervised", "partially_labeled_leaves"):
                score = rf_test_tmp
            elif inputs.setting == "high_level_partition":
                score = rf_train_tmp
            else:
                score = rf_test_tmp

            if score < best_score:
                best_score = score
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"[Best] epoch={epoch+1} score={best_score:.4f} -> saved {best_model_path}")
        finally:
            model.train()

    # Ensure at least one checkpoint exists
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    # Final eval
    metrics = _final_eval(model, cfg, out_dir, inputs)
    return TrainResult(metrics=metrics, best_rf=min(metrics.get("rf_test", float("inf")), metrics.get("rf_train", float("inf"))), best_model_path=best_model_path)


