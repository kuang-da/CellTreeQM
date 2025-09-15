import argparse
import logging
import os
import time
import torch
import torch.optim as optim

from . import (
    CellTreeQMAttention,
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
    generate_quartets_from_level,
    generate_all_quartets,
    quartet_dict_to_tensors,
    generate_quartets_tensor_from_tensor_vectorized,
)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def cmd_train(args: argparse.Namespace) -> None:
    try:
        from celltreebench.datasets.celegans import load_celegans_supervised_split
    except Exception as exc:
        raise SystemExit(
            "CellTreeBench is required for training; install and configure dataset paths."
        ) from exc

    device = torch.device(args.device)
    setup_logging(args.verbose)
    logging.info("Loading dataset ...")

    data_dir = os.path.join(args.bench_root, "data")
    uid = args.uid or time.strftime("%Y%m%d-%H%M%S")
    default_run_name = f"{args.dataset}_{args.lineage}_{args.setting}_{uid}"
    run_name = args.run_name or default_run_name

    # Experiment output directory (models, metrics, results)
    # Default experiment outputs to env or local folder, independent from CellTreeBench
    default_output_root = os.environ.get("CELLTREEQM_OUTPUT_DIR", os.path.abspath("./celltreeqm-outputs"))
    output_root = args.output_dir or default_output_root
    out_dir = os.path.join(output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Dataset artifacts directory (reusable dataset-level derivatives)
    default_artifacts_dir = os.path.join(
        args.bench_root, "data", args.dataset, args.lineage, "artifacts"
    )
    dataset_artifacts_dir = args.dataset_artifacts_dir or default_artifacts_dir
    os.makedirs(dataset_artifacts_dir, exist_ok=True)

    # Load base dataset
    full_train_dataset, full_test_dataset = load_celegans_supervised_split(
        dataset_name=args.dataset,
        lineage_name=args.lineage,
        data_dir=data_dir,
        # Write dataset-level artifacts (tree ascii/pickle, gene_list, caches)
        out_dir=dataset_artifacts_dir,
        sampling_method="biological",
        seed=args.seed,
    )
    
    # Handle partially_labeled_leaves setting
    if args.setting == "partially_labeled_leaves":
        import random
        random.seed(args.seed)
        
        # Split leaves into known/unknown
        all_leaves = full_test_dataset.leave_names  # Use full dataset for splitting
        subset_size = int(len(all_leaves) * args.known_fraction)
        known_leaves = random.sample(all_leaves, subset_size)
        unknown_leaves = list(set(all_leaves) - set(known_leaves))
        
        logging.info(f"Known leaves: {len(known_leaves)}")
        logging.info(f"Unknown leaves: {len(unknown_leaves)}")
        
        # Create datasets: train on known, test on full and unknown
        train_dataset = prune_dataset_to_leaves(full_test_dataset, known_leaves)
        test_dataset = full_test_dataset  # test_full
        test_unknown_dataset = prune_dataset_to_leaves(full_test_dataset, unknown_leaves)
        
        # Cache leaf splits for evaluation (avoid pickling dataset objects)
        torch.save({
            "known_leaves": known_leaves,
            "unknown_leaves": unknown_leaves,
        }, os.path.join(out_dir, "pll_datasets.pt"))
        
    else:
        train_dataset = full_train_dataset
        test_dataset = full_test_dataset

    input_dim = train_dataset.data_normalized.shape[1]
    logging.info(f"Input dimension: {input_dim}")
    logging.info(f"Train dataset leaves: {train_dataset.n_leaves}")
    logging.info(f"Test dataset leaves: {test_dataset.n_leaves}")

    model = CellTreeQMAttention(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        output_dim=args.output_dim,
        dropout_data=args.dropout_data,
        dropout_metric=args.dropout_metric,
        norm_method=args.norm,
        proj_dim=args.proj_dim,
        gate_mode=args.gate,
        device=str(device),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    node_mtx_dict = train_dataset.get_node_mtx()
    pts_mtx = (
        torch.tensor(node_mtx_dict["node_mtx"], dtype=torch.float).unsqueeze(0).to(device)
    )
    dm_ref = train_dataset.ref_dm.unsqueeze(0).to(device)

    best_rf = float("inf")
    training_seed = args.dm_quartet_seed if args.dm_quartet_seed is not None else args.seed
    for epoch in range(args.epochs):
        model.train()
        # Control the number of steps explicitly for scalability
        max_step = max(1, args.steps_per_epoch)
        for step in range(max_step):
            optimizer.zero_grad()
            trans_pts_mtx = model(pts_mtx)
            dm = pairwise_distances(trans_pts_mtx, metric=args.metric).to(device)
            if args.setting == "high_level_partition":
                # Generate or load known quartets derived from tree level; cache under dataset artifacts
                if step == 0 and epoch == 0:
                    qk_dir = os.path.join(
                        dataset_artifacts_dir, "quartets", f"level-{args.prior_level}"
                    )
                    os.makedirs(qk_dir, exist_ok=True)
                    qk_tensor_path = os.path.join(qk_dir, "quartets_known_tensor.pt")
                    qk_codes_path = os.path.join(qk_dir, "quartets_known_codes_tensor.pt")

                    if os.path.exists(qk_tensor_path) and os.path.exists(qk_codes_path):
                        logging.info(
                            f"Loading cached known quartets from {qk_dir} (level={args.prior_level})"
                        )
                        qk_tensor = torch.load(qk_tensor_path, map_location=device)
                        qk_codes = torch.load(qk_codes_path, map_location=device)
                    else:
                        logging.info(
                            f"Generating known quartets from tree level {args.prior_level} and caching to {qk_dir}"
                        )
                        tree = train_dataset.topology_tree
                        quartet_dict_known = generate_quartets_from_level(
                            tree, args.prior_level, "known"
                        )
                        qk_tensor, qk_codes = quartet_dict_to_tensors(quartet_dict_known)
                        torch.save(qk_tensor, qk_tensor_path)
                        torch.save(qk_codes, qk_codes_path)

                    cli_cached = {"qk_tensor": qk_tensor, "qk_codes": qk_codes}

                if "cli_cached" in locals():
                    dm_quartets, dm_ref_quartets = generate_quartets_tensor_from_tensor_vectorized(
                        batch_size=args.quartets,
                        dm=dm,
                        quartets_tensor=cli_cached["qk_tensor"],
                        codes_tensor=cli_cached["qk_codes"],
                        device=device,
                    )
                else:
                    dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                        batch_size=args.quartets, dm=dm, dm_ref=dm_ref, device=device
                    )
            else:
                dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                    batch_size=args.quartets, dm=dm, dm_ref=dm_ref, device=device, seed=training_seed
                )

            if args.metric_loss == "additivity":
                loss_P, _, _, _ = additivity_error_quartet_tensor(
                    dm_quartets, dm_ref_quartets, args.weight_close, args.weight_push, args.margin, device=device
                )
            elif args.metric_loss == "triplet":
                loss_P = triplet_loss_quartet_tensor_vectorized(
                    dm_quartets, dm_ref_quartets, margin=args.margin, device=device
                )
            else:
                loss_P, *_ = quadruplet_loss_quartet_tensor_vectorized(
                    dm_quartets, dm_ref_quartets, alpha=0.5, beta=0.5, device=device
                )

            loss_D = distance_error(pts_mtx, trans_pts_mtx, diff_norm="fro", dist_metric=args.metric)
            total_loss = args.weight_D * loss_D + args.weight_P * loss_P
            total_loss.backward()
            optimizer.step()

            # Periodic lightweight train evaluation (train RF)
            if step % max(1, args.eval_interval) == 0:
                with torch.no_grad():
                    _, emb_dm_tmp, node_names_tmp = check_embedding(train_dataset, model, args.metric, device)
                    emb_tree_tmp = reconstruct_from_dm(emb_dm_tmp, node_names_tmp, method=args.recon_method)
                    rf_train_tmp = train_dataset.compare_trees(emb_tree_tmp, ref_tree="topology_tree")["relative_rf"]
                    logging.info(
                        f"[Epoch {epoch+1}/{args.epochs} | Step {step}/{max_step}] train RF={rf_train_tmp:.4f} loss={total_loss.item():.4f}"
                    )

        logging.info(f"Epoch {epoch+1}/{args.epochs} - loss={total_loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
    logging.info(f"Saved model to {out_dir}")

    # Evaluate on train/test and save results.json
    import json
    model.eval()
    with torch.no_grad():
        # Train metrics
        _, emb_dm_train, node_names_train = check_embedding(train_dataset, model, args.metric, device)
        emb_tree_train = reconstruct_from_dm(emb_dm_train, node_names_train, method=args.recon_method)
        rf_train = train_dataset.compare_trees(emb_tree_train, ref_tree="topology_tree")["relative_rf"]

        # Test metrics
        _, emb_dm_test, node_names_test = check_embedding(test_dataset, model, args.metric, device)
        emb_tree_test = reconstruct_from_dm(emb_dm_test, node_names_test, method=args.recon_method)
        rf_test = test_dataset.compare_trees(emb_tree_test, ref_tree="topology_tree")["relative_rf"]

        # Quartet distance evaluation
        pts_mtx_test = (
            torch.tensor(test_dataset.get_node_mtx()["node_mtx"], dtype=torch.float).unsqueeze(0).to(device)
        )
        dm_test = pairwise_distances(model(pts_mtx_test), metric=args.metric).to(device)
        dm_test = dm_test.squeeze(0)  # (N,N)
        dm_ref_test = test_dataset.ref_dm
        if not isinstance(dm_ref_test, torch.Tensor):
            dm_ref_test = torch.tensor(dm_ref_test, dtype=torch.float32)
        dm_ref_test = dm_ref_test.to(device)
        
        metrics = {
            "rf_train": rf_train,
            "rf_test": rf_test,
        }
        
        if args.setting == "high_level_partition":
            # Load cached known quartets from dataset artifacts and evaluate known/unknown separately
            qk_dir = os.path.join(
                dataset_artifacts_dir, "quartets", f"level-{args.prior_level}"
            )
            qk_tensor_path = os.path.join(qk_dir, "quartets_known_tensor.pt")
            qk_codes_path = os.path.join(qk_dir, "quartets_known_codes_tensor.pt")
            if os.path.exists(qk_tensor_path) and os.path.exists(qk_codes_path):
                qk_tensor = torch.load(qk_tensor_path, map_location=device)
                qk_codes = torch.load(qk_codes_path, map_location=device)
                
                # Generate all quartets and separate known/unknown
                logging.info("Generating all quartets for evaluation...")
                quartet_dict_all = generate_all_quartets(test_dataset.topology_tree)
                known_quartets_set = set()
                for i in range(qk_tensor.size(0)):
                    quartet_key = tuple(sorted(qk_tensor[i].tolist()))
                    known_quartets_set.add(quartet_key)
                
                quartet_dict_unknown = {q: code for q, code in quartet_dict_all.items() if q not in known_quartets_set}
                
                # Convert to tensors
                qu_tensor, qu_codes = quartet_dict_to_tensors(quartet_dict_unknown)
                qa_tensor, qa_codes = quartet_dict_to_tensors(quartet_dict_all)
                
                # Cap evaluation quartets; -1 means all
                eval_bs = args.eval_quartets_cap if args.eval_quartets_cap > 0 else -1
                if eval_bs > 0 and args.dm_quartet_seed is not None:
                    torch.manual_seed(args.dm_quartet_seed)

                # Evaluate known quartets
                sel_k = qk_tensor if eval_bs == -1 or eval_bs >= qk_tensor.size(0) else qk_tensor[torch.randperm(qk_tensor.size(0), device=device)[:eval_bs]]
                idx_rows = sel_k.unsqueeze(2).expand(-1, -1, 4)
                idx_cols = sel_k.unsqueeze(1).expand(-1, 4, -1)
                dm_quartets_known = dm_test[idx_rows, idx_cols]
                dm_ref_quartets_known = dm_ref_test[idx_rows, idx_cols]
                quartet_dist_known = get_quartet_dist(dm_quartets_known, dm_ref_quartets_known).item()
                
                # Evaluate unknown quartets
                sel_u = qu_tensor if eval_bs == -1 or eval_bs >= qu_tensor.size(0) else qu_tensor[torch.randperm(qu_tensor.size(0), device=device)[:eval_bs]]
                idx_rows = sel_u.unsqueeze(2).expand(-1, -1, 4)
                idx_cols = sel_u.unsqueeze(1).expand(-1, 4, -1)
                dm_quartets_unknown = dm_test[idx_rows, idx_cols]
                dm_ref_quartets_unknown = dm_ref_test[idx_rows, idx_cols]
                quartet_dist_unknown = get_quartet_dist(dm_quartets_unknown, dm_ref_quartets_unknown).item()
                
                # Evaluate all quartets
                sel_a = qa_tensor if eval_bs == -1 or eval_bs >= qa_tensor.size(0) else qa_tensor[torch.randperm(qa_tensor.size(0), device=device)[:eval_bs]]
                idx_rows = sel_a.unsqueeze(2).expand(-1, -1, 4)
                idx_cols = sel_a.unsqueeze(1).expand(-1, 4, -1)
                dm_quartets_all = dm_test[idx_rows, idx_cols]
                dm_ref_quartets_all = dm_ref_test[idx_rows, idx_cols]
                quartet_dist_all = get_quartet_dist(dm_quartets_all, dm_ref_quartets_all).item()
                
                metrics.update({
                    "quartet_dist_all": quartet_dist_all,
                    "quartet_dist_known": quartet_dist_known,
                    "quartet_dist_unknown": quartet_dist_unknown,
                })
                
                logging.info(f"Quartet distances - All: {quartet_dist_all:.4f}, Known: {quartet_dist_known:.4f}, Unknown: {quartet_dist_unknown:.4f}")
            else:
                logging.warning("Known quartets not found, using standard evaluation")
                dm_quartets, dm_ref_quartets = generate_quartets_tensor(batch_size=min(100000, 4096), dm=dm_test, dm_ref=dm_ref_test, device=device)
                quartet_dist_test = get_quartet_dist(dm_quartets, dm_ref_quartets).item()
                metrics["quartet_dist_test"] = quartet_dist_test
                
        elif args.setting == "partially_labeled_leaves":
            # Load cached datasets and evaluate known/partial/unknown quartets
            if os.path.exists(os.path.join(out_dir, "pll_datasets.pt")):
                pll_data = torch.load(os.path.join(out_dir, "pll_datasets.pt"), map_location="cpu", weights_only=False)
                known_leaves = pll_data["known_leaves"]
                unknown_leaves = pll_data["unknown_leaves"]
                # Reconstruct unknown dataset on the fly to avoid unpickling complex objects
                test_unknown_dataset = prune_dataset_to_leaves(test_dataset, unknown_leaves)
                
                # Generate all quartets from full test dataset
                logging.info("Generating all quartets for partially_labeled_leaves evaluation...")
                quartet_dict_all = generate_all_quartets(test_dataset.topology_tree)
                
                # Classify quartets as known/partial/unknown based on leaf membership
                leaf_name_to_index = {name: i for i, name in enumerate(test_dataset.leave_names)}
                known_set = set(known_leaves)
                
                quartet_dict_known = {}
                quartet_dict_partial = {}
                quartet_dict_unknown = {}
                
                for quartet_tuple, code in quartet_dict_all.items():
                    leaf_names_in_q = [test_dataset.leave_names[idx] for idx in quartet_tuple]
                    known_count = sum(1 for name in leaf_names_in_q if name in known_set)
                    
                    if known_count == 4:
                        quartet_dict_known[quartet_tuple] = code
                    elif known_count == 0:
                        quartet_dict_unknown[quartet_tuple] = code
                    else:
                        quartet_dict_partial[quartet_tuple] = code
                
                logging.info(f"Quartet classification - Known: {len(quartet_dict_known)}, Partial: {len(quartet_dict_partial)}, Unknown: {len(quartet_dict_unknown)}")
                
                # Convert to tensors and evaluate each category
                qa_tensor, qa_codes = quartet_dict_to_tensors(quartet_dict_all)
                qk_tensor, qk_codes = quartet_dict_to_tensors(quartet_dict_known)
                qp_tensor, qp_codes = quartet_dict_to_tensors(quartet_dict_partial)
                qu_tensor, qu_codes = quartet_dict_to_tensors(quartet_dict_unknown)
                
                eval_bs = args.eval_quartets_cap if args.eval_quartets_cap > 0 else -1
                if eval_bs > 0 and args.dm_quartet_seed is not None:
                    torch.manual_seed(args.dm_quartet_seed)

                # Evaluate all
                sel_a = qa_tensor if eval_bs == -1 or eval_bs >= qa_tensor.size(0) else qa_tensor[torch.randperm(qa_tensor.size(0), device=device)[:eval_bs]]
                idx_rows = sel_a.unsqueeze(2).expand(-1, -1, 4)
                idx_cols = sel_a.unsqueeze(1).expand(-1, 4, -1)
                dm_quartets_all = dm_test[idx_rows, idx_cols]
                dm_ref_quartets_all = dm_ref_test[idx_rows, idx_cols]
                quartet_dist_all = get_quartet_dist(dm_quartets_all, dm_ref_quartets_all).item()
                
                # Evaluate known
                sel_k = qk_tensor if eval_bs == -1 or eval_bs >= qk_tensor.size(0) else qk_tensor[torch.randperm(qk_tensor.size(0), device=device)[:eval_bs]]
                idx_rows = sel_k.unsqueeze(2).expand(-1, -1, 4)
                idx_cols = sel_k.unsqueeze(1).expand(-1, 4, -1)
                dm_quartets_known = dm_test[idx_rows, idx_cols]
                dm_ref_quartets_known = dm_ref_test[idx_rows, idx_cols]
                quartet_dist_known = get_quartet_dist(dm_quartets_known, dm_ref_quartets_known).item()
                
                # Evaluate partial
                sel_p = qp_tensor if eval_bs == -1 or eval_bs >= qp_tensor.size(0) else qp_tensor[torch.randperm(qp_tensor.size(0), device=device)[:eval_bs]]
                idx_rows = sel_p.unsqueeze(2).expand(-1, -1, 4)
                idx_cols = sel_p.unsqueeze(1).expand(-1, 4, -1)
                dm_quartets_partial = dm_test[idx_rows, idx_cols]
                dm_ref_quartets_partial = dm_ref_test[idx_rows, idx_cols]
                quartet_dist_partial = get_quartet_dist(dm_quartets_partial, dm_ref_quartets_partial).item()
                
                # Evaluate unknown
                sel_u = qu_tensor if eval_bs == -1 or eval_bs >= qu_tensor.size(0) else qu_tensor[torch.randperm(qu_tensor.size(0), device=device)[:eval_bs]]
                idx_rows = sel_u.unsqueeze(2).expand(-1, -1, 4)
                idx_cols = sel_u.unsqueeze(1).expand(-1, 4, -1)
                dm_quartets_unknown = dm_test[idx_rows, idx_cols]
                dm_ref_quartets_unknown = dm_ref_test[idx_rows, idx_cols]
                quartet_dist_unknown = get_quartet_dist(dm_quartets_unknown, dm_ref_quartets_unknown).item()
                
                # Evaluate on test_unknown dataset (RF)
                _, emb_dm_unknown, node_names_unknown = check_embedding(test_unknown_dataset, model, args.metric, device)
                emb_tree_unknown = reconstruct_from_dm(emb_dm_unknown, node_names_unknown, method=args.recon_method)
                rf_test_unknown = test_unknown_dataset.compare_trees(emb_tree_unknown, ref_tree="topology_tree")["relative_rf"]
                
                metrics.update({
                    "quartet_dist_all": quartet_dist_all,
                    "quartet_dist_known": quartet_dist_known,
                    "quartet_dist_partial": quartet_dist_partial,
                    "quartet_dist_unknown": quartet_dist_unknown,
                    "rf_test_unknown": rf_test_unknown,
                })
                
                logging.info(f"Quartet distances - All: {quartet_dist_all:.4f}, Known: {quartet_dist_known:.4f}, Partial: {quartet_dist_partial:.4f}, Unknown: {quartet_dist_unknown:.4f}")
                logging.info(f"RF test_unknown: {rf_test_unknown:.4f}")
            else:
                logging.warning("PLL datasets not found, using standard evaluation")
                dm_quartets, dm_ref_quartets = generate_quartets_tensor(batch_size=min(100000, 4096), dm=dm_test, dm_ref=dm_ref_test, device=device)
                quartet_dist_test = get_quartet_dist(dm_quartets, dm_ref_quartets).item()
                metrics["quartet_dist_test"] = quartet_dist_test
                
        else:
            # Standard evaluation for fully_supervised
            eval_bs = args.eval_quartets_cap if args.eval_quartets_cap > 0 else -1
            dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                batch_size=eval_bs if eval_bs != -1 else 100000,
                dm=dm_test,
                dm_ref=dm_ref_test,
                device=device,
                seed=args.dm_quartet_seed if args.dm_quartet_seed is not None else args.seed,
            )
            quartet_dist_test = get_quartet_dist(dm_quartets, dm_ref_quartets).item()
            metrics["quartet_dist_test"] = quartet_dist_test

        results = {
            "run_name": run_name,
            "dataset": args.dataset,
            "lineage": args.lineage,
            "setting": args.setting,
            "uid": uid,
            "metrics": metrics,
        }
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved results.json with metrics: {results['metrics']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="celltreeqm", description="CellTreeQM computational tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train command (Celegans split via CellTreeBench)
    t = sub.add_parser("train", help="Train CellTreeQM on CellTreeBench celegans split")
    t.add_argument("--bench-root", default="/workspaces/CellTreeQM/CellTreeBench", help="CellTreeBench root directory")
    t.add_argument("--dataset", default="celegans_small")
    t.add_argument("--lineage", default="P0")
    t.add_argument(
        "--setting",
        default="fully_supervised",
        choices=[
            "fully_supervised",
            "high_level_partition",
            "partially_labeled_leaves",
            "unsupervised",
        ],
        help="Training setting label (matches research codebase)",
    )
    t.add_argument(
        "--prior-level",
        type=int,
        default=2,
        help="Tree level for high_level_partition setting (higher = more known quartets)",
    )
    t.add_argument(
        "--known-fraction",
        type=float,
        default=0.5,
        help="Fraction of leaves to use as 'known' for partially_labeled_leaves setting",
    )
    t.add_argument("--uid", default=None, help="Unique id for this run; default is timestamp YYYYMMDD-HHMMSS")
    t.add_argument("--run-name", default=None, help="Override full run name; default is <dataset>_<lineage>_<setting>_<uid>")
    t.add_argument(
        "--output-dir",
        default=None,
        help="Root directory for experiment outputs (models/results). Defaults to <bench-root>/examples/out",
    )
    t.add_argument(
        "--dataset-artifacts-dir",
        default=None,
        help="Directory for dataset-level reusable artifacts (trees, gene lists, known quartets). Defaults to <bench-root>/data/<dataset>/<lineage>/artifacts",
    )
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    t.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan", "cosine", "inf"]) 
    t.add_argument("--metric-loss", default="additivity", choices=["additivity", "triplet", "quadruplet"]) 
    t.add_argument("--quartets", type=int, default=2048, help="quartets per step")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--steps-per-epoch", type=int, default=300, help="Number of training steps per epoch")
    t.add_argument("--eval-interval", type=int, default=50, help="Evaluate train RF every N steps")
    t.add_argument("--eval-quartets-cap", type=int, default=100000, help="Max quartets for evaluation; <=0 means all")
    t.add_argument("--dm-quartet-seed", type=int, default=None, help="Random seed for quartet sampling during training/eval; defaults to --seed")
    t.add_argument("--recon-method", default="nj", choices=["nj", "upgma", "ward", "single"], help="Tree reconstruction method for evaluation")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--weight-D", type=float, default=0.1)
    t.add_argument("--weight-P", type=float, default=20.0)
    t.add_argument("--weight-close", type=float, default=1.0)
    t.add_argument("--weight-push", type=float, default=30.0)
    t.add_argument("--margin", type=float, default=0.1)
    t.add_argument("--hidden-dim", type=int, default=1024)
    t.add_argument("--num-heads", type=int, default=2)
    t.add_argument("--num-layers", type=int, default=8)
    t.add_argument("--output-dim", type=int, default=128)
    t.add_argument("--proj-dim", type=int, default=1024)
    t.add_argument("--dropout-data", type=float, default=0.1)
    t.add_argument("--dropout-metric", type=float, default=0.1)
    t.add_argument("--norm", default="batch_norm", choices=["batch_norm", "layer_norm", "none"])
    t.add_argument("--gate", default="none", choices=["none", "linear", "sigmoid", "softmax", "gumbel"])
    t.add_argument("--verbose", action="store_true")
    t.set_defaults(func=cmd_train)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


