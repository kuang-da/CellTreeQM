import argparse
import logging
import os
import time
import torch
import torch.optim as optim

from . import CellTreeQMAttention
from .trainer import TrainConfig, TrainInputs, train as trainer_train
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

    # Prepare optional HLP known quartets (cache/load under dataset artifacts)
    hlp_quartets = None
    if args.setting == "high_level_partition":
        qk_dir = os.path.join(dataset_artifacts_dir, "quartets", f"level-{args.prior_level}")
        os.makedirs(qk_dir, exist_ok=True)
        qk_tensor_path = os.path.join(qk_dir, "quartets_known_tensor.pt")
        qk_codes_path = os.path.join(qk_dir, "quartets_known_codes_tensor.pt")
        if os.path.exists(qk_tensor_path) and os.path.exists(qk_codes_path):
            logging.info(f"Loading cached known quartets from {qk_dir} (level={args.prior_level})")
            qk_tensor = torch.load(qk_tensor_path, map_location=device)
            qk_codes = torch.load(qk_codes_path, map_location=device)
        else:
            # Sampled or full known quartets depending on cap
            from .utils.quartet_utils import generate_quartets_from_level_sampled
            tree = train_dataset.topology_tree
            if args.known_quartets_cap and args.known_quartets_cap > 0:
                logging.info(
                    f"Generating sampled known quartets (cap={args.known_quartets_cap}) from level {args.prior_level} and caching to {qk_dir}"
                )
                quartet_dict_known = generate_quartets_from_level_sampled(tree, args.prior_level, args.known_quartets_cap, seed=args.seed)
            else:
                logging.info(f"Generating all known quartets from tree level {args.prior_level} and caching to {qk_dir}")
                quartet_dict_known = generate_quartets_from_level(tree, args.prior_level, "known")
            qk_tensor, qk_codes = quartet_dict_to_tensors(quartet_dict_known)
            torch.save(qk_tensor, qk_tensor_path)
            torch.save(qk_codes, qk_codes_path)
        hlp_quartets = (qk_tensor, qk_codes)

    # Build trainer inputs/config and delegate
    inputs = TrainInputs(
        setting=args.setting,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        hlp_quartets=hlp_quartets,
        pll_known_leaves=(known_leaves if args.setting == "partially_labeled_leaves" else None),
        pll_unknown_leaves=(unknown_leaves if args.setting == "partially_labeled_leaves" else None),
    )

    cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        quartets_per_step=args.quartets,
        metric=args.metric,
        metric_loss=args.metric_loss,
        weight_D=args.weight_D,
        weight_P=args.weight_P,
        weight_close=args.weight_close,
        weight_push=args.weight_push,
        margin=args.margin,
        eval_interval=args.eval_interval,
        eval_quartets_cap=args.eval_quartets_cap,
        dm_quartet_seed=args.dm_quartet_seed,
        seed=args.seed,
        recon_method=args.recon_method,
        device=str(device),
    )

    result = trainer_train(model, inputs, cfg, out_dir)
    logging.info(f"Saved model to {out_dir}")

    # Append run metadata to results.json
    import json
    with open(os.path.join(out_dir, "results.json"), "r+") as f:
        data = json.load(f)
        data.update({
            "run_name": run_name,
            "dataset": args.dataset,
            "lineage": args.lineage,
            "setting": args.setting,
            "uid": uid,
        })
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

    # Persist full CLI args for reproducibility
    try:
        args_dict = dict(vars(args))
        # Ensure JSON-serializable: stringify or reduce problematic fields
        func_obj = args_dict.pop("func", None)
        if func_obj is not None:
            args_dict["func"] = getattr(func_obj, "__name__", str(func_obj))
        with open(os.path.join(out_dir, "args_or_config.json"), "w") as fa:
            json.dump(args_dict, fa, indent=2, default=str)
    except Exception as e:
        logging.warning(f"Failed to write args_or_config.json: {e}")


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
        "--known-quartets-cap",
        type=int,
        default=500000,
        help="Cap for known quartets generation in HLP; <=0 means full enumeration",
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


