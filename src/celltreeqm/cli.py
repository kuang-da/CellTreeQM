import argparse
import logging
import os
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
    out_dir = os.path.join(
        args.bench_root,
        "examples/out",
        f"minimal_example_{args.dataset}_{args.lineage}",
    )
    os.makedirs(out_dir, exist_ok=True)

    train_dataset, test_dataset = load_celegans_supervised_split(
        dataset_name=args.dataset,
        lineage_name=args.lineage,
        data_dir=data_dir,
        out_dir=out_dir,
        sampling_method="biological",
        seed=args.seed,
    )

    input_dim = train_dataset.data_normalized.shape[1]
    logging.info(f"Input dimension: {input_dim}")

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
    for epoch in range(args.epochs):
        model.train()
        max_step = max(1, len(train_dataset) // args.quartets)
        for step in range(max_step):
            optimizer.zero_grad()
            trans_pts_mtx = model(pts_mtx)
            dm = pairwise_distances(trans_pts_mtx, metric=args.metric).to(device)
            dm_quartets, dm_ref_quartets = generate_quartets_tensor(
                batch_size=args.quartets, dm=dm, dm_ref=dm_ref, device=device
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

        logging.info(f"Epoch {epoch+1}/{args.epochs} - loss={total_loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
    logging.info(f"Saved model to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="celltreeqm", description="CellTreeQM computational tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Train command (Celegans split via CellTreeBench)
    t = sub.add_parser("train", help="Train CellTreeQM on CellTreeBench celegans split")
    t.add_argument("--bench-root", default="/workspaces/CellTreeQM/CellTreeBench", help="CellTreeBench root directory")
    t.add_argument("--dataset", default="celegans_small")
    t.add_argument("--lineage", default="P0")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    t.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan", "cosine", "inf"]) 
    t.add_argument("--metric-loss", default="additivity", choices=["additivity", "triplet", "quadruplet"]) 
    t.add_argument("--quartets", type=int, default=2048, help="quartets per step")
    t.add_argument("--epochs", type=int, default=5)
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


