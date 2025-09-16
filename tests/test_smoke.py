import json
import os
import sys
import subprocess

import pytest


@pytest.mark.library
def test_trainer_library_smoke(tmp_path):
    # Import lazily to ensure editable install
    from celltreeqm import CellTreeQMAttention, TrainConfig, TrainInputs, train
    from celltreebench.datasets.celegans import load_celegans_supervised_split

    bench_root = "/workspaces/CellTreeQM/CellTreeBench"
    data_dir = os.path.join(bench_root, "data")
    out_dir = tmp_path

    tr, te = load_celegans_supervised_split(
        dataset_name="celegans_small", lineage_name="P0", data_dir=data_dir, out_dir=str(out_dir), seed=123
    )
    model = CellTreeQMAttention(
        input_dim=tr.data_normalized.shape[1], hidden_dim=128, num_heads=2, num_layers=2, output_dim=32,
        dropout_data=0.1, dropout_metric=0.1, norm_method="batch_norm", proj_dim=128, gate_mode="none", device="cpu"
    )
    cfg = TrainConfig(epochs=1, steps_per_epoch=2, quartets_per_step=64, device="cpu", eval_interval=1, eval_quartets_cap=64)
    inputs = TrainInputs(setting="fully_supervised", train_dataset=tr, test_dataset=te)

    res = train(model, inputs, cfg, str(out_dir))
    assert (out_dir / "best_model.pth").exists()
    with open(out_dir / "results.json") as f:
        data = json.load(f)
    assert "metrics" in data and "rf_train" in data["metrics"]


@pytest.mark.e2e
def test_cli_smoke(tmp_path):
    bench_root = "/workspaces/CellTreeQM/CellTreeBench"
    env = os.environ.copy()
    env["CELLTREEQM_OUTPUT_DIR"] = str(tmp_path)
    cmd = [
        sys.executable, "-m", "celltreeqm.cli", "train",
        "--bench-root", bench_root,
        "--dataset", "celegans_small",
        "--lineage", "P0",
        "--setting", "fully_supervised",
        "--epochs", "1",
        "--quartets", "64",
        "--steps-per-epoch", "2",
        "--eval-interval", "1",
        "--eval-quartets-cap", "64",
        "--device", "cpu",
        "--seed", "123",
        "--recon-method", "nj",
    ]
    subprocess.run(cmd, check=True, env=env)
    # Expect exactly one run directory under tmp_path
    entries = list(tmp_path.iterdir())
    assert any((p / "best_model.pth").exists() for p in entries)

