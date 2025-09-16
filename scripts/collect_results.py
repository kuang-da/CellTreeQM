#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
import csv


def parse_args():
    p = argparse.ArgumentParser(description="Collect results.json under an output root into a CSV")
    p.add_argument("--output-root", required=True, help="Root directory containing run subdirectories")
    p.add_argument("--out-csv", required=True, help="Destination CSV path")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.output_root)
    rows = []
    for run_dir in sorted(root.glob("*/")):
        res_path = run_dir / "results.json"
        if not res_path.exists():
            continue
        try:
            with open(res_path, "r") as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            row = {
                "run_name": data.get("run_name", run_dir.name),
                "dataset": data.get("dataset", ""),
                "lineage": data.get("lineage", ""),
                "setting": data.get("setting", ""),
                "rf_train": metrics.get("rf_train", ""),
                "rf_test": metrics.get("rf_test", ""),
                "quartet_dist_test": metrics.get("quartet_dist_test", ""),
                "quartet_dist_all": metrics.get("quartet_dist_all", ""),
                "quartet_dist_known": metrics.get("quartet_dist_known", ""),
                "quartet_dist_partial": metrics.get("quartet_dist_partial", ""),
                "quartet_dist_unknown": metrics.get("quartet_dist_unknown", ""),
            }
            rows.append(row)
        except Exception:
            continue

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "dataset",
        "lineage",
        "setting",
        "rf_train",
        "rf_test",
        "quartet_dist_test",
        "quartet_dist_all",
        "quartet_dist_known",
        "quartet_dist_partial",
        "quartet_dist_unknown",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()


