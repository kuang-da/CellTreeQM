# Benchmark scripts (one per todo.md task)

- Purpose: Each script runs a reproducible command matching one task in `../../todo.md`.
- Output directory naming follows: `<bench-root>/examples/out/<dataset>_<lineage>_<setting>_<uid>/`.
- Convention:
  - Script name: `<dataset>_<lineage>_<setting>.sh`
  - Required tools installed: `celltreeqm` CLI and `CellTreeBench`
  - You can override parameters via env vars (see each script header)

Example:
```bash
./celegans_small_supervised.sh
```

Environment overrides (common):
- BENCH_ROOT (default: `/workspaces/CellTreeQM/CellTreeBench`)
- DATASET (default: `celegans_small`)
- LINEAGE (default: `P0`)
- SETTING (default: `supervised`)
- EPOCHS (default: `3`)
- QUARTETS (default: `2048`)
- UID_OVERRIDE (default: current timestamp)
- EXTRA (any extra CLI args)
