#!/usr/bin/env python3
"""Aggregate ARCGym reward-ablation result folders."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


GROUP_KEYS = ["colon_id", "task_id", "algorithm", "constrained", "reward_variant", "seed"]
METRIC_KEYS = [
    "normalized_mean_step_reward",
    "raw_mean_step_reward",
    "success_rate",
    "normalized_progress",
    "roi_alignment_rate",
    "lumen_visible_ratio",
]


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _last_training_row(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "constrained"}
    return bool(value)


def collect_runs(root: Path) -> list[dict]:
    runs = []
    for eval_path in sorted(root.glob("**/evaluation_metrics.json")):
        run_dir = eval_path.parent
        eval_metrics = _load_json(eval_path)
        config_path = run_dir / "config.json"
        config = _load_json(config_path) if config_path.exists() else {}
        run_config = config.get("run_config", {})
        training_row = _last_training_row(run_dir / "training_log.csv")

        row = {
            "run_dir": str(run_dir),
            "colon_id": eval_metrics.get("colon_id", run_config.get("colon_id", "unknown")),
            "task_id": eval_metrics.get("task_id", run_config.get("task_id", "unknown")),
            "algorithm": eval_metrics.get("algorithm", run_config.get("algorithm", "unknown")),
            "constrained": _coerce_bool(eval_metrics.get("constrained", run_config.get("constrained", False))),
            "reward_variant": eval_metrics.get("reward_variant", run_config.get("reward_variant", "unknown")),
            "seed": str(eval_metrics.get("seed", run_config.get("seed", "unknown"))),
        }

        for metric in METRIC_KEYS:
            value = _to_float(eval_metrics.get(metric))
            if value is None:
                value = _to_float(training_row.get(metric))
            row[metric] = value
        runs.append(row)
    return runs


def aggregate(runs: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for run in runs:
        key = tuple(run[k] for k in GROUP_KEYS)
        grouped[key].append(run)

    summary = []
    for key, rows in sorted(grouped.items()):
        out = dict(zip(GROUP_KEYS, key))
        out["num_runs"] = len(rows)
        for metric in METRIC_KEYS:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            out[f"{metric}_mean"] = mean(values) if values else None
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else None
        summary.append(out)
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_heatmap(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return

    if not rows:
        return
    df = pd.DataFrame(rows)
    metric = "normalized_mean_step_reward_mean"
    if metric not in df or df[metric].isna().all():
        return
    df["setting"] = df["algorithm"].astype(str) + "_" + df["constrained"].astype(str)
    pivot = df.pivot_table(index="reward_variant", columns="setting", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot.index) * 0.5)))
    image = ax.imshow(pivot.values, vmin=-1, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Normalized Mean Step Reward")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="reward_ablation_results")
    parser.add_argument("--output_dir", default="reward_ablation_results")
    parser.add_argument("--heatmap", action="store_true", help="Also write reward_ablation_heatmap.png when plotting deps exist")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(results_dir)
    summary = aggregate(runs)

    csv_path = output_dir / "reward_ablation_summary.csv"
    json_path = output_dir / "reward_ablation_summary.json"
    write_csv(summary, csv_path)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.heatmap:
        maybe_write_heatmap(summary, output_dir / "reward_ablation_heatmap.png")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
