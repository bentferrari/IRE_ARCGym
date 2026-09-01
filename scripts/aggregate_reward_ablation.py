#!/usr/bin/env python3
"""Aggregate reward-ablation evaluations and audit experiment completeness.

Only ``evaluation_metrics.json`` is used for the final task-success table.  In
particular, this script never substitutes a reward or training metric for the
independent, geometric ``goal_reached`` evaluation outcome.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev


RUN_KEYS = ["colon_id", "task_id", "algorithm", "constrained", "reward_variant", "seed"]
BY_COLON_KEYS = ["colon_id", "task_id", "algorithm", "constrained", "reward_variant"]
PAPER_KEYS = ["task_id", "algorithm", "constrained", "reward_variant"]
METRIC_KEYS = [
    # Independent task outcomes (not terms in the ablated reward).
    "success_rate",
    "normalized_progress",
    "mean_episode_length",
    # Secondary diagnostics; these must not replace task success.
    "roi_alignment_rate",
    "lumen_visible_ratio",
    "normalized_mean_step_reward",
    "raw_mean_step_reward",
    "mean_s_c",
    "mean_s_1",
    "mean_s_2",
    "mean_s_3",
    "mean_s_o",
]
EXPECTED_VARIANTS = [
    ("center_only", False),
    ("depth_existence", False),
    ("deep_area", False),
    ("lumen_evidence", False),
    ("full_reward", False),
    ("full_reward", True),
]


def _load_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "constrained"}
    return bool(value)


def _seed_sort_key(value):
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(value, path: Path) -> None:
    with path.open("w") as file:
        json.dump(value, file, indent=2)


def collect_runs(root: Path) -> list[dict]:
    """Collect deterministic post-training evaluations, one row per run."""
    runs = []
    for eval_path in sorted(root.glob("**/evaluation_metrics.json")):
        run_dir = eval_path.parent
        evaluation = _load_json(eval_path)
        config_path = run_dir / "config.json"
        config = _load_json(config_path) if config_path.exists() else {}
        run_config = config.get("run_config", {})

        episode_successes = evaluation.get("episode_successes") or []
        successes = sum(bool(value) for value in episode_successes)
        episodes = evaluation.get("episodes_completed", evaluation.get("test_episodes"))
        try:
            episodes = int(episodes)
        except (TypeError, ValueError):
            episodes = len(episode_successes)
        if not episode_successes:
            reported_rate = _to_float(evaluation.get("success_rate"))
            successes = round(reported_rate * episodes) if reported_rate is not None and episodes else 0

        row = {
            "run_dir": str(run_dir),
            "timestamp": str(run_config.get("timestamp", "")),
            "colon_id": evaluation.get("colon_id", run_config.get("colon_id", "unknown")),
            "task_id": evaluation.get("task_id", run_config.get("task_id", "unknown")),
            "algorithm": evaluation.get("algorithm", run_config.get("algorithm", "unknown")),
            "constrained": _coerce_bool(
                evaluation.get("constrained", run_config.get("constrained", False))
            ),
            "reward_variant": evaluation.get(
                "reward_variant", run_config.get("reward_variant", "unknown")
            ),
            "seed": str(evaluation.get("seed", run_config.get("seed", "unknown"))),
            "evaluation_episodes": episodes,
            "evaluation_successes": int(successes),
            "evaluation_source": "deterministic_post_training",
            "training_log": str(run_dir / "training_log.csv"),
        }
        for metric in METRIC_KEYS:
            row[metric] = _to_float(evaluation.get(metric))
        if episodes:
            # Prefer the recorded binary task outcomes to a rounded summary.
            row["success_rate"] = successes / episodes
        runs.append(row)
    return runs


def collect_training_runs(root: Path) -> list[dict]:
    """Collect geometric success recorded during training for curve diagnostics."""
    runs = []
    for log_path in sorted(root.glob("**/training_log.csv")):
        run_dir = log_path.parent
        config_path = run_dir / "config.json"
        config = _load_json(config_path) if config_path.exists() else {}
        run_config = config.get("run_config", {})
        with log_path.open(newline="") as file:
            log_rows = list(csv.DictReader(file))
        final_row = log_rows[-1] if log_rows else {}

        success_path = run_dir / "trajectory_data" / "success_rate_data.json"
        success_summary = _load_json(success_path).get("summary", {}) if success_path.exists() else {}
        status_path = run_dir / "run_status.json"
        run_status = _load_json(status_path) if status_path.exists() else {}
        episodes = success_summary.get("total_episodes", final_row.get("episodes", 0))
        successes = success_summary.get("total_successes")
        try:
            episodes = int(episodes)
        except (TypeError, ValueError):
            episodes = 0
        if successes is None:
            rate = _to_float(final_row.get("success_rate"))
            successes = round(rate * episodes) if rate is not None and episodes else 0
        try:
            successes = int(successes)
        except (TypeError, ValueError):
            successes = 0

        row = {
            "run_dir": str(run_dir),
            "timestamp": str(run_config.get("timestamp", "")),
            "colon_id": run_config.get("colon_id", "unknown"),
            "task_id": run_config.get("task_id", "unknown"),
            "algorithm": run_config.get("algorithm", "unknown"),
            "constrained": _coerce_bool(run_config.get("constrained", False)),
            "reward_variant": run_config.get("reward_variant", "unknown"),
            "seed": str(run_config.get("seed", "unknown")),
            "training_episodes": episodes,
            "training_successes": successes,
            "training_success_rate": successes / episodes if episodes else None,
            "final_timestep": int(float(final_row.get("timestep", 0) or 0)),
            "run_status": run_status.get("status", "missing"),
            "model_path": run_status.get("model_path"),
            "training_log": str(log_path),
        }
        runs.append(row)
    return runs


def deduplicate_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the newest evaluation for an exact setting/seed combination."""
    grouped = defaultdict(list)
    for run in runs:
        grouped[tuple(run[key] for key in RUN_KEYS)].append(run)

    selected, superseded = [], []
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["timestamp"], row["run_dir"]))
        selected.append(rows[-1])
        superseded.extend(rows[:-1])
    selected.sort(key=lambda row: tuple(str(row[key]) for key in RUN_KEYS))
    return selected, superseded


def aggregate_training_runs(runs: list[dict], group_keys: list[str]) -> list[dict]:
    """Summarize training success without presenting it as held-out evaluation."""
    grouped = defaultdict(list)
    for run in runs:
        grouped[tuple(run[key] for key in group_keys)].append(run)
    summaries = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        output = dict(zip(group_keys, key))
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)
        seed_rates = []
        for seed_rows in by_seed.values():
            successes = sum(row["training_successes"] for row in seed_rows)
            episodes = sum(row["training_episodes"] for row in seed_rows)
            if episodes:
                seed_rates.append(successes / episodes)
        output.update(
            {
                "num_runs": len(rows),
                "num_seeds": len(by_seed),
                "seed_ids": ",".join(sorted(by_seed, key=_seed_sort_key)),
                "training_successes_total": sum(row["training_successes"] for row in rows),
                "training_episodes_total": sum(row["training_episodes"] for row in rows),
                "training_success_rate_mean_across_seeds": mean(seed_rates) if seed_rates else None,
                "training_success_rate_std_across_seeds": stdev(seed_rates) if len(seed_rates) >= 2 else None,
                "three_seed_complete": len(by_seed) >= 3,
                "all_runs_completed": all(row["run_status"] == "completed" for row in rows),
            }
        )
        summaries.append(output)
    return summaries


def _wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return center - half_width, center + half_width


def aggregate(runs: list[dict], group_keys: list[str]) -> list[dict]:
    """Aggregate at the seed level first, then report sample SD across seeds."""
    grouped = defaultdict(list)
    for run in runs:
        grouped[tuple(run[key] for key in group_keys)].append(run)

    summaries = []
    for key, rows in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        output = dict(zip(group_keys, key))
        by_seed = defaultdict(list)
        for row in rows:
            by_seed[row["seed"]].append(row)

        seed_ids = sorted(by_seed, key=_seed_sort_key)
        output.update(
            {
                "num_runs": len(rows),
                "num_seeds": len(seed_ids),
                "seed_ids": ",".join(seed_ids),
                "runs_per_seed_min": min(map(len, by_seed.values())),
                "runs_per_seed_max": max(map(len, by_seed.values())),
                "three_seed_complete": len(seed_ids) >= 3,
            }
        )

        total_successes = sum(row["evaluation_successes"] for row in rows)
        total_episodes = sum(row["evaluation_episodes"] for row in rows)
        ci_low, ci_high = _wilson_interval(total_successes, total_episodes)
        output.update(
            {
                "evaluation_successes_total": total_successes,
                "evaluation_episodes_total": total_episodes,
                "success_rate_pooled": total_successes / total_episodes if total_episodes else None,
                "success_rate_wilson95_low": ci_low,
                "success_rate_wilson95_high": ci_high,
            }
        )

        for metric in METRIC_KEYS:
            seed_values = []
            for seed_id in seed_ids:
                seed_rows = by_seed[seed_id]
                if metric == "success_rate":
                    seed_successes = sum(row["evaluation_successes"] for row in seed_rows)
                    seed_episodes = sum(row["evaluation_episodes"] for row in seed_rows)
                    value = seed_successes / seed_episodes if seed_episodes else None
                else:
                    values = [row[metric] for row in seed_rows if row.get(metric) is not None]
                    value = mean(values) if values else None
                if value is not None:
                    seed_values.append(value)
            output[f"{metric}_mean_across_seeds"] = mean(seed_values) if seed_values else None
            output[f"{metric}_std_across_seeds"] = stdev(seed_values) if len(seed_values) >= 2 else None
        summaries.append(output)
    return summaries


def artifact_audit(root: Path) -> dict:
    configs = list(root.glob("*/config.json"))
    statuses = list(root.glob("*/prelaunch_status.json"))
    configuration_counts = Counter()
    for path in configs:
        config = _load_json(path).get("run_config", {})
        setting = (
            config.get("algorithm", "unknown"),
            config.get("reward_variant", "unknown"),
            _coerce_bool(config.get("constrained", False)),
            config.get("colon_id", "unknown"),
            config.get("task_id", "unknown"),
            str(config.get("seed", "unknown")),
        )
        configuration_counts[setting] += 1
    return {
        "root": str(root),
        "result_directories": sum(path.is_dir() for path in root.iterdir()) if root.exists() else 0,
        "config_files": len(configs),
        "prelaunch_status_files": len(statuses),
        "evaluation_metrics_files": len(list(root.glob("**/evaluation_metrics.json"))),
        "training_log_files": len(list(root.glob("**/training_log.csv"))),
        "training_summary_files": len(list(root.glob("**/training_metrics_summary.json"))),
        "tensorboard_event_files": len(list(root.glob("**/events.out.tfevents*"))),
        "checkpoint_files": len(list(root.glob("**/*.zip"))),
        "video_files": len(list(root.glob("**/*.mp4"))),
        "configurations": [
            {
                "algorithm": setting[0],
                "reward_variant": setting[1],
                "constrained": setting[2],
                "colon_id": setting[3],
                "task_id": setting[4],
                "seed": setting[5],
                "count": count,
            }
            for setting, count in sorted(
                configuration_counts.items(), key=lambda item: tuple(map(str, item[0]))
            )
        ],
    }


def coverage_rows(
    runs: list[dict], colons: list[str], tasks: list[str], seeds: list[str],
    present_field: str = "evaluation_present",
) -> list[dict]:
    available = {
        (run["colon_id"], run["task_id"], run["reward_variant"], run["constrained"], run["seed"])
        for run in runs
    }
    rows = []
    for colon in colons:
        for task in tasks:
            for variant, constrained in EXPECTED_VARIANTS:
                for seed in seeds:
                    key = (colon, task, variant, constrained, seed)
                    rows.append(
                        {
                            "colon_id": colon,
                            "task_id": task,
                            "reward_variant": variant,
                            "constrained": constrained,
                            "seed": seed,
                            present_field: key in available,
                        }
                    )
    return rows


def write_latex_success_table(rows: list[dict], path: Path) -> None:
    """Write the paper table directly from seed-level success summaries."""
    lookup = {
        (row["reward_variant"], row["constrained"], row["task_id"]): row
        for row in rows
    }
    variants = [
        ("center_only", False, r"PPO, $s_c$ only"),
        ("depth_existence", False, r"PPO, $s_1$ only"),
        ("deep_area", False, r"PPO, $s_2$ only"),
        ("lumen_evidence", False, r"PPO, $s_o$ only"),
        ("full_reward", False, r"PPO, full $R$"),
        ("full_reward", True, r"Constrained PPO, full $R$"),
    ]

    def cell(variant, constrained, task):
        row = lookup.get((variant, constrained, task))
        if not row:
            return "--"
        average = row.get("success_rate_mean_across_seeds")
        deviation = row.get("success_rate_std_across_seeds")
        if average is None or deviation is None:
            return "--"
        return rf"{100.0 * average:.1f}$\pm${100.0 * deviation:.1f}"

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Training objective / method & Task~1 & Task~2 & Task~3 & Task~4 \\",
        r"\midrule",
    ]
    for index, (variant, constrained, label) in enumerate(variants):
        if index == 5:
            lines.append(r"\midrule")
        cells = [cell(variant, constrained, task) for task in ("t1", "t2", "t3", "t4")]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_learning_curves(
    runs: list[dict], output_dir: Path, success_scale: str = "fourth_root",
    x_axis: str = "timesteps",
) -> list[str]:
    """Plot mean geometric training success with between-seed SD bands."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    series = []
    curve_metrics = set()
    x_field = "episodes" if x_axis == "episodes" else "timestep"
    for run in runs:
        path = Path(run["training_log"])
        if not path.exists():
            continue
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        metric = (
            "success_rate_recent_100ep"
            if rows and "success_rate_recent_100ep" in rows[0]
            else "success_rate"
        )
        curve_metrics.add(metric)
        points = []
        for row in rows:
            x = _to_float(row.get(x_field))
            y = _to_float(row.get(metric))
            if x is not None and y is not None:
                points.append((x, y))
        if len(points) >= 2:
            # Logging can produce several metric rows at the same episode count.
            # Keep the latest value so interpolation has a strictly increasing x-axis.
            points = sorted(dict(points).items())
            series.append((run, np.asarray(points, dtype=float)))
    if not series:
        return []

    tasks = sorted({run["task_id"] for run, _ in series})
    figure, axes = plt.subplots(
        1, len(tasks), figsize=(4.1 * len(tasks), 3.3), squeeze=False, sharey=True
    )
    label_order = EXPECTED_VARIANTS
    labels = {
        ("center_only", False): r"$s_c$",
        ("depth_existence", False): r"$s_1$",
        ("deep_area", False): r"$s_2$",
        ("lumen_evidence", False): r"$s_o$",
        ("full_reward", False): r"$R$ (PPO)",
        ("full_reward", True): r"$R$ (constrained PPO)",
    }

    for axis, task in zip(axes[0], tasks):
        for variant_key in label_order:
            matching = [
                item
                for item in series
                if item[0]["task_id"] == task
                and (item[0]["reward_variant"], item[0]["constrained"]) == variant_key
            ]
            if not matching:
                continue
            max_step = max(points[-1, 0] for _, points in matching)
            grid = np.linspace(0.0, max_step, 250)
            seed_curves = []
            for seed in sorted({run["seed"] for run, _ in matching}, key=_seed_sort_key):
                colon_curves = []
                for _, points in [item for item in matching if item[0]["seed"] == seed]:
                    curve = np.interp(grid, points[:, 0], points[:, 1])
                    curve[(grid < points[0, 0]) | (grid > points[-1, 0])] = np.nan
                    colon_curves.append(curve)
                colon_curves = np.vstack(colon_curves)
                colon_counts = np.sum(np.isfinite(colon_curves), axis=0)
                seed_curves.append(
                    np.divide(
                        np.nansum(colon_curves, axis=0),
                        colon_counts,
                        out=np.full_like(grid, np.nan),
                        where=colon_counts > 0,
                    )
                )
            seed_curves = np.vstack(seed_curves)
            valid_counts = np.sum(np.isfinite(seed_curves), axis=0)
            curve_mean = np.divide(
                np.nansum(seed_curves, axis=0),
                valid_counts,
                out=np.full_like(grid, np.nan),
                where=valid_counts > 0,
            )
            axis.plot(grid, curve_mean, label=labels[variant_key], linewidth=1.6)
            if seed_curves.shape[0] >= 2:
                curve_std = np.nanstd(seed_curves, axis=0, ddof=1)
                axis.fill_between(
                    grid,
                    np.clip(curve_mean - curve_std, 0.0, 1.0),
                    np.clip(curve_mean + curve_std, 0.0, 1.0),
                    alpha=0.16,
                )
        axis.set_title(task.upper())
        axis.set_xlabel("Completed training episodes" if x_axis == "episodes" else "Training timesteps")
        if success_scale == "sqrt":
            axis.set_yscale(
                "function",
                functions=(
                    lambda values: np.sign(values) * np.sqrt(np.abs(values)),
                    lambda values: np.sign(values) * np.square(np.abs(values)),
                ),
            )
        elif success_scale == "fourth_root":
            axis.set_yscale(
                "function",
                functions=(
                    lambda values: np.sign(values) * np.power(np.abs(values), 0.25),
                    lambda values: np.sign(values) * np.power(np.abs(values), 4.0),
                ),
            )
        axis.set_ylim(0.0, 1.0)
        if success_scale != "linear":
            ticks = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
            axis.set_yticks(ticks)
            axis.set_yticklabels(
                ["0", "0.1%", "0.5%", "1%", "2%", "5%", "10%", "25%", "50%", "100%"]
            )
        axis.grid(alpha=0.25)
    if curve_metrics == {"success_rate_recent_100ep"}:
        axes[0, 0].set_ylabel("Rolling 100-episode goal success")
    else:
        axes[0, 0].set_ylabel("Cumulative training goal success")
    handles, legend_labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.03),
            ncol=3, frameon=False,
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    filename_stem = (
        "reward_ablation_success_by_episode"
        if x_axis == "episodes"
        else "reward_ablation_success_learning_curves"
    )
    png_path = output_dir / f"{filename_stem}.png"
    pdf_path = output_dir / f"{filename_stem}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [str(png_path), str(pdf_path)]


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="reward_ablation_results")
    parser.add_argument("--output_dir", default="reward_ablation_results/analysis")
    parser.add_argument("--expected_colons", default="c1,c2")
    parser.add_argument("--expected_tasks", default="t1,t2,t3,t4")
    parser.add_argument("--expected_seeds", default="0,1,2")
    parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero unless the full evaluation grid is present"
    )
    parser.add_argument("--no_learning_curves", action="store_true")
    parser.add_argument(
        "--success_scale",
        choices=["linear", "sqrt", "fourth_root"],
        default="fourth_root",
        help="Y-axis transform for success curves; tick labels always show the original rates",
    )
    parser.add_argument(
        "--curve_x_axis",
        choices=["timesteps", "episodes"],
        default="timesteps",
        help="Horizontal axis for the generated success curves",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = collect_runs(results_dir)
    runs, superseded = deduplicate_runs(all_runs)
    all_training_runs = collect_training_runs(results_dir)
    training_runs, superseded_training = deduplicate_runs(all_training_runs)
    by_colon = aggregate(runs, BY_COLON_KEYS)
    paper_summary = aggregate(runs, PAPER_KEYS)
    training_summary = aggregate_training_runs(training_runs, BY_COLON_KEYS)
    coverage = coverage_rows(
        runs,
        _parse_csv_list(args.expected_colons),
        _parse_csv_list(args.expected_tasks),
        _parse_csv_list(args.expected_seeds),
    )
    missing = [row for row in coverage if not row["evaluation_present"]]
    training_coverage = coverage_rows(
        training_runs,
        _parse_csv_list(args.expected_colons),
        _parse_csv_list(args.expected_tasks),
        _parse_csv_list(args.expected_seeds),
        present_field="training_present",
    )
    missing_training = [row for row in training_coverage if not row["training_present"]]

    _write_csv(runs, output_dir / "reward_ablation_runs.csv")
    _write_csv(by_colon, output_dir / "reward_ablation_summary.csv")
    _write_json(by_colon, output_dir / "reward_ablation_summary.json")
    _write_csv(paper_summary, output_dir / "reward_ablation_paper_summary.csv")
    _write_json(paper_summary, output_dir / "reward_ablation_paper_summary.json")
    _write_csv(training_runs, output_dir / "reward_ablation_training_runs.csv")
    _write_csv(training_summary, output_dir / "reward_ablation_training_summary.csv")
    _write_json(training_summary, output_dir / "reward_ablation_training_summary.json")
    _write_csv(coverage, output_dir / "reward_ablation_coverage.csv")
    _write_json(
        {
            "expected_evaluations": len(coverage),
            "present_evaluations": len(coverage) - len(missing),
            "missing_evaluations": len(missing),
            "complete": not missing,
            "missing": missing,
            "superseded_evaluation_runs": [row["run_dir"] for row in superseded],
        },
        output_dir / "reward_ablation_coverage.json",
    )
    _write_json(
        {
            "expected_training_runs": len(training_coverage),
            "present_training_runs": len(training_coverage) - len(missing_training),
            "missing_training_runs": len(missing_training),
            "complete": not missing_training,
            "missing": missing_training,
            "superseded_training_runs": [row["run_dir"] for row in superseded_training],
        },
        output_dir / "reward_ablation_training_coverage.json",
    )
    _write_json(artifact_audit(results_dir), output_dir / "reward_ablation_artifact_audit.json")
    write_latex_success_table(
        paper_summary, output_dir / "reward_ablation_success_table.tex"
    )
    curve_paths = (
        []
        if args.no_learning_curves
        else write_learning_curves(
            training_runs, output_dir, args.success_scale, args.curve_x_axis
        )
    )

    print(
        f"Found {len(all_runs)} deterministic evaluation(s); retained {len(runs)} after deduplication."
    )
    print(
        f"Found {len(all_training_runs)} training log(s); retained {len(training_runs)} after deduplication."
    )
    print(
        f"Coverage: {len(coverage) - len(missing)}/{len(coverage)} expected setting/seed evaluations."
    )
    print(
        f"Training coverage: {len(training_coverage) - len(missing_training)}/{len(training_coverage)} expected setting/seed runs."
    )
    print(f"Wrote analysis to {output_dir}")
    if curve_paths:
        print("Wrote learning curves: " + ", ".join(curve_paths))
    if args.strict and missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
