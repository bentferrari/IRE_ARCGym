#!/usr/bin/env python3
"""Generate simple SVG plots from trajectory JSON logs without external deps."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
TRAJ_ROOT = ROOT / "trajectory_data"
PLOTS_DIR = TRAJ_ROOT / "plots"

SVG_WIDTH = 900
SVG_HEIGHT = 520
MARGIN = 70

COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _linspace_ticks(min_val: float, max_val: float, count: int = 5) -> List[float]:
    if count <= 1:
        return [min_val]
    if math.isclose(min_val, max_val):
        return [min_val + i for i in range(count)]
    step = (max_val - min_val) / (count - 1)
    return [min_val + i * step for i in range(count)]


def _scale(value: float, vmin: float, vmax: float, pixel_min: float, pixel_max: float) -> float:
    if math.isclose(vmin, vmax):
        return (pixel_min + pixel_max) / 2.0
    return pixel_min + (value - vmin) * (pixel_max - pixel_min) / (vmax - vmin)


def _svg_header(width: int, height: int) -> List[str]:
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white' />",
    ]


def _svg_footer() -> List[str]:
    return ["</svg>"]


def _draw_axes(lines: List[str], x_min: float, x_max: float, y_min: float, y_max: float,
               x_label: str, y_label: str, title: str) -> None:
    plot_left = MARGIN
    plot_right = SVG_WIDTH - MARGIN
    plot_top = MARGIN
    plot_bottom = SVG_HEIGHT - MARGIN

    lines.append(f"<line x1='{plot_left}' y1='{plot_bottom}' x2='{plot_right}' y2='{plot_bottom}' stroke='#222' stroke-width='1' />")
    lines.append(f"<line x1='{plot_left}' y1='{plot_bottom}' x2='{plot_left}' y2='{plot_top}' stroke='#222' stroke-width='1' />")

    for tick in _linspace_ticks(x_min, x_max, 5):
        x = _scale(tick, x_min, x_max, plot_left, plot_right)
        lines.append(f"<line x1='{x:.1f}' y1='{plot_bottom}' x2='{x:.1f}' y2='{plot_bottom + 6}' stroke='#222' stroke-width='1' />")
        lines.append(
            f"<text x='{x:.1f}' y='{plot_bottom + 22}' text-anchor='middle' font-size='11' fill='#222'>"
            f"{tick:.0f}</text>"
        )

    for tick in _linspace_ticks(y_min, y_max, 5):
        y = _scale(tick, y_min, y_max, plot_bottom, plot_top)
        lines.append(f"<line x1='{plot_left - 6}' y1='{y:.1f}' x2='{plot_left}' y2='{y:.1f}' stroke='#222' stroke-width='1' />")
        lines.append(
            f"<text x='{plot_left - 10}' y='{y + 4:.1f}' text-anchor='end' font-size='11' fill='#222'>"
            f"{tick:.2f}</text>"
        )

    lines.append(
        f"<text x='{(plot_left + plot_right) / 2:.1f}' y='{SVG_HEIGHT - 20}' text-anchor='middle' "
        f"font-size='13' fill='#111'>{x_label}</text>"
    )
    lines.append(
        f"<text x='20' y='{(plot_top + plot_bottom) / 2:.1f}' text-anchor='middle' "
        f"font-size='13' fill='#111' transform='rotate(-90 20 {(plot_top + plot_bottom) / 2:.1f})'>{y_label}</text>"
    )
    lines.append(
        f"<text x='{(plot_left + plot_right) / 2:.1f}' y='32' text-anchor='middle' font-size='16' fill='#111'>"
        f"{title}</text>"
    )


def _draw_legend(lines: List[str], labels: List[str], colors: List[str]) -> None:
    start_x = SVG_WIDTH - MARGIN + 10
    start_y = MARGIN + 10
    for idx, (label, color) in enumerate(zip(labels, colors)):
        y = start_y + idx * 18
        lines.append(f"<rect x='{start_x}' y='{y - 10}' width='12' height='12' fill='{color}' />")
        lines.append(
            f"<text x='{start_x + 18}' y='{y}' font-size='11' fill='#111' dominant-baseline='middle'>{label}</text>"
        )


def _line_plot(series: List[Tuple[List[float], List[float]]], labels: List[str],
               x_label: str, y_label: str, title: str, out_path: Path) -> None:
    all_x = [x for xs, _ in series for x in xs]
    all_y = [y for _, ys in series for y in ys]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    lines = _svg_header(SVG_WIDTH, SVG_HEIGHT)
    _draw_axes(lines, x_min, x_max, y_min, y_max, x_label, y_label, title)

    plot_left = MARGIN
    plot_right = SVG_WIDTH - MARGIN
    plot_top = MARGIN
    plot_bottom = SVG_HEIGHT - MARGIN

    for idx, (xs, ys) in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        points = []
        for x, y in zip(xs, ys):
            px = _scale(x, x_min, x_max, plot_left, plot_right)
            py = _scale(y, y_min, y_max, plot_bottom, plot_top)
            points.append(f"{px:.1f},{py:.1f}")
        if points:
            lines.append(
                f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(points)}' />"
            )
            for point in points:
                px, py = point.split(",")
                lines.append(f"<circle cx='{px}' cy='{py}' r='3' fill='{color}' />")

    _draw_legend(lines, labels, [COLORS[i % len(COLORS)] for i in range(len(labels))])
    lines.extend(_svg_footer())
    out_path.write_text("\n".join(lines))


def _line_plot_with_band(xs: List[float], ys_mean: List[float], ys_low: List[float], ys_high: List[float],
                         x_label: str, y_label: str, title: str, out_path: Path,
                         color: str = "#1b9e77", band_opacity: float = 0.2) -> None:
    if not xs or not ys_mean:
        return
    x_min, x_max = min(xs), max(xs)
    y_min = min(min(ys_low), min(ys_mean))
    y_max = max(max(ys_high), max(ys_mean))

    lines = _svg_header(SVG_WIDTH, SVG_HEIGHT)
    _draw_axes(lines, x_min, x_max, y_min, y_max, x_label, y_label, title)

    plot_left = MARGIN
    plot_right = SVG_WIDTH - MARGIN
    plot_top = MARGIN
    plot_bottom = SVG_HEIGHT - MARGIN

    if len(xs) > 1:
        upper_points = []
        lower_points = []
        for x, y in zip(xs, ys_high):
            px = _scale(x, x_min, x_max, plot_left, plot_right)
            py = _scale(y, y_min, y_max, plot_bottom, plot_top)
            upper_points.append(f"{px:.1f},{py:.1f}")
        for x, y in zip(reversed(xs), reversed(ys_low)):
            px = _scale(x, x_min, x_max, plot_left, plot_right)
            py = _scale(y, y_min, y_max, plot_bottom, plot_top)
            lower_points.append(f"{px:.1f},{py:.1f}")
        band_points = " ".join(upper_points + lower_points)
        lines.append(
            f"<polygon points='{band_points}' fill='{color}' fill-opacity='{band_opacity}' stroke='none' />"
        )

    mean_points = []
    for x, y in zip(xs, ys_mean):
        px = _scale(x, x_min, x_max, plot_left, plot_right)
        py = _scale(y, y_min, y_max, plot_bottom, plot_top)
        mean_points.append(f"{px:.1f},{py:.1f}")
    if mean_points:
        lines.append(
            f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{' '.join(mean_points)}' />"
        )
        for point in mean_points:
            px, py = point.split(",")
            lines.append(f"<circle cx='{px}' cy='{py}' r='3' fill='{color}' />")

    lines.extend(_svg_footer())
    out_path.write_text("\n".join(lines))


def _bar_chart(categories: List[str], series: Dict[str, List[float]],
               x_label: str, y_label: str, title: str, out_path: Path) -> None:
    max_val = max(max(vals) for vals in series.values()) if series else 1.0
    min_val = min(min(vals) for vals in series.values()) if series else 0.0

    lines = _svg_header(SVG_WIDTH, SVG_HEIGHT)
    _draw_axes(lines, 0, len(categories), min_val, max_val, x_label, y_label, title)

    plot_left = MARGIN
    plot_right = SVG_WIDTH - MARGIN
    plot_top = MARGIN
    plot_bottom = SVG_HEIGHT - MARGIN

    group_width = (plot_right - plot_left) / max(len(categories), 1)
    series_count = max(len(series), 1)
    bar_width = group_width / (series_count + 1)

    for idx, category in enumerate(categories):
        x_group = plot_left + idx * group_width
        for s_idx, (label, values) in enumerate(series.items()):
            color = COLORS[s_idx % len(COLORS)]
            val = values[idx]
            bar_height = _scale(val, min_val, max_val, 0, plot_bottom - plot_top)
            x = x_group + (s_idx + 0.2) * bar_width
            y = plot_bottom - bar_height
            lines.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width * 0.8:.1f}' height='{bar_height:.1f}' fill='{color}' />"
            )
        lines.append(
            f"<text x='{x_group + group_width / 2:.1f}' y='{plot_bottom + 18}' text-anchor='middle' "
            f"font-size='11' fill='#111'>{category}</text>"
        )

    _draw_legend(lines, list(series.keys()), [COLORS[i % len(COLORS)] for i in range(len(series))])
    lines.extend(_svg_footer())
    out_path.write_text("\n".join(lines))


def _histogram(data: List[float], bins: int = 20) -> Tuple[List[float], List[int]]:
    if not data:
        return [], []
    dmin, dmax = min(data), max(data)
    if math.isclose(dmin, dmax):
        return [dmin], [len(data)]
    bin_width = (dmax - dmin) / bins
    edges = [dmin + i * bin_width for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for value in data:
        idx = min(int((value - dmin) / bin_width), bins - 1)
        counts[idx] += 1
    centers = [edges[i] + bin_width / 2 for i in range(bins)]
    return centers, counts


def _hist_plot(data: List[float], bins: int, x_label: str, y_label: str,
               title: str, out_path: Path) -> None:
    centers, counts = _histogram(data, bins)
    if not centers:
        return
    x_min, x_max = min(centers), max(centers)
    y_min, y_max = 0, max(counts)

    lines = _svg_header(SVG_WIDTH, SVG_HEIGHT)
    _draw_axes(lines, x_min, x_max, y_min, y_max, x_label, y_label, title)

    plot_left = MARGIN
    plot_right = SVG_WIDTH - MARGIN
    plot_top = MARGIN
    plot_bottom = SVG_HEIGHT - MARGIN

    bar_width = (plot_right - plot_left) / max(len(centers), 1)
    for idx, (center, count) in enumerate(zip(centers, counts)):
        x = _scale(center, x_min, x_max, plot_left, plot_right) - bar_width * 0.4
        bar_height = _scale(count, y_min, y_max, 0, plot_bottom - plot_top)
        y = plot_bottom - bar_height
        lines.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width * 0.8:.1f}' height='{bar_height:.1f}' fill='#4c78a8' />"
        )

    lines.extend(_svg_footer())
    out_path.write_text("\n".join(lines))


def _scatter_plot(xs: List[float], ys: List[float], x_label: str, y_label: str,
                  title: str, out_path: Path) -> None:
    if not xs or not ys:
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    lines = _svg_header(SVG_WIDTH, SVG_HEIGHT)
    _draw_axes(lines, x_min, x_max, y_min, y_max, x_label, y_label, title)

    plot_left = MARGIN
    plot_right = SVG_WIDTH - MARGIN
    plot_top = MARGIN
    plot_bottom = SVG_HEIGHT - MARGIN

    for x, y in zip(xs, ys):
        px = _scale(x, x_min, x_max, plot_left, plot_right)
        py = _scale(y, y_min, y_max, plot_bottom, plot_top)
        lines.append(f"<circle cx='{px:.1f}' cy='{py:.1f}' r='3' fill='#1b9e77' fill-opacity='0.75' />")

    lines.extend(_svg_footer())
    out_path.write_text("\n".join(lines))


def _load_success_data() -> List[Tuple[str, Dict]]:
    runs = []
    for path in TRAJ_ROOT.rglob("success_rate_data.json"):
        with path.open() as f:
            data = json.load(f)
        runs.append((path.parent.name, data))
    runs.sort(key=lambda item: item[0])
    return runs


def _collect_episode_arrays(run_data: Dict) -> Tuple[List[float], List[float]]:
    rewards: List[float] = []
    lengths: List[float] = []
    for env_data in run_data.get("per_environment", {}).values():
        rewards.extend(env_data.get("episode_rewards", []))
        lengths.extend(env_data.get("episode_lengths", []))
    return rewards, lengths


def _collect_metadata_points(run_dir: Path) -> Tuple[List[float], List[float]]:
    rewards: List[float] = []
    stresses: List[float] = []
    for path in run_dir.rglob("metadata.json"):
        with path.open() as f:
            data = json.load(f)
        rewards.append(data.get("episode_reward", 0.0))
        stresses.append(data.get("colon_stress", 0.0))
    return rewards, stresses


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def _success_rate_band(primary_data: Dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    xs: List[float] = []
    means: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    for point in primary_data.get("history", []):
        env_rates = list(point.get("env_success_rates_cumulative", {}).values())
        if not env_rates:
            continue
        mean, std = _mean_std(env_rates)
        xs.append(point["timestep"])
        means.append(mean)
        lows.append(max(0.0, mean - std))
        highs.append(min(1.0, mean + std))
    return xs, means, lows, highs


def _reward_band(primary_data: Dict) -> Tuple[List[float], List[float], List[float], List[float]]:
    per_env = primary_data.get("per_environment", {})
    reward_series = [env.get("episode_rewards", []) for env in per_env.values() if env.get("episode_rewards")]
    if not reward_series:
        return [], [], [], []
    max_len = max(len(series) for series in reward_series)
    xs: List[float] = []
    means: List[float] = []
    lows: List[float] = []
    highs: List[float] = []
    for idx in range(max_len):
        values = [series[idx] for series in reward_series if idx < len(series)]
        if not values:
            continue
        mean, std = _mean_std(values)
        xs.append(idx + 1)
        means.append(mean)
        lows.append(mean - std)
        highs.append(mean + std)
    return xs, means, lows, highs


def main() -> None:
    _ensure_dir(PLOTS_DIR)
    runs = _load_success_data()
    if not runs:
        raise SystemExit("No success_rate_data.json files found in trajectory_data.")

    # Success rate over time (all runs)
    series = []
    labels = []
    for run_id, data in runs:
        history = data.get("history", [])
        if not history:
            continue
        xs = [point["timestep"] for point in history]
        ys = [point["overall_success_rate"] for point in history]
        series.append((xs, ys))
        labels.append(run_id)

    if series:
        _line_plot(
            series,
            labels,
            x_label="Timesteps",
            y_label="Overall Success Rate",
            title="Success Rate Over Training",
            out_path=PLOTS_DIR / "success_rate_over_time.svg",
        )

    # Choose the run with the most episodes for per-env and distribution plots
    primary_run_id, primary_data = max(
        runs,
        key=lambda item: item[1].get("summary", {}).get("total_episodes", 0),
    )

    per_env = primary_data.get("per_environment", {})
    env_ids = sorted(per_env.keys(), key=int)
    cumulative_rates = [per_env[e]["success_rate_cumulative"] for e in env_ids]
    window_rates = [per_env[e]["success_rate_10ep"] for e in env_ids]

    _bar_chart(
        env_ids,
        {"Cumulative": cumulative_rates, "Last-10 Episodes": window_rates},
        x_label="Environment",
        y_label="Success Rate",
        title=f"Per-Environment Success Rates ({primary_run_id})",
        out_path=PLOTS_DIR / "per_env_success_rate.svg",
    )

    xs, means, lows, highs = _success_rate_band(primary_data)
    _line_plot_with_band(
        xs,
        means,
        lows,
        highs,
        x_label="Timesteps",
        y_label="Success Rate (Mean across envs)",
        title=f"Success Rate with Env Variability ({primary_run_id})",
        out_path=PLOTS_DIR / "success_rate_with_band.svg",
    )

    rewards, lengths = _collect_episode_arrays(primary_data)
    _hist_plot(
        rewards,
        bins=20,
        x_label="Episode Reward",
        y_label="Count",
        title=f"Episode Reward Distribution ({primary_run_id})",
        out_path=PLOTS_DIR / "episode_reward_hist.svg",
    )
    _hist_plot(
        lengths,
        bins=20,
        x_label="Episode Length",
        y_label="Count",
        title=f"Episode Length Distribution ({primary_run_id})",
        out_path=PLOTS_DIR / "episode_length_hist.svg",
    )
    _scatter_plot(
        lengths,
        rewards,
        x_label="Episode Length",
        y_label="Episode Reward",
        title=f"Reward vs. Length ({primary_run_id})",
        out_path=PLOTS_DIR / "reward_vs_length.svg",
    )

    xs, means, lows, highs = _reward_band(primary_data)
    _line_plot_with_band(
        xs,
        means,
        lows,
        highs,
        x_label="Episode Index (per environment)",
        y_label="Episode Reward (Mean across envs)",
        title=f"Episode Reward with Env Variability ({primary_run_id})",
        out_path=PLOTS_DIR / "reward_with_band.svg",
    )

    # Optional safety/strain metric scatter from metadata
    run_dir = TRAJ_ROOT / primary_run_id
    meta_rewards, meta_stress = _collect_metadata_points(run_dir)
    if meta_rewards and meta_stress:
        _scatter_plot(
            meta_stress,
            meta_rewards,
            x_label="Colon Stress",
            y_label="Episode Reward",
            title=f"Colon Stress vs. Reward ({primary_run_id})",
            out_path=PLOTS_DIR / "colon_stress_vs_reward.svg",
        )


if __name__ == "__main__":
    main()
