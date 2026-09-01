#!/usr/bin/env python3
"""Pool repeated deterministic evaluations without calling them training seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(args.results_dir.glob("*/eval-seed-*/evaluation_metrics.json")):
        with path.open() as file:
            metrics = json.load(file)
        metrics["_path"] = str(path)
        grouped[path.parents[1].name].append(metrics)

    rows = []
    for source_run, repeats in sorted(grouped.items()):
        successes = sum(sum(bool(value) for value in row.get("episode_successes", [])) for row in repeats)
        episodes = sum(int(row.get("episodes_completed", 0)) for row in repeats)
        first = repeats[0]
        rows.append(
            {
                "source_run": source_run,
                "colon_id": first.get("colon_id"),
                "task_id": first.get("task_id"),
                "algorithm": first.get("algorithm"),
                "reward_variant": first.get("reward_variant"),
                "constrained": first.get("constrained"),
                "training_seed": first.get("training_seed", first.get("seed")),
                "evaluation_seeds": ",".join(str(row.get("evaluation_seed")) for row in repeats),
                "evaluation_repeats": len(repeats),
                "episodes_per_repeat": ",".join(str(row.get("episodes_completed", 0)) for row in repeats),
                "successes_total": successes,
                "episodes_total": episodes,
                "success_rate_pooled": successes / episodes if episodes else None,
                "repeat_success_rates": ",".join(str(row.get("success_rate")) for row in repeats),
                "episode_length_s": first.get("episode_length_s"),
                "max_episode_steps": first.get("max_episode_steps"),
                "model_path": first.get("model_path"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        if rows:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(rows)} pooled policy evaluation(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
