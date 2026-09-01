import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "aggregate_reward_ablation.py"
SPEC = importlib.util.spec_from_file_location("aggregate_reward_ablation", MODULE_PATH)
AGGREGATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATION)


def _run(seed, successes, episodes=10, colon_id="c1", timestamp=None):
    return {
        "run_dir": f"run-{colon_id}-{seed}-{timestamp or seed}",
        "timestamp": str(timestamp or seed),
        "colon_id": colon_id,
        "task_id": "t1",
        "algorithm": "PPO",
        "constrained": False,
        "reward_variant": "full_reward",
        "seed": str(seed),
        "evaluation_episodes": episodes,
        "evaluation_successes": successes,
        "success_rate": successes / episodes,
        "training_log": "missing.csv",
        **{key: None for key in AGGREGATION.METRIC_KEYS if key != "success_rate"},
    }


def test_aggregate_reports_sample_variance_across_three_seeds():
    rows = [_run(0, 5), _run(1, 6), _run(2, 7)]

    summary = AGGREGATION.aggregate(rows, AGGREGATION.BY_COLON_KEYS)[0]

    assert summary["num_seeds"] == 3
    assert summary["three_seed_complete"] is True
    assert summary["success_rate_mean_across_seeds"] == pytest.approx(0.6)
    assert summary["success_rate_std_across_seeds"] == pytest.approx(0.1)
    assert summary["success_rate_pooled"] == pytest.approx(0.6)
    assert summary["evaluation_episodes_total"] == 30
    assert summary["success_rate_wilson95_low"] < 0.6 < summary["success_rate_wilson95_high"]


def test_paper_summary_averages_colons_within_each_seed_first():
    rows = [
        _run(0, 10, colon_id="c1"),
        _run(0, 0, colon_id="c2"),
        _run(1, 8, colon_id="c1"),
        _run(1, 4, colon_id="c2"),
        _run(2, 6, colon_id="c1"),
        _run(2, 8, colon_id="c2"),
    ]

    summary = AGGREGATION.aggregate(rows, AGGREGATION.PAPER_KEYS)[0]

    assert summary["runs_per_seed_min"] == 2
    assert summary["runs_per_seed_max"] == 2
    assert summary["success_rate_mean_across_seeds"] == pytest.approx(0.6)
    assert summary["success_rate_std_across_seeds"] == pytest.approx(0.1)


def test_deduplicate_keeps_newest_run_for_each_seed():
    old = _run(0, 1, timestamp="20260101_000000")
    new = _run(0, 9, timestamp="20260102_000000")

    selected, superseded = AGGREGATION.deduplicate_runs([new, old])

    assert selected == [new]
    assert superseded == [old]


def test_collect_runs_uses_recorded_binary_evaluation_outcomes(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_config": {
                    "algorithm": "PPO",
                    "reward_variant": "center_only",
                    "constrained": False,
                    "colon_id": "c1",
                    "task_id": "t2",
                    "seed": 2,
                    "timestamp": "20260101_000000",
                }
            }
        )
    )
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(
            {
                "episodes_completed": 4,
                "episode_successes": [True, False, True, True],
                # Deliberately inconsistent: binary outcomes must win.
                "success_rate": 0.0,
                "normalized_progress": 0.75,
            }
        )
    )

    run = AGGREGATION.collect_runs(tmp_path)[0]

    assert run["evaluation_successes"] == 3
    assert run["success_rate"] == pytest.approx(0.75)
    assert run["normalized_progress"] == pytest.approx(0.75)
    assert run["evaluation_source"] == "deterministic_post_training"


def test_collect_training_runs_keeps_training_success_separate(tmp_path):
    run_dir = tmp_path / "run"
    success_dir = run_dir / "trajectory_data"
    success_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_config": {
                    "algorithm": "PPO",
                    "reward_variant": "full_reward",
                    "constrained": True,
                    "colon_id": "c1",
                    "task_id": "t4",
                    "seed": 0,
                    "timestamp": "20260101_000000",
                }
            }
        )
    )
    (run_dir / "training_log.csv").write_text(
        "timestep,success_rate,episodes\n500,0.2,10\n"
    )
    (success_dir / "success_rate_data.json").write_text(
        json.dumps({"summary": {"total_episodes": 10, "total_successes": 3}})
    )

    run = AGGREGATION.collect_training_runs(tmp_path)[0]

    assert run["training_successes"] == 3
    assert run["training_episodes"] == 10
    assert run["training_success_rate"] == pytest.approx(0.3)
    assert "evaluation_successes" not in run


def test_expected_grid_contains_six_variants_per_colon_task_seed():
    coverage = AGGREGATION.coverage_rows([], ["c1", "c2"], ["t1", "t2", "t3", "t4"], ["0", "1", "2"])

    assert len(coverage) == 144
    assert not any(row["evaluation_present"] for row in coverage)


def test_latex_table_formats_seed_mean_and_standard_deviation(tmp_path):
    summaries = AGGREGATION.aggregate(
        [_run(0, 5), _run(1, 6), _run(2, 7)], AGGREGATION.PAPER_KEYS
    )
    path = tmp_path / "table.tex"

    AGGREGATION.write_latex_success_table(summaries, path)

    table = path.read_text()
    assert r"PPO, full $R$ & 60.0$\pm$10.0" in table
    assert r"Constrained PPO, full $R$ & --" in table
