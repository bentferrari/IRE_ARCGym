import argparse
from datetime import datetime
import logging
import os
import pdb
import json
import csv
import random
import subprocess
import sys
import faulthandler
import numpy as np
from PIL import Image

from sane_rich_logging import setup_logging

setup_logging()
faulthandler.enable(all_threads=True)

ABLATION_REWARD_RUNS = [
    ("center_only", False),
    ("depth_existence", False),
    ("deep_area", False),
    ("lumen_evidence", False),
    ("full_reward", False),
    ("full_reward", True),
]
PARAMETER_SENSITIVITY_SETTINGS = [
    (0.5, 0.45),
    (0.5, 0.49),
    (5.0, 0.45),
    (5.0, 0.49),
    (10.0, 0.45),
    (10.0, 0.49),
]
PARAMETER_SENSITIVITY_TASKS = ["t1", "t2", "t3", "t4"]
ANCHOR_SENSITIVITY_SETTINGS = ["sparse", "default", "dense"]
ANCHOR_SENSITIVITY_TASKS = ["t1", "t2", "t3", "t4"]
CROSS_COLON2_TASKS = ["t1", "t2", "t3", "t4"]
CROSS_COLON2_TRIALS = 2
CROSS_COLON_IDS = ["c2", "c3", "c4", "c5"]
CROSS_COLON_TASKS = ["t1", "t2", "t3", "t4"]
DATA_SAMPLING_SEQUENCE_COLONS = ["c1", "c3", "c4", "c5"]
DATA_SAMPLING_SEQUENCE_REPEATS = 2
DEFAULT_MEDICAL_DATA_DIR = "/home/guanglin/data_arcgym1/sim4medical_training_data"


def _parse_data_sampling_sequence_colons(value):
    if value is None:
        return list(DATA_SAMPLING_SEQUENCE_COLONS)
    selected_colons = []
    for raw_colon_id in str(value).split(","):
        colon_id = raw_colon_id.strip().lower()
        if not colon_id:
            continue
        if colon_id.isdigit():
            colon_id = f"c{colon_id}"
        elif colon_id.startswith("colon") and colon_id[5:].isdigit():
            colon_id = f"c{colon_id[5:]}"
        if colon_id not in selected_colons:
            selected_colons.append(colon_id)
    if not selected_colons:
        raise ValueError("No valid colon ids found for --data_sampling_sequence_colons")
    return selected_colons


def _resolve_model_path(model_path):
    """Resolve a checkpoint path and accept either with or without .zip."""
    expanded_path = os.path.expanduser(model_path)
    candidates = [expanded_path]
    if expanded_path.endswith(".zip"):
        candidates.append(expanded_path[:-4])
    else:
        candidates.append(f"{expanded_path}.zip")

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    checked = ", ".join(candidates)
    raise FileNotFoundError(f"Model checkpoint not found. Checked: {checked}")


def _format_parameter_value(value):
    return f"{float(value):g}".replace(".", "p")


def _write_parameter_sensitivity_summary(rows, csv_path, json_path):
    fieldnames = [
        "youngs_modulus_mpa",
        "poisson_ratio",
        "task_id",
        "result_dir",
        "status",
        "total_episodes",
        "success_rate",
        "raw_mean_step_reward",
        "normalized_mean_step_reward",
        "mean_s_c",
        "mean_s_1",
        "mean_s_2",
        "mean_s_3",
        "mean_s_o",
        "normalized_progress",
        "roi_alignment_rate",
        "lumen_visible_ratio",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


def _write_anchor_sensitivity_summary(rows, csv_path, json_path):
    fieldnames = [
        "anchor_setting",
        "task_id",
        "result_dir",
        "status",
        "total_episodes",
        "success_rate",
        "raw_mean_step_reward",
        "normalized_mean_step_reward",
        "mean_s_c",
        "mean_s_1",
        "mean_s_2",
        "mean_s_3",
        "mean_s_o",
        "normalized_progress",
        "roi_alignment_rate",
        "lumen_visible_ratio",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


def _write_cross_colon_summary(rows, csv_path, json_path):
    fieldnames = [
        "row_type",
        "colon_id",
        "task_id",
        "trial_id",
        "result_dir",
        "status",
        "expected_episodes",
        "total_episodes",
        "total_successes",
        "success_rate",
        "raw_mean_step_reward",
        "normalized_mean_step_reward",
        "mean_s_c",
        "mean_s_1",
        "mean_s_2",
        "mean_s_3",
        "mean_s_o",
        "normalized_progress",
        "roi_alignment_rate",
        "lumen_visible_ratio",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


def _write_data_sampling_sequence_summary(rows, csv_path, json_path):
    fieldnames = [
        "colon_id",
        "trial_id",
        "result_dir",
        "medical_data_dir",
        "init_from_csv",
        "init_endpose_from_csv",
        "status",
        "target_reaches",
        "total_env_steps",
        "model_path",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


def _weighted_mean(rows, key):
    weighted_sum = 0.0
    total_weight = 0
    for row in rows:
        value = row.get(key)
        weight = row.get("total_episodes") or 0
        if value is None or weight is None:
            continue
        try:
            weighted_sum += float(value) * int(weight)
            total_weight += int(weight)
        except (TypeError, ValueError):
            continue
    return weighted_sum / total_weight if total_weight else None


def _load_json_if_exists(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        logging.warning("Failed to load %s: %s", path, exc)
        return {}


def _strip_controlled_cli_args(argv, value_options, flag_options):
    """Remove options controlled by the ablation launcher before spawning child runs."""
    stripped = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        option = arg.split("=", 1)[0]
        if option in value_options:
            index += 1 if "=" in arg else 2
            continue
        if option in flag_options:
            index += 1
            continue
        stripped.append(arg)
        index += 1
    return stripped


def _run_reward_ablation_launcher_if_requested():
    """
    Launch reward ablations before importing Isaac/Kit.

    Keeping the parent process free of Isaac native modules avoids brittle
    repeated Kit initialization when the launcher starts six child trainings.
    """
    early_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    early_parser.add_argument("--ablation_t1", action="store_true")
    early_parser.add_argument("--ablation_t2", action="store_true")
    early_parser.add_argument("--ablation_t3", action="store_true")
    early_parser.add_argument("--ablation_t4", action="store_true")
    early_parser.add_argument("--ablation_episodes", type=int, default=300)
    early_parser.add_argument("--colon_id", type=str, default="c1")
    early_args, _ = early_parser.parse_known_args()

    ablation_tasks = [
        ("t1", early_args.ablation_t1),
        ("t2", early_args.ablation_t2),
        ("t3", early_args.ablation_t3),
        ("t4", early_args.ablation_t4),
    ]
    selected_tasks = [task_id for task_id, enabled in ablation_tasks if enabled]
    if not selected_tasks:
        return
    if len(selected_tasks) > 1:
        raise SystemExit("Use only one of --ablation_t1, --ablation_t2, --ablation_t3, or --ablation_t4")

    value_options = {
        "--ablation_episodes",
        "--stop_after_episodes",
        "--model_path",
        "--reward_variant",
        "--colon_id",
        "--task_id",
    }
    flag_options = {
        "--ablation_t1",
        "--ablation_t2",
        "--ablation_t3",
        "--ablation_t4",
        "--train",
        "--test",
        "--continual_training",
        "--data_sampling",
        "--clip_actions",
        "--no-clip_actions",
    }
    base_child_args = _strip_controlled_cli_args(sys.argv[1:], value_options, flag_options)
    selected_task_id = selected_tasks[0]

    for run_index, (variant, constrained) in enumerate(ABLATION_REWARD_RUNS, start=1):
        child_cmd = [
            sys.executable,
            os.path.abspath(__file__),
            *base_child_args,
            "--train",
            "--colon_id",
            early_args.colon_id,
            "--task_id",
            selected_task_id,
            "--reward_variant",
            variant,
            "--stop_after_episodes",
            str(early_args.ablation_episodes),
        ]
        if constrained:
            child_cmd.append("--clip_actions")
            label = "constrained_full_reward"
        else:
            child_cmd.append("--no-clip_actions")
            label = variant

        logging.info(
            "Reward ablation %s/%s for %s: %s",
            run_index,
            len(ABLATION_REWARD_RUNS),
            selected_task_id,
            label,
        )
        logging.info("Command: %s", " ".join(child_cmd))
        completed = subprocess.run(child_cmd)
        if completed.returncode != 0:
            logging.error(
                "Reward ablation run failed for task=%s variant=%s constrained=%s with exit code %s",
                selected_task_id,
                variant,
                constrained,
                completed.returncode,
            )
            sys.exit(completed.returncode)

    logging.info(
        "Completed reward ablation sequence for colon=%s task=%s, %s episodes per run",
        early_args.colon_id,
        selected_task_id,
        early_args.ablation_episodes,
    )
    sys.exit(0)


def _run_parameter_sensitivity_launcher_if_requested():
    """
    Launch parameter sensitivity runs before importing Isaac/Kit.

    Each child process loads the same checkpoint with --continual_training and
    applies one colon material setting to all vectorized environments. The
    sequence is setting 1 task 1-4, then setting 2 task 1-4, and so on.
    """
    early_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    early_parser.add_argument("--parameter_sensitivity_test", action="store_true")
    early_parser.add_argument("--model_path", type=str, default=None)
    early_parser.add_argument("--seed", type=int, default=0)
    early_parser.add_argument("--num_envs", type=int, default=2)
    early_parser.add_argument("--parameter_sensitivity_episodes", type=int, default=300)
    early_parser.add_argument("--parameter_sensitivity_results_dir", type=str, default=None)
    early_args, _ = early_parser.parse_known_args()

    if not early_args.parameter_sensitivity_test:
        return
    if early_args.model_path is None:
        raise SystemExit("--model_path is required for --parameter_sensitivity_test")
    try:
        early_args.model_path = _resolve_model_path(early_args.model_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = early_args.parameter_sensitivity_results_dir or os.path.join(
        "parameter_sensitivity_results",
        f"parameter_sensitivity_colon-c1_seed-{early_args.seed}_{timestamp}",
    )
    os.makedirs(results_root, exist_ok=True)

    value_options = {
        "--parameter_sensitivity_episodes",
        "--parameter_sensitivity_results_dir",
        "--model_path",
        "--youngs_modulus",
        "--poisson_ratio",
        "--result_dir",
        "--colon_id",
        "--task_id",
        "--stop_after_episodes",
        "--continual_training_target_episodes",
    }
    flag_options = {
        "--parameter_sensitivity_test",
        "--train",
        "--test",
        "--continual_training",
        "--data_sampling",
    }
    base_child_args = _strip_controlled_cli_args(sys.argv[1:], value_options, flag_options)
    summary_rows = []
    summary_csv = os.path.join(results_root, "parameter_sensitivity_summary.csv")
    summary_json = os.path.join(results_root, "parameter_sensitivity_summary.json")
    per_env_target_episodes = max(
        1,
        int(np.ceil(early_args.parameter_sensitivity_episodes / max(1, early_args.num_envs))),
    )

    for setting_index, (youngs_modulus_mpa, poisson_ratio) in enumerate(PARAMETER_SENSITIVITY_SETTINGS, start=1):
        material_name = (
            f"E_{_format_parameter_value(youngs_modulus_mpa)}MPa_"
            f"nu_{_format_parameter_value(poisson_ratio)}"
        )
        for task_index, task_id in enumerate(PARAMETER_SENSITIVITY_TASKS, start=1):
            child_result_dir = os.path.join(results_root, material_name, f"task-{task_id}")
            child_cmd = [
                sys.executable,
                os.path.abspath(__file__),
                *base_child_args,
                "--continual_training",
                "--model_path",
                early_args.model_path,
                "--colon_id",
                "c1",
                "--task_id",
                task_id,
                "--youngs_modulus",
                str(youngs_modulus_mpa),
                "--poisson_ratio",
                str(poisson_ratio),
                "--result_dir",
                child_result_dir,
                "--stop_after_episodes",
                str(early_args.parameter_sensitivity_episodes),
                "--continual_training_target_episodes",
                str(per_env_target_episodes),
            ]

            logging.info(
                "Parameter sensitivity setting %s/%s task %s/%s: E=%s MPa, nu=%s, task=%s",
                setting_index,
                len(PARAMETER_SENSITIVITY_SETTINGS),
                task_index,
                len(PARAMETER_SENSITIVITY_TASKS),
                youngs_modulus_mpa,
                poisson_ratio,
                task_id,
            )
            logging.info("Command: %s", " ".join(child_cmd))
            completed = subprocess.run(child_cmd)

            run_status = _load_json_if_exists(os.path.join(child_result_dir, "run_status.json"))
            metrics = _load_json_if_exists(os.path.join(child_result_dir, "training_metrics_summary.json"))
            final_metrics = {
                key.removeprefix("final_"): value
                for key, value in metrics.items()
                if key.startswith("final_")
            }
            row = {
                "youngs_modulus_mpa": youngs_modulus_mpa,
                "poisson_ratio": poisson_ratio,
                "task_id": task_id,
                "result_dir": child_result_dir,
                "status": run_status.get("status", "failed" if completed.returncode else "unknown"),
                "total_episodes": metrics.get("total_episodes"),
                "success_rate": metrics.get("success_rate"),
                "raw_mean_step_reward": final_metrics.get("raw_mean_step_reward"),
                "normalized_mean_step_reward": final_metrics.get("normalized_mean_step_reward"),
                "mean_s_c": final_metrics.get("mean_s_c"),
                "mean_s_1": final_metrics.get("mean_s_1"),
                "mean_s_2": final_metrics.get("mean_s_2"),
                "mean_s_3": final_metrics.get("mean_s_3"),
                "mean_s_o": final_metrics.get("mean_s_o"),
                "normalized_progress": final_metrics.get("normalized_progress"),
                "roi_alignment_rate": final_metrics.get("roi_alignment_rate"),
                "lumen_visible_ratio": final_metrics.get("lumen_visible_ratio"),
            }
            summary_rows.append(row)
            _write_parameter_sensitivity_summary(summary_rows, summary_csv, summary_json)

            if completed.returncode != 0:
                logging.error(
                    "Parameter sensitivity run failed for task=%s E=%s MPa nu=%s with exit code %s",
                    task_id,
                    youngs_modulus_mpa,
                    poisson_ratio,
                    completed.returncode,
                )
                sys.exit(completed.returncode)

    logging.info("Parameter sensitivity summary saved to: %s", summary_csv)
    sys.exit(0)


def _run_anchor_sensitivity_launcher_if_requested():
    """
    Launch anchor sensitivity runs before importing Isaac/Kit.

    Each child process loads the same checkpoint with --continual_training and
    applies one anchor setting to all vectorized environments. The sequence is
    sparse task 1-4, default task 1-4, then dense task 1-4.
    """
    early_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    early_parser.add_argument("--anchor_sensitivity_test", action="store_true")
    early_parser.add_argument("--model_path", type=str, default=None)
    early_parser.add_argument("--seed", type=int, default=0)
    early_parser.add_argument("--num_envs", type=int, default=2)
    early_parser.add_argument("--test_episodes", type=int, default=1)
    early_parser.add_argument("--anchor_sensitivity_results_dir", type=str, default=None)
    early_args, _ = early_parser.parse_known_args()

    if not early_args.anchor_sensitivity_test:
        return
    if early_args.model_path is None:
        raise SystemExit("--model_path is required for --anchor_sensitivity_test")
    try:
        early_args.model_path = _resolve_model_path(early_args.model_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = early_args.anchor_sensitivity_results_dir or os.path.join(
        "anchor_sensitivity_results",
        f"anchor_sensitivity_colon-c1_seed-{early_args.seed}_{timestamp}",
    )
    os.makedirs(results_root, exist_ok=True)

    value_options = {
        "--anchor_sensitivity_results_dir",
        "--model_path",
        "--anchor_setting",
        "--result_dir",
        "--colon_id",
        "--task_id",
        "--stop_after_episodes",
        "--continual_training_target_episodes",
    }
    flag_options = {
        "--anchor_sensitivity_test",
        "--train",
        "--test",
        "--continual_training",
        "--data_sampling",
    }
    base_child_args = _strip_controlled_cli_args(sys.argv[1:], value_options, flag_options)
    summary_rows = []
    summary_csv = os.path.join(results_root, "anchor_sensitivity_summary.csv")
    summary_json = os.path.join(results_root, "anchor_sensitivity_summary.json")
    per_env_target_episodes = max(
        1,
        int(np.ceil(early_args.test_episodes / max(1, early_args.num_envs))),
    )

    for setting_index, anchor_setting in enumerate(ANCHOR_SENSITIVITY_SETTINGS, start=1):
        for task_index, task_id in enumerate(ANCHOR_SENSITIVITY_TASKS, start=1):
            child_result_dir = os.path.join(results_root, anchor_setting, f"task-{task_id}")
            child_cmd = [
                sys.executable,
                os.path.abspath(__file__),
                *base_child_args,
                "--continual_training",
                "--model_path",
                early_args.model_path,
                "--colon_id",
                "c1",
                "--task_id",
                task_id,
                "--anchor_setting",
                anchor_setting,
                "--result_dir",
                child_result_dir,
                "--stop_after_episodes",
                str(early_args.test_episodes),
                "--continual_training_target_episodes",
                str(per_env_target_episodes),
            ]

            logging.info(
                "Anchor sensitivity setting %s/%s task %s/%s: anchors=%s, task=%s",
                setting_index,
                len(ANCHOR_SENSITIVITY_SETTINGS),
                task_index,
                len(ANCHOR_SENSITIVITY_TASKS),
                anchor_setting,
                task_id,
            )
            logging.info("Command: %s", " ".join(child_cmd))
            completed = subprocess.run(child_cmd)

            run_status = _load_json_if_exists(os.path.join(child_result_dir, "run_status.json"))
            metrics = _load_json_if_exists(os.path.join(child_result_dir, "training_metrics_summary.json"))
            final_metrics = {
                key.removeprefix("final_"): value
                for key, value in metrics.items()
                if key.startswith("final_")
            }
            row = {
                "anchor_setting": anchor_setting,
                "task_id": task_id,
                "result_dir": child_result_dir,
                "status": run_status.get("status", "failed" if completed.returncode else "unknown"),
                "total_episodes": metrics.get("total_episodes"),
                "success_rate": metrics.get("success_rate"),
                "raw_mean_step_reward": final_metrics.get("raw_mean_step_reward"),
                "normalized_mean_step_reward": final_metrics.get("normalized_mean_step_reward"),
                "mean_s_c": final_metrics.get("mean_s_c"),
                "mean_s_1": final_metrics.get("mean_s_1"),
                "mean_s_2": final_metrics.get("mean_s_2"),
                "mean_s_3": final_metrics.get("mean_s_3"),
                "mean_s_o": final_metrics.get("mean_s_o"),
                "normalized_progress": final_metrics.get("normalized_progress"),
                "roi_alignment_rate": final_metrics.get("roi_alignment_rate"),
                "lumen_visible_ratio": final_metrics.get("lumen_visible_ratio"),
            }
            summary_rows.append(row)
            _write_anchor_sensitivity_summary(summary_rows, summary_csv, summary_json)

            if completed.returncode != 0:
                logging.error(
                    "Anchor sensitivity run failed for task=%s anchor_setting=%s with exit code %s",
                    task_id,
                    anchor_setting,
                    completed.returncode,
                )
                sys.exit(completed.returncode)

    logging.info("Anchor sensitivity summary saved to: %s", summary_csv)
    sys.exit(0)


def _run_cross_colon2_launcher_if_requested():
    """
    Launch a cross-colon transfer sequence on Colon 2 before importing Isaac/Kit.

    The same continual-training checkpoint is loaded for each child run. The
    sequence is c2t1 twice, c2t2 twice, c2t3 twice, and c2t4 twice, each with
    five parallel environments.
    """
    early_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    early_parser.add_argument("--cross_colon2", action="store_true")
    early_parser.add_argument("--model_path", type=str, default=None)
    early_parser.add_argument("--seed", type=int, default=0)
    early_parser.add_argument("--cross_colon_trials", type=int, default=CROSS_COLON2_TRIALS)
    early_parser.add_argument("--cross_colon_results_dir", type=str, default=None)
    early_args, _ = early_parser.parse_known_args()

    if not early_args.cross_colon2:
        return
    if early_args.model_path is None:
        raise SystemExit("--model_path is required for --cross_colon2")
    try:
        early_args.model_path = _resolve_model_path(early_args.model_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = early_args.cross_colon_results_dir or os.path.join(
        "cross_colon_transfer_results",
        f"cross_colon2_seed-{early_args.seed}_{timestamp}",
    )
    os.makedirs(results_root, exist_ok=True)

    value_options = {
        "--cross_colon_trials",
        "--cross_colon_results_dir",
        "--model_path",
        "--result_dir",
        "--colon_id",
        "--task_id",
        "--num_envs",
    }
    flag_options = {
        "--cross_colon2",
        "--train",
        "--test",
        "--continual_training",
        "--continual_learning",
        "--data_sampling",
    }
    base_child_args = _strip_controlled_cli_args(sys.argv[1:], value_options, flag_options)
    summary_rows = []
    summary_csv = os.path.join(results_root, "cross_colon2_summary.csv")
    summary_json = os.path.join(results_root, "cross_colon2_summary.json")

    for task_id in CROSS_COLON2_TASKS:
        for trial_id in range(1, int(early_args.cross_colon_trials) + 1):
            child_result_dir = os.path.join(results_root, f"task-{task_id}", f"trial-{trial_id}")
            child_cmd = [
                sys.executable,
                os.path.abspath(__file__),
                *base_child_args,
                "--continual_training",
                "--model_path",
                early_args.model_path,
                "--num_envs",
                "5",
                "--colon_id",
                "c2",
                "--task_id",
                task_id,
                "--result_dir",
                child_result_dir,
            ]

            logging.info(
                "Cross-colon2 transfer: task=%s trial=%s/%s num_envs=5",
                task_id,
                trial_id,
                early_args.cross_colon_trials,
            )
            logging.info("Command: %s", " ".join(child_cmd))
            completed = subprocess.run(child_cmd)

            run_status = _load_json_if_exists(os.path.join(child_result_dir, "run_status.json"))
            metrics = _load_json_if_exists(os.path.join(child_result_dir, "training_metrics_summary.json"))
            final_metrics = {
                key.removeprefix("final_"): value
                for key, value in metrics.items()
                if key.startswith("final_")
            }
            row = {
                "colon_id": "c2",
                "task_id": task_id,
                "trial_id": trial_id,
                "result_dir": child_result_dir,
                "status": run_status.get("status", "failed" if completed.returncode else "unknown"),
                "total_episodes": metrics.get("total_episodes"),
                "success_rate": metrics.get("success_rate"),
                "raw_mean_step_reward": final_metrics.get("raw_mean_step_reward"),
                "normalized_mean_step_reward": final_metrics.get("normalized_mean_step_reward"),
                "mean_s_c": final_metrics.get("mean_s_c"),
                "mean_s_1": final_metrics.get("mean_s_1"),
                "mean_s_2": final_metrics.get("mean_s_2"),
                "mean_s_3": final_metrics.get("mean_s_3"),
                "mean_s_o": final_metrics.get("mean_s_o"),
                "normalized_progress": final_metrics.get("normalized_progress"),
                "roi_alignment_rate": final_metrics.get("roi_alignment_rate"),
                "lumen_visible_ratio": final_metrics.get("lumen_visible_ratio"),
            }
            summary_rows.append(row)
            _write_cross_colon_summary(summary_rows, summary_csv, summary_json)

            if completed.returncode != 0:
                logging.error(
                    "Cross-colon2 transfer run failed for task=%s trial=%s with exit code %s",
                    task_id,
                    trial_id,
                    completed.returncode,
                )
                sys.exit(completed.returncode)

    logging.info("Cross-colon2 transfer summary saved to: %s", summary_csv)
    sys.exit(0)


def _run_cross_colon_launcher_if_requested():
    """
    Launch cross-colon transfer evaluations before importing Isaac/Kit.

    Each child process loads the same checkpoint through --continual_training
    with constrained actions. For each colon/task, the launcher runs repeated
    trials; each trial uses five parallel copies of the same colon and stops
    after each environment completes the requested number of episodes.
    """
    early_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    early_parser.add_argument("--cross_colon", action="store_true")
    early_parser.add_argument("--model_path", type=str, default=None)
    early_parser.add_argument("--seed", type=int, default=0)
    early_parser.add_argument("--cross_colon_episodes", type=int, default=1)
    early_parser.add_argument("--cross_colon_trials", type=int, default=CROSS_COLON2_TRIALS)
    early_parser.add_argument("--cross_colon_results_dir", type=str, default=None)
    early_args, _ = early_parser.parse_known_args()

    if not early_args.cross_colon:
        return
    if early_args.model_path is None:
        raise SystemExit("--model_path is required for --cross_colon")
    try:
        early_args.model_path = _resolve_model_path(early_args.model_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = early_args.cross_colon_results_dir or os.path.join(
        "cross_colon_transfer_results",
        f"cross_colon_seed-{early_args.seed}_{timestamp}",
    )
    os.makedirs(results_root, exist_ok=True)

    value_options = {
        "--cross_colon_episodes",
        "--cross_colon_trials",
        "--cross_colon_results_dir",
        "--model_path",
        "--result_dir",
        "--colon_id",
        "--task_id",
        "--num_envs",
        "--test_episodes",
        "--stop_after_episodes",
        "--continual_training_target_episodes",
    }
    flag_options = {
        "--cross_colon",
        "--cross_colon_eval_only",
        "--clip_actions",
        "--no-clip_actions",
        "--train",
        "--test",
        "--continual_training",
        "--continual_learning",
        "--data_sampling",
    }
    base_child_args = _strip_controlled_cli_args(sys.argv[1:], value_options, flag_options)
    summary_rows = []
    aggregate_rows = []
    summary_csv = os.path.join(results_root, "cross_colon_summary.csv")
    summary_json = os.path.join(results_root, "cross_colon_summary.json")
    aggregate_csv = os.path.join(results_root, "cross_colon_task_summary.csv")
    aggregate_json = os.path.join(results_root, "cross_colon_task_summary.json")
    launched_num_envs = 5
    per_env_target_episodes = max(1, int(early_args.cross_colon_episodes))
    expected_episodes_per_trial = launched_num_envs * per_env_target_episodes

    for colon_id in CROSS_COLON_IDS:
        for task_id in CROSS_COLON_TASKS:
            task_rows = []
            for trial_id in range(1, int(early_args.cross_colon_trials) + 1):
                child_result_dir = os.path.join(
                    results_root,
                    f"colon-{colon_id}",
                    f"task-{task_id}",
                    f"trial-{trial_id}",
                )
                child_cmd = [
                    sys.executable,
                    os.path.abspath(__file__),
                    *base_child_args,
                    "--continual_training",
                    "--clip_actions",
                    "--model_path",
                    early_args.model_path,
                    "--num_envs",
                    str(launched_num_envs),
                    "--colon_id",
                    colon_id,
                    "--task_id",
                    task_id,
                    "--result_dir",
                    child_result_dir,
                    "--continual_training_target_episodes",
                    str(per_env_target_episodes),
                ]

                logging.info(
                    "Cross-colon transfer: colon=%s task=%s trial=%s/%s "
                    "episodes_per_env=%s expected_total=%s constrained=True",
                    colon_id,
                    task_id,
                    trial_id,
                    early_args.cross_colon_trials,
                    per_env_target_episodes,
                    expected_episodes_per_trial,
                )
                logging.info("Command: %s", " ".join(child_cmd))
                completed = subprocess.run(child_cmd)

                run_status = _load_json_if_exists(os.path.join(child_result_dir, "run_status.json"))
                metrics = _load_json_if_exists(os.path.join(child_result_dir, "training_metrics_summary.json"))
                final_metrics = {
                    key.removeprefix("final_"): value
                    for key, value in metrics.items()
                    if key.startswith("final_")
                }
                total_episodes = metrics.get("total_episodes")
                total_successes = metrics.get("total_successes")
                success_rate = metrics.get("success_rate")
                row = {
                    "row_type": "trial",
                    "colon_id": colon_id,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "result_dir": child_result_dir,
                    "status": run_status.get("status", "failed" if completed.returncode else "unknown"),
                    "expected_episodes": expected_episodes_per_trial,
                    "total_episodes": total_episodes,
                    "total_successes": total_successes,
                    "success_rate": success_rate,
                    "raw_mean_step_reward": final_metrics.get("raw_mean_step_reward"),
                    "normalized_mean_step_reward": final_metrics.get("normalized_mean_step_reward"),
                    "mean_s_c": final_metrics.get("mean_s_c"),
                    "mean_s_1": final_metrics.get("mean_s_1"),
                    "mean_s_2": final_metrics.get("mean_s_2"),
                    "mean_s_3": final_metrics.get("mean_s_3"),
                    "mean_s_o": final_metrics.get("mean_s_o"),
                    "normalized_progress": final_metrics.get("normalized_progress"),
                    "roi_alignment_rate": final_metrics.get("roi_alignment_rate"),
                    "lumen_visible_ratio": final_metrics.get("lumen_visible_ratio"),
                }
                summary_rows.append(row)
                task_rows.append(row)
                _write_cross_colon_summary(summary_rows, summary_csv, summary_json)

                if completed.returncode != 0:
                    logging.error(
                        "Cross-colon transfer run failed for colon=%s task=%s trial=%s with exit code %s",
                        colon_id,
                        task_id,
                        trial_id,
                        completed.returncode,
                    )
                    sys.exit(completed.returncode)

            aggregate_episodes = sum(int(row.get("total_episodes") or 0) for row in task_rows)
            aggregate_successes = sum(int(row.get("total_successes") or 0) for row in task_rows)
            aggregate_row = {
                "row_type": "aggregate",
                "colon_id": colon_id,
                "task_id": task_id,
                "trial_id": "all",
                "result_dir": os.path.join(results_root, f"colon-{colon_id}", f"task-{task_id}"),
                "status": "completed",
                "expected_episodes": expected_episodes_per_trial * int(early_args.cross_colon_trials),
                "total_episodes": aggregate_episodes,
                "total_successes": aggregate_successes,
                "success_rate": aggregate_successes / aggregate_episodes if aggregate_episodes else 0.0,
                "raw_mean_step_reward": _weighted_mean(task_rows, "raw_mean_step_reward"),
                "normalized_mean_step_reward": _weighted_mean(task_rows, "normalized_mean_step_reward"),
                "mean_s_c": _weighted_mean(task_rows, "mean_s_c"),
                "mean_s_1": _weighted_mean(task_rows, "mean_s_1"),
                "mean_s_2": _weighted_mean(task_rows, "mean_s_2"),
                "mean_s_3": _weighted_mean(task_rows, "mean_s_3"),
                "mean_s_o": _weighted_mean(task_rows, "mean_s_o"),
                "normalized_progress": _weighted_mean(task_rows, "normalized_progress"),
                "roi_alignment_rate": _weighted_mean(task_rows, "roi_alignment_rate"),
                "lumen_visible_ratio": _weighted_mean(task_rows, "lumen_visible_ratio"),
            }
            aggregate_rows.append(aggregate_row)
            _write_cross_colon_summary(aggregate_rows, aggregate_csv, aggregate_json)

    logging.info("Cross-colon transfer summary saved to: %s", summary_csv)
    logging.info("Cross-colon aggregate summary saved to: %s", aggregate_csv)
    sys.exit(0)


def _run_data_sampling_sequence_launcher_if_requested():
    """
    Launch the requested data-sampling sequence before importing Isaac/Kit.

    Colon assets are constructed at app startup, so the robust way to "switch"
    colons is to run one child Isaac process per sampling trial.
    """
    early_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    early_parser.add_argument("--data_sampling_sequence", action="store_true")
    early_parser.add_argument("--data_sampling", action="store_true")
    early_parser.add_argument("--continual_training", action="store_true")
    early_parser.add_argument("--continual_learning", action="store_true")
    early_parser.add_argument("--model_path", type=str, default=None)
    early_parser.add_argument("--seed", type=int, default=0)
    early_parser.add_argument("--num_envs", type=int, default=2)
    early_parser.add_argument("--task_id", type=str, default="t1")
    early_parser.add_argument("--data_sampling_sequence_repeats", type=int, default=DATA_SAMPLING_SEQUENCE_REPEATS)
    early_parser.add_argument("--data_sampling_sequence_colons", type=str, default=",".join(DATA_SAMPLING_SEQUENCE_COLONS))
    early_parser.add_argument("--data_sampling_sequence_results_dir", type=str, default=None)
    early_parser.add_argument("--medical_data_dir", type=str, default=DEFAULT_MEDICAL_DATA_DIR)
    early_args, _ = early_parser.parse_known_args()

    if not early_args.data_sampling_sequence:
        return
    if not early_args.data_sampling:
        raise SystemExit("--data_sampling_sequence requires --data_sampling")
    if early_args.continual_learning:
        early_args.continual_training = True
    if early_args.continual_training and early_args.model_path is None:
        raise SystemExit("--model_path is required for --data_sampling_sequence with --continual_training")
    if early_args.model_path is not None:
        try:
            early_args.model_path = _resolve_model_path(early_args.model_path)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        sequence_colons = _parse_data_sampling_sequence_colons(early_args.data_sampling_sequence_colons)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = early_args.data_sampling_sequence_results_dir or os.path.join(
        "data_sampling_sequence_results",
        f"sampling_sequence_seed-{early_args.seed}_{timestamp}",
    )
    os.makedirs(results_root, exist_ok=True)
    sequence_name = os.path.basename(os.path.normpath(results_root))
    safe_sequence_name = "".join(
        ch if ch.isalnum() or ch in "._-" else "-"
        for ch in sequence_name
    ) or f"sampling_sequence_seed-{early_args.seed}_{timestamp}"
    medical_sequence_root = os.path.join(
        os.path.abspath(os.path.expanduser(early_args.medical_data_dir)),
        safe_sequence_name,
    )
    os.makedirs(medical_sequence_root, exist_ok=True)

    value_options = {
        "--data_sampling_sequence_repeats",
        "--data_sampling_sequence_colons",
        "--data_sampling_sequence_results_dir",
        "--data_sampling_target_reaches",
        "--medical_data_dir",
        "--medical_run_dir",
        "--model_path",
        "--result_dir",
        "--colon_id",
        "--task_id",
        "--num_envs",
    }
    flag_options = {
        "--data_sampling_sequence",
        "--train",
        "--test",
        "--data_sampling",
        "--continual_training",
        "--continual_learning",
    }
    base_child_args = _strip_controlled_cli_args(sys.argv[1:], value_options, flag_options)
    summary_rows = []
    summary_csv = os.path.join(results_root, "data_sampling_sequence_summary.csv")
    summary_json = os.path.join(results_root, "data_sampling_sequence_summary.json")

    repeats = max(1, int(early_args.data_sampling_sequence_repeats))
    for colon_id in sequence_colons:
        colon_suffix = colon_id[1:] if colon_id.lower().startswith("c") else colon_id
        for trial_id in range(1, repeats + 1):
            child_result_dir = os.path.join(results_root, f"colon-{colon_id}", f"trial-{trial_id}")
            child_medical_run_dir = os.path.join(
                medical_sequence_root,
                f"sampling_colon{colon_suffix}",
                f"trial-{trial_id}",
            )
            child_cmd = [
                sys.executable,
                os.path.abspath(__file__),
                *base_child_args,
                "--data_sampling",
                "--colon_id",
                colon_id,
                "--task_id",
                early_args.task_id,
                "--num_envs",
                str(early_args.num_envs),
                "--result_dir",
                child_result_dir,
                "--data_sampling_target_reaches",
                "1",
                "--medical_data_dir",
                early_args.medical_data_dir,
                "--medical_run_dir",
                child_medical_run_dir,
            ]
            if early_args.continual_training:
                child_cmd.append("--continual_training")
            if early_args.model_path is not None:
                child_cmd.extend(["--model_path", early_args.model_path])

            logging.info(
                "Data-sampling sequence: colon=%s trial=%s/%s target=sampling_colon%s",
                colon_id,
                trial_id,
                repeats,
                colon_suffix,
            )
            logging.info("Command: %s", " ".join(child_cmd))
            completed = subprocess.run(child_cmd)

            run_status = _load_json_if_exists(os.path.join(child_result_dir, "run_status.json"))
            child_status = run_status.get("status", "failed" if completed.returncode else "unknown")
            row = {
                "colon_id": colon_id,
                "trial_id": trial_id,
                "result_dir": child_result_dir,
                "medical_data_dir": run_status.get("medical_data_dir", child_medical_run_dir),
                "init_from_csv": run_status.get("init_from_csv"),
                "init_endpose_from_csv": run_status.get("init_endpose_from_csv"),
                "status": child_status,
                "target_reaches": run_status.get("target_reaches"),
                "total_env_steps": run_status.get("total_env_steps"),
                "model_path": run_status.get("model_path", early_args.model_path),
            }
            summary_rows.append(row)
            _write_data_sampling_sequence_summary(summary_rows, summary_csv, summary_json)

            if completed.returncode != 0:
                logging.error(
                    "Data-sampling sequence child failed for colon=%s trial=%s with exit code %s",
                    colon_id,
                    trial_id,
                    completed.returncode,
                )
                sys.exit(completed.returncode)
            if child_status != "completed":
                logging.error(
                    "Data-sampling sequence child did not complete target reach for colon=%s trial=%s: status=%s",
                    colon_id,
                    trial_id,
                    child_status,
                )
                sys.exit(1)

    logging.info("Data-sampling sequence summary saved to: %s", summary_csv)
    sys.exit(0)


_run_reward_ablation_launcher_if_requested()
_run_parameter_sensitivity_launcher_if_requested()
_run_anchor_sensitivity_launcher_if_requested()
_run_cross_colon2_launcher_if_requested()
_run_cross_colon_launcher_if_requested()
_run_data_sampling_sequence_launcher_if_requested()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Create an application to launch ARC environment")
parser.add_argument("--train", action="store_true", help="Run in training mode")
parser.add_argument("--test", action="store_true", help="Run in test mode (load and evaluate a trained model)")
parser.add_argument("--continual_training", action="store_true",
                    help="Continue training from a pre-trained model (for transfer learning to other colons)")
parser.add_argument("--continual_learning", action="store_true",
                    help="Alias for --continual_training")
parser.add_argument("--data_sampling", action="store_true",
                    help="Run autonomous data-sampling mode. By default it uses random actions; with --model_path or --continual_training it uses deterministic policy actions without training updates.")
parser.add_argument("--data_sampling_sequence", action="store_true",
                    help="Run data sampling in sequence: c1 twice, c3 twice, c4 twice, c5 twice, skipping c2")
parser.add_argument("--data_sampling_sequence_repeats", type=int, default=DATA_SAMPLING_SEQUENCE_REPEATS,
                    help="Number of successful sampling trials per colon in --data_sampling_sequence")
parser.add_argument("--data_sampling_sequence_colons", type=str, default=",".join(DATA_SAMPLING_SEQUENCE_COLONS),
                    help="Comma-separated colon ids for --data_sampling_sequence, e.g. c5 or c1,c3,c4,c5")
parser.add_argument("--data_sampling_sequence_results_dir", type=str, default=None,
                    help="Output root for --data_sampling_sequence child runs")
parser.add_argument("--data_sampling_target_reaches", type=int, default=None,
                    help="Stop this data-sampling run after this many fresh target-reached events")
parser.add_argument("--model_path", type=str, default=None,
                    help="Path to the trained model to load for testing, continual training, or policy-guided data sampling")
parser.add_argument("--test_episodes", type=int, default=1, help="Number of episodes to run in test mode")
parser.add_argument("--print_actions", action="store_true",
                    help="Print and save deterministic policy actions during evaluation")
parser.add_argument("--print_action_steps", type=int, default=50,
                    help="Maximum number of vectorized environment steps to print when --print_actions is enabled")
parser.add_argument("--sampling_deterministic_actions", action=argparse.BooleanOptionalAction, default=False,
                    help="Use deterministic policy actions in policy-guided data sampling. Default is false to match continual-training rollout action selection.")
parser.add_argument("--benchmark", action="store_true", help="benchmark a couple of steps using viztracer")
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments")
parser.add_argument("--clip_actions", action=argparse.BooleanOptionalAction, default=None,
                    help="Enable action constraints (clipping/projection/masks)")
parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC", "TD3", "A2C", "DDPG"],
                    help="RL algorithm to use (default: PPO)")
parser.add_argument("--reward_variant", type=str, default=None,
                    choices=["center_only", "depth_existence", "deep_area", "lumen_evidence", "full_reward"],
                    help="Reward ablation variant. Omit to preserve the legacy full reward behavior.")
parser.add_argument("--reward_wc", type=float, default=0.5,
                    help="Center-alignment weight for full_reward ablations")
parser.add_argument("--reward_wo", type=float, default=0.5,
                    help="Lumen-visibility weight for full_reward ablations")
parser.add_argument("--reward_lambda_o", type=float, default=1.0,
                    help="Obstacle/lumen cue multiplier for partial reward ablations")
parser.add_argument("--reward_beta", type=float, default=0.6,
                    help="Weight on s_1 in lumen evidence and full lumen visibility")
parser.add_argument("--train_with_normalized_reward", action="store_true",
                    help="Use normalized reward for policy training instead of only logging it")
parser.add_argument("--seed", type=int, default=0, help="Random seed recorded with the run")
parser.add_argument("--colon_id", type=str, default="c1",
                    help="Colon id. Selects the colon mesh and saved_states/{colon_id}{task_id}_start.csv/_end.csv when present.")
parser.add_argument("--task_id", type=str, default="t1",
                    help="Task id metadata. Also selects saved_states/{colon_id}{task_id}_start.csv and _end.csv when present.")
parser.add_argument("--eval_after_train", action="store_true",
                    help="Run deterministic evaluation after training and write evaluation_metrics.json")
parser.add_argument("--total_timesteps", type=float, default=1e7,
                    help="Total SB3 training timesteps")
parser.add_argument("--ablation_t1", action="store_true",
                    help="Run the six reward-ablation trainings for task 1, stopping each after 300 completed episodes")
parser.add_argument("--ablation_t2", action="store_true",
                    help="Run the six reward-ablation trainings for task 2, stopping each after 300 completed episodes")
parser.add_argument("--ablation_t3", action="store_true",
                    help="Run the six reward-ablation trainings for task 3, stopping each after 300 completed episodes")
parser.add_argument("--ablation_t4", action="store_true",
                    help="Run the six reward-ablation trainings for task 4, stopping each after 300 completed episodes")
parser.add_argument("--ablation_episodes", type=int, default=300,
                    help="Number of completed episodes to train each reward variant in --ablation_t* mode")
parser.add_argument("--stop_after_episodes", type=int, default=None,
                    help="Stop a training run after this many completed episodes across all environments")
parser.add_argument("--continual_training_target_episodes", type=int, default=1,
                    help="Number of completed episodes per environment before continual-training child runs stop")
parser.add_argument("--parameter_sensitivity_test", action="store_true",
                    help="Run colon material parameter sensitivity over 6 settings and tasks t1-t4 using colon c1")
parser.add_argument("--parameter_sensitivity_episodes", type=int, default=300,
                    help="Total completed episodes for each parameter sensitivity child run")
parser.add_argument("--parameter_sensitivity_results_dir", type=str, default=None,
                    help="Output root for --parameter_sensitivity_test")
parser.add_argument("--anchor_sensitivity_test", action="store_true",
                    help="Run colon anchor sensitivity over sparse/default/dense anchor settings and tasks t1-t4 using colon c1")
parser.add_argument("--anchor_sensitivity_results_dir", type=str, default=None,
                    help="Output root for --anchor_sensitivity_test")
parser.add_argument("--anchor_setting", type=str, default="default", choices=["sparse", "default", "dense"],
                    help="Colon anchor-point setting: sparse, default, or dense")
parser.add_argument("--cross_colon2", action="store_true",
                    help="Run cross-colon transfer on Colon 2: tasks t1-t4, two trials each, with 5 parallel envs")
parser.add_argument("--cross_colon", action="store_true",
                    help="Run cross-colon transfer evaluation on Colons 2-5 and Tasks t1-t4 with 5 parallel envs")
parser.add_argument("--cross_colon_eval_only", action="store_true",
                    help=argparse.SUPPRESS)
parser.add_argument("--cross_colon_episodes", type=int, default=1,
                    help="Completed episodes per parallel environment in each --cross_colon trial")
parser.add_argument("--cross_colon_trials", type=int, default=CROSS_COLON2_TRIALS,
                    help="Number of repeated child trials per colon/task for cross-colon transfer")
parser.add_argument("--cross_colon_results_dir", type=str, default=None,
                    help="Output root for cross-colon transfer launchers")
parser.add_argument("--youngs_modulus", type=float, default=None,
                    help="Optional colon deformable Young's modulus override in MPa")
parser.add_argument("--poisson_ratio", type=float, default=None,
                    help="Optional colon deformable Poisson ratio override")
parser.add_argument("--result_dir", type=str, default=None,
                    help="Override the automatically generated result directory")
parser.add_argument("--disable_video", action="store_true",
                    help="Disable RecordVideo output during training/evaluation")
parser.add_argument("--trajectory_save_interval", type=int, default=1,
                    help="Save trajectory metadata every N completed episodes per environment")
parser.add_argument("--save_trajectory_images", action=argparse.BooleanOptionalAction, default=True,
                    help="Save per-frame camera PNGs in trajectory_data")
parser.add_argument("--medical_data_dir", type=str, default=DEFAULT_MEDICAL_DATA_DIR,
                    help="Root directory for medical-simulator data saved during --data_sampling")
parser.add_argument("--medical_run_dir", type=str, default=None,
                    help="Exact run directory for medical-simulator data saved during --data_sampling")
parser.add_argument("--disable_medical_data_save", action="store_true",
                    help="Disable medical-simulator dataset writing during --data_sampling")
parser.add_argument("--medical_save_images", action=argparse.BooleanOptionalAction, default=True,
                    help="Save RGB PNG frames for the medical-simulator dataset during --data_sampling")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli = parser.parse_args()
if args_cli.continual_learning:
    args_cli.continual_training = True

# --data_sampling can be combined with --continual_training to reuse the
# checkpoint-loading/policy-action path without entering model.learn().
continual_training_mode = args_cli.continual_training and not args_cli.data_sampling
policy_guided_data_sampling = args_cli.data_sampling and (
    args_cli.continual_training or args_cli.model_path is not None
)
random_data_sampling = args_cli.data_sampling and not policy_guided_data_sampling
continual_training_action_mode = continual_training_mode or policy_guided_data_sampling
policy_sampling_deterministic = bool(args_cli.sampling_deterministic_actions)
colon_id_normalized = str(args_cli.colon_id).lower()
data_sampling_colon1_forward_only = (
    args_cli.data_sampling and colon_id_normalized in {"c1", "colon1", "1"}
)
data_sampling_no_backward_recovery = (
    args_cli.data_sampling and colon_id_normalized in {"c1", "colon1", "1", "c5", "colon5", "5"}
)

if args_cli.clip_actions is None:
    args_cli.clip_actions = args_cli.train or args_cli.test or args_cli.continual_training or args_cli.data_sampling

active_modes = sum([args_cli.train, args_cli.test, continual_training_mode, args_cli.data_sampling])
if active_modes > 1:
    parser.error(
        "Only one of --train, --test, --continual_training, or --data_sampling can be enabled at a time. "
        "The supported exception is --data_sampling --continual_training --model_path, which runs "
        "policy-guided sampling without training updates."
    )

# Validate test mode arguments
if args_cli.test and args_cli.model_path is None:
    parser.error("--model_path is required when using --test mode")

# Validate continual training mode arguments
if continual_training_mode and args_cli.model_path is None:
    parser.error("--model_path is required when using --continual_training mode")
if args_cli.data_sampling and args_cli.continual_training and args_cli.model_path is None:
    parser.error("--model_path is required when using --data_sampling with --continual_training")
if args_cli.model_path is not None:
    try:
        args_cli.model_path = _resolve_model_path(args_cli.model_path)
    except FileNotFoundError as exc:
        parser.error(str(exc))

random.seed(args_cli.seed)
np.random.seed(args_cli.seed)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
algo_name = args_cli.algo.lower()
reward_variant_label = args_cli.reward_variant or "legacy_full_reward"
constraint_label = "constrained" if args_cli.clip_actions else "nonconstrained"
run_name = (
    f"{algo_name}_{reward_variant_label}_{constraint_label}_"
    f"colon-{args_cli.colon_id}_task-{args_cli.task_id}_seed-{args_cli.seed}_{timestamp}"
)
result_dir = args_cli.result_dir or os.path.join("./reward_ablation_results", run_name)
log_dir = os.path.join(result_dir, "logs")
tensorboard_log = os.path.join(result_dir, "tensorboard")
model_save_path = os.path.join(result_dir, "checkpoints")
video_save_path = os.path.join(tensorboard_log, "videos", "train")
trajectory_save_path = os.path.join(result_dir, "trajectory_data")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(trajectory_save_path, exist_ok=True)

with open(os.path.join(result_dir, "prelaunch_status.json"), "w") as f:
    json.dump(
        {
            "status": "created_before_isaac_app_launch",
            "command": sys.argv,
            "colon_id": args_cli.colon_id,
            "task_id": args_cli.task_id,
            "reward_variant": reward_variant_label,
            "constrained": bool(args_cli.clip_actions),
            "seed": args_cli.seed,
            "timestamp": timestamp,
            "youngs_modulus_mpa": args_cli.youngs_modulus,
            "youngs_modulus_pa": args_cli.youngs_modulus * 1e6 if args_cli.youngs_modulus is not None else None,
            "poisson_ratio": args_cli.poisson_ratio,
            "anchor_setting": args_cli.anchor_setting,
            "continual_training_mode": bool(continual_training_mode),
            "continual_training_action_mode": bool(continual_training_action_mode),
            "policy_guided_data_sampling": bool(policy_guided_data_sampling),
            "random_data_sampling": bool(random_data_sampling),
            "sampling_deterministic_actions": bool(policy_sampling_deterministic),
            "model_path": args_cli.model_path,
            "medical_data_dir": args_cli.medical_data_dir,
            "medical_run_dir": args_cli.medical_run_dir,
            "medical_data_save": bool(args_cli.data_sampling and not args_cli.disable_medical_data_save),
            "data_sampling_colon1_forward_only": bool(data_sampling_colon1_forward_only),
            "data_sampling_no_backward_recovery": bool(data_sampling_no_backward_recovery),
        },
        f,
        indent=2,
    )

# Enable cameras (needed for camera sensors)
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

import gymnasium as gym

from stable_baselines3 import PPO, DQN, SAC, TD3, A2C, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, LogEveryNTimesteps
from stable_baselines3.common.vec_env import VecNormalize

from arcgym.utils.callbacks import (
    FreezeLearningUntilAllEnvsNEpisodesCallback,
    PerEnvRewardCallback,
    RewardAblationMetricsCallback,
    TrajectoryDataSaver,
    SuccessRateDataSaver,
    StopAfterAllEnvsNEpisodesCallback,
    StopAfterTotalEpisodesCallback,
)

from isaaclab.envs import (
    DirectRLEnvCfg,
)

from arcgym.envs.sb3_wrapper import Sb3VecEnvWrapper

from arcgym.envs.arc_isaac_env import make_isaac_env_cfg
from arcgym.envs.framestack import FrameStack
from arcgym.assets.robots.robot_factory import RobotFactory

from gymnasium.envs.registration import register

if args_cli.benchmark:
    from viztracer import VizTracer
    import atexit

    tracer = VizTracer()
    tracer.start()

    # Register cleanup function
    def save_trace():
        tracer.stop()
        tracer.save("trace_results.json")

    atexit.register(save_trace)
else:
    tracer = None

register(
    id="ArcIsaacEnv-v0",
    entry_point="arcgym.envs.arc_isaac_env:ARCIsaacEnv",
    disable_env_checker=True,
    #max_episode_steps = 2500, # This breaks IsaacLab do not set this.
)

use_pose_in_obs = False
use_camera = True

device = args_cli.device

learning_config = {
    "total_timesteps": args_cli.total_timesteps,
    "learning_rate" : 1e-4,
    "batch_size" : 1024,
    "verbose" : True,
}
robot_config = {
    "robot_type" : "magnetic_endoscope", # "capsule" or "soft_endoscope" or "magnetic_endoscope" or "proximal_actuated"
    # Capsule config 
    "capsule_radius" : 0.004,
    "capsule_height" : 0.012,
    "collision_contact_offset" : 0.0001,
    "collision_rest_offset" : 0.0,
    # Soft endoscope specific parameters (from original soft_endoscope.py)
    "num_passive_links" : 20,
    "num_active_links" : 5,
    "num_links_total" : 2,
    "link_radius" : 0.01,
    "link_height" : 0.02,
    "passive_stiffness" : 1,
    "passive_damping" : 1,
    "active_stiffness" : 1e1,
    "active_damping" : 1e2,
    "max_linear_velocity": 10,
    "max_angular_velocity": 10,
    #"link_density" : 0.1,
    # Camera and light configuration
    "camera_resolution" : (84, 84),
    "front_camera_focal_length" : 5.0, 
    "front_camera_focus_distance" : 10.0, 
    "front_camera_horizontal_aperture" : 20, 
    "front_camera_clipping_range" : (0.001, 5.0),
    "front_light_color" : (1.0, 1.0, 1.0),  # Pure white for maximum contrast
    "front_light_color_temperature" : 6500,  # Daylight color temp for natural appearance
    "front_light_intensity" : 30000,  # Very high intensity for bright tissue highlights (was 4000)
    "front_light_radius" : 0.005,  # Smaller point source for stronger falloff/contrast (was 0.01)
    "front_light_exposure" : 6,  # Lower exposure to darken lumen center (was 7)
    "num_segments": 20,
    "joint_stiffness": 1e4,  # Adjust for desired compliance
    "joint_damping": 1e3,
    "joint_friction": 0.1
}
if robot_config["robot_type"] == "capsule":
    env_spacing = 0.5
else:
    env_spacing = 5

def _select_task_pose_csv(colon_id, task_id, suffix, num_envs, use_single_env=True):
    if use_single_env and int(num_envs) == 1:
        single_env_csv = f"./saved_states/env1_{colon_id}{task_id}_{suffix}.csv"
        if os.path.exists(single_env_csv):
            logging.info("Using single-env %s-pose CSV: %s", suffix, single_env_csv)
            return single_env_csv
        logging.warning(
            "Single-env %s-pose CSV does not exist: %s. Falling back to standard colon/task CSV.",
            suffix,
            single_env_csv,
        )

    task_pose_csv = f"./saved_states/{colon_id}{task_id}_{suffix}.csv"
    if os.path.exists(task_pose_csv):
        logging.info("Using %s-pose CSV: %s", suffix, task_pose_csv)
        return task_pose_csv

    fallback_csv = f"./saved_states/c1t1_{suffix}.csv"
    logging.warning(
        "Requested colon/task %s-pose CSV does not exist: %s. Falling back to %s",
        suffix,
        task_pose_csv,
        fallback_csv,
    )
    return fallback_csv


def _select_sampling_endpose_csv(colon_id, task_id, num_envs):
    colon_suffix = colon_id[1:] if colon_id.lower().startswith("c") else colon_id
    sampling_csv = f"./saved_states/sampling_colon{colon_suffix}.csv"
    if os.path.exists(sampling_csv):
        logging.info("Using data-sampling end-pose CSV: %s", sampling_csv)
        return sampling_csv

    logging.warning(
        "Data-sampling end-pose CSV does not exist: %s. Falling back to task end pose.",
        sampling_csv,
    )
    return _select_task_pose_csv(colon_id, task_id, "end", num_envs)


use_task_startpose_csv = args_cli.train or args_cli.test or continual_training_action_mode or random_data_sampling
use_task_endpose_csv = args_cli.train or args_cli.test or continual_training_action_mode
if use_task_startpose_csv:
    task_startpose_csv = _select_task_pose_csv(
        args_cli.colon_id,
        args_cli.task_id,
        "start",
        args_cli.num_envs,
        use_single_env=not args_cli.data_sampling,
    )
else:
    task_startpose_csv = None
if args_cli.data_sampling:
    task_endpose_csv = _select_sampling_endpose_csv(args_cli.colon_id, args_cli.task_id, args_cli.num_envs)
elif use_task_endpose_csv:
    task_endpose_csv = _select_task_pose_csv(args_cli.colon_id, args_cli.task_id, "end", args_cli.num_envs)
else:
    task_endpose_csv = None

env_config = {
    "discrete_action_space" : False, # currently unsupported TODO: Figure out if this is something we want to be determinable from the outside, or if it is a property of the robot implementation.
    "use_pose" : False,
    "use_camera" : True,
    "render_mode" : "rgb_array",
    "env_spacing" : env_spacing,
    "num_envs" : args_cli.num_envs,
    "colon_id": args_cli.colon_id,
    "replicate_physics" : False,
    "action_scale" : 0.001,
    "translation_action_scale" : 0.001,
    "rotation_action_scale" : 0.02 if random_data_sampling else 0.01,
    "forward_increment": 0.0005 if args_cli.data_sampling else 0.001,
    "debug_vis" : False,
    "episode_length_s" : 40.0 if use_task_endpose_csv else 20000000.0,
    "constraint_point_A": 1,  # Distance from robot tip to constraint point A along the robot's local z-axis
    "init_from_csv": task_startpose_csv,
    "init_endpose_from_csv": task_endpose_csv,
    "random_initial_configuration": False,  # Use straight configuration (especially for teleoperation mode)
    "clip_actions": args_cli.clip_actions,
    "disable_movement_constraints": not args_cli.clip_actions,  # Back-compat: tie movement constraints to clip_actions
    "use_txt_files_for_attachments": True,  # Use txt files to load precise vertex indices for colon attachments
    "anchor_setting": args_cli.anchor_setting,
    "disable_lumen_visibility_reset": args_cli.test,  # Disable lumen visibility counter reset in test mode
    "teleoperate_mode": not args_cli.train and not args_cli.test and not continual_training_action_mode and not random_data_sampling,
    "data_sampling_mode": random_data_sampling,
    "data_sampling_policy_actions": policy_guided_data_sampling,
    "data_sampling_random_actions": random_data_sampling,
    "use_continual_training_action_logic": continual_training_action_mode,
    "disable_automatic_resets": args_cli.data_sampling,
    "disable_recovery_mode": data_sampling_no_backward_recovery,
    "highlight_recover_coverage_threshold": 0.9 if args_cli.data_sampling else 0.5,
    "wall_recovery_brightness_threshold": 0.85,
    "wall_recover_consecutive_threshold": 5 if random_data_sampling else 3,
    "lumen_visibility_threshold": -0.55 if random_data_sampling else -0.4,
    "lumen_negative_threshold": 35 if random_data_sampling else 20,
    "backward_increment": 0.0008 if random_data_sampling else 0.001,
    "backward_duration_s": 0.2 if random_data_sampling else 1.0,
    "max_wall_recovery_s": 0.6 if random_data_sampling else 1.0,
    "backward_rotation_scale": 1.0 * 0.01 if random_data_sampling else 0.5 * 0.01,
    "reorient_steps": 3 if random_data_sampling else 12,
    "reorient_rotation_scale": 1.5 * 0.01 if random_data_sampling else 0.01,
    "target_depth_region_fraction": 0.7,
    "colon_init_rot": (0.7071, 0.0, -0.7071, 0.0),
    "robot_init_rot": (0.0, 0.7071, 0.0, 0.7071),
    "robot_init_pos": (5.6430, -1.3124, -2),
}
if args_cli.youngs_modulus is not None:
    env_config["youngs_modulus_mpa"] = args_cli.youngs_modulus
    env_config["youngs_modulus_pa"] = args_cli.youngs_modulus * 1e6
if args_cli.poisson_ratio is not None:
    env_config["poisson_ratio"] = args_cli.poisson_ratio

reward_config = {
    "reward_type" : "final_reward", #"test_reward_action",#"depth_goal",#"default",#,"final_reward"
    "reward_scale" : 1.0,
    "eps" : 0.1,
    "running_penalty" : -0.1,
    "goal_reward" : 50.0,
    # Make center alignment dominant in the reward function
    "center_weight": args_cli.reward_wc,      # Current default is 0.5
    "goal_weight": 0.0,        # Decreased from 0.4 to 0.2
    "obstruction_weight": args_cli.reward_wo, # Current default is 0.5
    "reward_variant": args_cli.reward_variant,
    "reward_wc": args_cli.reward_wc,
    "reward_wo": args_cli.reward_wo,
    "reward_lambda_o": args_cli.reward_lambda_o,
    "reward_beta": args_cli.reward_beta,
    "train_with_normalized_reward": args_cli.train_with_normalized_reward,
    "alignment_threshold": 0.7,
    # Vanilla PPO (--no-clip_actions) resets and penalizes when lumen visibility is low.
    "reset_on_low_lumen_visibility": not args_cli.clip_actions,
    "low_lumen_visibility_threshold": -0.4,
    "low_lumen_reset_penalty": -1.0,
    # Parameters for calculating max_episode_length (used for success criterion)
    "episode_length_s": 40.0 if use_task_endpose_csv else 2000000.0,
    "decimation": 2,
    "dt": 1.0 / 240.0,
    "success_alignment_ratio": 0.9,  # 90% of steps must have center_alignment > 0.85
    # Distance-based success criterion: robot tip must be within this threshold of target position
    "success_distance_threshold": 0.2,
}

simulation_config = {
    "device" : device,
    # The device to run the simulation on. Default is ``"cuda:0"``.
    # Valid options are:
    # - ``"cpu"``: Use CPU.
    # - ``"cuda"``: Use GPU, where the device ID is inferred from :class:`~isaaclab.app.AppLauncher`'s config.
    # - ``"cuda:N"``: Use GPU, where N is the device ID. For example, "cuda:0".
    "dt" : 1.0 / 240.0,
    # The physics simulation time-step (in seconds). Default is 0.0167 seconds.
    "render_interval": 2,
    # The number of physics simulation steps per rendering step. Default is 1.
    "gravity" : (0, 0.0, 0),
    # The gravity vector (in m/s^2). Default is (0.0, 0.0, -9.81).
    # If set to (0.0, 0.0, 0.0), gravity is disabled.
    "enable_scene_query_support" : False,
    # Enable/disable scene query support for collision shapes. Default is False.
    # This flag allows performing collision queries (raycasts, sweeps, and overlaps) on actors and
    # attached shapes in the scene. This is useful for implementing custom collision detection logic
    # outside of the physics engine.
    # If set to False, the physics engine does not create the scene query manager and the scene query
    # functionality will not be available. However, this provides some performance speed-up.
    # Note:
    #     This flag is overridden to True inside the :class:`SimulationContext` class when running the simulation
    #     with the GUI enabled. This is to allow certain GUI features to work properly.
    "use_fabric" : True,
    # Enable/disable reading of physics buffers directly. Default is True.
    # When running the simulation, updates in the states in the scene is normally synchronized with USD.
    # This leads to an overhead in reading the data and does not scale well with massive parallelization.
    # This flag allows disabling the synchronization and reading the data directly from the physics buffers.
    # It is recommended to set this flag to :obj:`True` when running the simulation with a large number
    # of primitives in the scene.
    # Note:
    #     When enabled, the GUI will not update the physics parameters in real-time. To enable real-time
    #     updates, please set this flag to :obj:`False`.
    #     When using GPU simulation, it is required to enable Fabric to visualize updates in the renderer.
    #     Transform updates are propagated to the renderer through Fabric. If Fabric is disabled with GPU simulation,
    #     the renderer will not be able to render any updates in the simulation, although simulation will still be
    #     running under the hood.
}

render_config = {
    "enable_translucency" : False,
	# Bool. Enables translucency for specular transmissive surfaces such as glass at the cost of some performance.
    "enable_reflections" : False,
    # Bool. Enables reflections at the cost of some performance.
    "enable_global_illumination" : True,
    # Bool. Enables Diffused Global Illumination at the cost of some performance.
    "antialiasing_mode" : "FXAA",
	# Literal[“Off”, “FXAA”, “DLSS”, “TAA”, “DLAA”].
    # DLSS: Boosts performance by using AI to output higher resolution frames from a lower resolution input. DLSS samples multiple lower resolution images and uses motion data and feedback from prior frames to reconstruct native quality images. DLAA: Provides higher image quality with an AI-based anti-aliasing technique. DLAA uses the same Super Resolution technology developed for DLSS, reconstructing a native resolution image to maximize image quality.
    "enable_dlssg" : False,
    # Bool. Enables the use of DLSS-G. DLSS Frame Generation boosts performance by using AI to generate more frames. This feature requires an Ada Lovelace architecture GPU and can hurt performance due to additional thread-related activities.
    "enable_dl_denoiser" : False,
	# Bool. Enables the use of a DL denoiser, which improves the quality of renders at the cost of performance.
    "dlss_mode" : 1,
	# Literal[0, 1, 2, 3]. For DLSS anti-aliasing, selects the performance/ quality tradeoff mode. Valid values are 0 (Performance), 1 (Balanced), 2 (Quality), or 3 (Auto).
    "enable_direct_lighting" : True,
	# Bool. Enable direct light contributions from lights.
    "samples_per_pixel" : 1,
	# Int. Defines the Direct Lighting samples per pixel. Higher values increase the direct lighting quality at the cost of performance.
    "enable_shadows" : True,
	# Bool. Enables shadows at the cost of performance. When disabled, lights will not cast shadows.
    "enable_ambient_occlusion" : True,  # Enable to darken recessed areas (lumen center)
	# Bool. Enables ambient occlusion at the cost of some performance.
}

physx_config = {
    "solver_type" : 1,
    # The type of solver to use.Default is 1 (TGS).
    # Available solvers:
    # * :obj:`0`: PGS (Projective Gauss-Seidel)
    # * :obj:`1`: TGS (Temporal Gauss-Seidel)
    "min_position_iteration_count" : 1,
    # Minimum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 1.
    # .. note::
    #
    #     Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
    #     the number of iterations specified by the actor with the highest iteration and clamps it to
    #     the range ``[min_position_iteration_count, max_position_iteration_count]``.
    "max_position_iteration_count" : 32,
    # Maximum number of solver position iterations (rigid bodies, cloth, particles etc.). Default is 255.
    # .. note::
    #     Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
    #     the number of iterations specified by the actor with the highest iteration and clamps it to
    #     the range ``[min_position_iteration_count, max_position_iteration_count]``.
    "min_velocity_iteration_count" : 0,
    # Minimum number of solver velocity iterations (rigid bodies, cloth, particles etc.). Default is 0.
    # .. note::
    #     Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
    #     the number of iterations specified by the actor with the highest iteration and clamps it to
    #     the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    "max_velocity_iteration_count" : 32,
    # Maximum number of solver velocity iterations (rigid bodies, cloth, particles etc.). Default is 255.
    # .. note::
    #     Each physics actor in Omniverse specifies its own solver iteration count. The solver takes
    #     the number of iterations specified by the actor with the highest iteration and clamps it to
    #     the range ``[min_velocity_iteration_count, max_velocity_iteration_count]``.
    "enable_ccd" : True,
    # Enable a second broad-phase pass that makes it possible to prevent objects from tunneling through each other.
    # Default is False.
    "enable_stabilization" : True,
    # Enable/disable additional stabilization pass in solver. Default is False.
    # .. note::
    #     We recommend setting this flag to true only when the simulation step size is large (i.e., less than 30 Hz or more than 0.0333 seconds).
    # .. warning::
    #     Enabling this flag may lead to incorrect contact forces report from the contact sensor.
    "enable_enhanced_determinism" : False,
    # Enable/disable improved determinism at the expense of performance. Defaults to False.
    # For more information on PhysX determinism, please check `here`_.
    # .. _here: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyDynamics.html#enhanced-determinism
    "bounce_threshold_velocity" : 0.5,
    # Relative velocity threshold for contacts to bounce (in m/s). Default is 0.5 m/s.
    "friction_offset_threshold" : 0.04,
    # Threshold for contact point to experience friction force (in m). Default is 0.04 m.
    "friction_correlation_distance" : 0.025,
    # Distance threshold for merging contacts into a single friction anchor point (in m). Default is 0.025 m.
    "gpu_max_rigid_contact_count" : 2**23,
    # Size of rigid contact stream buffer allocated in pinned host memory. Default is 2 ** 23.
    "gpu_max_rigid_patch_count" : 5 * 2**15,
    # Size of the rigid contact patch stream buffer allocated in pinned host memory. Default is 5 * 2 ** 15.
    "gpu_found_lost_pairs_capacity" : 2**21,
    # Capacity of found and lost buffers allocated in GPU global memory. Default is 2 ** 21.
    # This is used for the found/lost pair reports in the BP.
    "gpu_found_lost_aggregate_pairs_capacity" : 2**25,
    # Capacity of found and lost buffers in aggregate system allocated in GPU global memory.
    # Default is 2 ** 25.
    # This is used for the found/lost pair reports in AABB manager.
    "gpu_total_aggregate_pairs_capacity" : 2**21,
    # Capacity of total number of aggregate pairs allocated in GPU global memory. Default is 2 ** 21.
    "gpu_collision_stack_size" : 2**26,
    # Size of the collision stack buffer allocated in pinned host memory. Default is 2 ** 26.
    "gpu_heap_capacity" : 2**26,
    # Initial capacity of the GPU and pinned host memory heaps. Additional memory will be allocated
    # if more memory is required. Default is 2 ** 26.
    "gpu_temp_buffer_capacity" : 2**24,
    # Capacity of temp buffer allocated in pinned host memory. Default is 2 ** 24.
    "gpu_max_num_partitions" : 8,
    # Limitation for the partitions in the GPU dynamics pipeline. Default is 8.
    # This variable must be power of 2. A value greater than 32 is currently not supported. Range: (1, 32)
    "gpu_max_soft_body_contacts" : 2**20,
    # Size of soft body contacts stream buffer allocated in pinned host memory. Default is 2 ** 20.
    "gpu_max_particle_contacts" : 2**20,
    # Size of particle contacts stream buffer allocated in pinned host memory. Default is 2 ** 20.
}

debug_config = {
    "show_markers" : False,
}

config = {
    "learning_config" : learning_config,
    "env_config" : env_config,
    "reward_config" : reward_config,
    "robot_config" : robot_config,
    "simulation_config" : simulation_config,
    "render_config" : render_config,
    "physx_config" : physx_config,
    "debug_config" : debug_config,
    "run_config": {
        "algorithm": args_cli.algo,
        "reward_variant": reward_variant_label,
        "constrained": bool(args_cli.clip_actions),
        "colon_id": args_cli.colon_id,
        "task_id": args_cli.task_id,
        "seed": args_cli.seed,
        "timestamp": timestamp,
        "result_dir": result_dir,
        "train_with_normalized_reward": bool(args_cli.train_with_normalized_reward),
        "stop_after_episodes": args_cli.stop_after_episodes,
        "disable_video": bool(args_cli.disable_video),
        "trajectory_save_interval": args_cli.trajectory_save_interval,
        "save_trajectory_images": bool(args_cli.save_trajectory_images),
        "youngs_modulus_mpa": args_cli.youngs_modulus,
        "youngs_modulus_pa": args_cli.youngs_modulus * 1e6 if args_cli.youngs_modulus is not None else None,
        "poisson_ratio": args_cli.poisson_ratio,
        "parameter_sensitivity_test": bool(args_cli.parameter_sensitivity_test),
        "parameter_sensitivity_episodes": args_cli.parameter_sensitivity_episodes,
        "anchor_setting": args_cli.anchor_setting,
        "anchor_sensitivity_test": bool(args_cli.anchor_sensitivity_test),
        "cross_colon2": bool(args_cli.cross_colon2),
        "cross_colon": bool(args_cli.cross_colon),
        "cross_colon_eval_only": bool(args_cli.cross_colon_eval_only),
        "cross_colon_episodes": args_cli.cross_colon_episodes,
        "cross_colon_trials": args_cli.cross_colon_trials,
        "continual_training_mode": bool(continual_training_mode),
        "continual_training_action_mode": bool(continual_training_action_mode),
        "policy_guided_data_sampling": bool(policy_guided_data_sampling),
        "random_data_sampling": bool(random_data_sampling),
        "sampling_deterministic_actions": bool(policy_sampling_deterministic),
        "data_sampling_colon1_forward_only": bool(data_sampling_colon1_forward_only),
        "data_sampling_no_backward_recovery": bool(data_sampling_no_backward_recovery),
        "model_path": args_cli.model_path,
        "data_sampling_sequence": bool(args_cli.data_sampling_sequence),
        "data_sampling_sequence_repeats": args_cli.data_sampling_sequence_repeats,
        "data_sampling_sequence_colons": args_cli.data_sampling_sequence_colons,
        "data_sampling_target_reaches": args_cli.data_sampling_target_reaches,
        "medical_data_dir": args_cli.medical_data_dir,
        "medical_run_dir": args_cli.medical_run_dir,
        "medical_data_save": bool(args_cli.data_sampling and not args_cli.disable_medical_data_save),
        "medical_save_images": bool(args_cli.medical_save_images),
        "init_from_csv": task_startpose_csv,
        "init_endpose_from_csv": task_endpose_csv,
    },
}
with open(os.path.join(result_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2, default=str)
#pdb.set_trace()
robot_factory = RobotFactory(config, device=device)
#pdb.set_trace()
env_cfg = make_isaac_env_cfg(config, robot_factory)

env = gym.make("ArcIsaacEnv-v0", cfg=env_cfg, robot_factory=robot_factory,
               config=config, tracer=tracer, disable_env_checker=True) # We need disable_env_checker=True because it fails due to the wrapper removing the 'policy' key
               #, render_mode = "rgb_array"

env = FrameStack(env, n_stack=5)

video_kwargs = {
    "video_folder": video_save_path,
    "step_trigger": lambda step: step % 5000 == 0,
    "video_length": 5000,
}
if not args_cli.disable_video:
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

env = Sb3VecEnvWrapper(env)

# Set camera to view robot and colon properly
# The unwrapped environment has access to the sim
try:
    import omni.isaac.core.utils.viewports as vp_utils
    # Get the entry position from the environment to center the camera on it
    # Assuming the robot and colon are around entry_positions
    # Set camera position: above and behind the scene
    # Adjust these values based on your scene scale
    eye = [20.0, 10.0, 8.0]  # Camera position (x, y, z) - zoomed out to see larger area
    target = [5.0, 1.0, 1.0]  # Look at point - center of scene
    vp_utils.set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    print(f"Camera set to eye={eye}, target={target}")
except Exception as e:
    print(f"Could not set camera view: {e}")

# env = VecVideoRecorder(
#     env,
#     video_folder=video_save_path,
#     record_video_trigger=lambda x: x % 10000 == 0,
#     video_length=2500,
#     name_prefix="training"
# )

# Debug your environment
#pdb.set_trace()
print(f"Obs space: {env.observation_space}")
print(f"Action space: {env.action_space}")

if config["env_config"]["discrete_action_space"]:
    raise NotImplementedError("Discrete action space is not currently supported")


def _float_from_info(info, key):
    if key not in info or info[key] is None:
        return None
    value = info[key]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _numpy_from_value(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sanitize_dataset_component(value):
    parts = [
        part
        for part in os.path.normpath(str(value)).split(os.sep)
        if part not in ("", ".", "..")
    ]
    safe_parts = []
    for part in parts[-4:]:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in part)
        if safe:
            safe_parts.append(safe)
    return "__".join(safe_parts) if safe_parts else "run"


def _image_to_uint8_hwc(image):
    image_np = _numpy_from_value(image)
    if image_np is None:
        return None
    image_np = np.asarray(image_np)
    if image_np.ndim == 3 and image_np.shape[0] in (1, 3, 4):
        image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.ndim == 2:
        image_np = np.repeat(image_np[:, :, None], 3, axis=2)
    if image_np.ndim == 3 and image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)
    if image_np.ndim == 3 and image_np.shape[2] > 4:
        image_np = image_np[:, :, :3]
    if image_np.dtype != np.uint8:
        if image_np.size and np.nanmax(image_np) <= 1.0:
            image_np = image_np * 255.0
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=255.0, neginf=0.0)
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return image_np


def _batched_action_array(actions, num_envs):
    actions_np = _numpy_from_value(actions)
    if actions_np is None:
        return None
    actions_np = np.asarray(actions_np, dtype=np.float64)
    if actions_np.ndim == 1:
        actions_np = actions_np.reshape(1, -1)
    else:
        actions_np = actions_np.reshape(actions_np.shape[0], -1)
    if actions_np.shape[0] == 1 and num_envs > 1:
        actions_np = np.repeat(actions_np, num_envs, axis=0)
    return actions_np


def _base_isaac_env(vec_env):
    base_env = getattr(vec_env, "unwrapped", None)
    return base_env() if callable(base_env) else base_env


class MedicalTrainingDataWriter:
    """Stream synchronized data-sampling frames, actions, rewards, and ROI labels."""

    def __init__(self, save_root, result_dir, config, action_dim, save_images=True, run_dir=None):
        self.save_root = os.path.abspath(os.path.expanduser(save_root))
        if run_dir is None:
            self.run_id = _sanitize_dataset_component(result_dir)
            self.run_dir = os.path.join(self.save_root, self.run_id)
        else:
            self.run_dir = os.path.abspath(os.path.expanduser(run_dir))
            self.run_id = os.path.relpath(self.run_dir, self.save_root)
        self.save_images = bool(save_images)
        self.action_dim = int(action_dim)
        self.sample_index = 0

        os.makedirs(self.run_dir, exist_ok=True)
        self.env_dirs = {}
        self.env_image_dirs = {}
        self.env_csv_paths = {}
        self.env_sample_indices = {}
        for env_idx in range(int(config["env_config"]["num_envs"])):
            env_dir = os.path.join(self.run_dir, f"env_{env_idx:03d}")
            image_dir = os.path.join(env_dir, "images")
            os.makedirs(env_dir, exist_ok=True)
            os.makedirs(image_dir, exist_ok=True)
            self.env_dirs[env_idx] = env_dir
            self.env_image_dirs[env_idx] = image_dir
            self.env_csv_paths[env_idx] = os.path.join(env_dir, "samples.csv")
            self.env_sample_indices[env_idx] = 0

        metadata = {
            "dataset_type": "sim4medical_training_data",
            "run_id": self.run_id,
            "result_dir": result_dir,
            "colon_id": config["run_config"]["colon_id"],
            "task_id": config["run_config"]["task_id"],
            "seed": config["run_config"]["seed"],
            "model_path": config["run_config"].get("model_path"),
            "action_columns": [f"action_{idx}" for idx in range(self.action_dim)],
            "policy_action_columns": [f"policy_action_{idx}" for idx in range(self.action_dim)],
            "action_definition": "Executed post-constraint action at the same timestep as the saved image and reward.",
            "roi_definition": "ROI centroid from the high-depth lumen region, relative to image center. x>0 is right, y>0 is down.",
            "csv_layout": "One samples.csv per environment directory. image_path is relative to that environment directory.",
            "env_csv_paths": {
                f"env_{env_idx:03d}": os.path.relpath(csv_path, self.run_dir)
                for env_idx, csv_path in self.env_csv_paths.items()
            },
        }
        with open(os.path.join(self.run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        self.fieldnames = [
            "sample_index",
            "vec_step",
            "env_idx",
            "image_path",
            "normalized_reward",
            "raw_step_reward",
            "env_reward",
            "roi_relative_x",
            "roi_relative_y",
            "roi_center_x_px",
            "roi_center_y_px",
            "roi_distance_from_center",
            "center_alignment",
            "roi_aligned",
            "lumen_visible",
            "goal_reached",
            "colon_id",
            "task_id",
        ]
        self.fieldnames.extend(f"action_{idx}" for idx in range(self.action_dim))
        self.fieldnames.extend(f"policy_action_{idx}" for idx in range(self.action_dim))

        self._csv_files = {}
        self._writers = {}
        for env_idx, csv_path in self.env_csv_paths.items():
            csv_file = open(csv_path, "w", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()
            csv_file.flush()
            self._csv_files[env_idx] = csv_file
            self._writers[env_idx] = writer

    def _action_fields(self, prefix, values):
        row = {}
        values = [] if values is None else np.asarray(values, dtype=np.float64).reshape(-1).tolist()
        for idx in range(self.action_dim):
            row[f"{prefix}_{idx}"] = float(values[idx]) if idx < len(values) else None
        return row

    def write_step(self, vec_step, policy_actions, executed_actions, rewards, infos, base_env, colon_id, task_id):
        rgb_images = _numpy_from_value(getattr(base_env, "_camera_data", None)) if base_env is not None else None
        policy_actions_np = _batched_action_array(policy_actions, len(infos))
        executed_actions_np = _batched_action_array(executed_actions, len(infos))
        rewards_np = np.asarray(rewards, dtype=np.float64).reshape(-1)

        for env_idx, info in enumerate(infos):
            image_rel_path = ""
            if self.save_images and rgb_images is not None and env_idx < len(rgb_images):
                frame_name = f"frame_{int(vec_step):08d}.png"
                image_abs_path = os.path.join(self.env_image_dirs[env_idx], frame_name)
                image_rel_path = os.path.relpath(image_abs_path, self.env_dirs[env_idx])
                image_np = _image_to_uint8_hwc(rgb_images[env_idx])
                if image_np is not None:
                    Image.fromarray(image_np).save(image_abs_path)

            normalized_reward = _float_from_info(info, "normalized_step_reward")
            if normalized_reward is None and env_idx < len(rewards_np):
                normalized_reward = float(np.clip(rewards_np[env_idx], -1.0, 1.0))

            row = {
                "sample_index": int(self.env_sample_indices.get(env_idx, 0)),
                "vec_step": int(vec_step),
                "env_idx": int(env_idx),
                "image_path": image_rel_path,
                "normalized_reward": normalized_reward,
                "raw_step_reward": _float_from_info(info, "raw_step_reward"),
                "env_reward": float(rewards_np[env_idx]) if env_idx < len(rewards_np) else None,
                "roi_relative_x": _float_from_info(info, "roi_relative_x"),
                "roi_relative_y": _float_from_info(info, "roi_relative_y"),
                "roi_center_x_px": _float_from_info(info, "roi_center_x_px"),
                "roi_center_y_px": _float_from_info(info, "roi_center_y_px"),
                "roi_distance_from_center": _float_from_info(info, "roi_distance_from_center"),
                "center_alignment": _float_from_info(info, "center_alignment"),
                "roi_aligned": _float_from_info(info, "roi_aligned"),
                "lumen_visible": _float_from_info(info, "lumen_visible"),
                "goal_reached": bool(info.get("goal_reached", False)),
                "colon_id": colon_id,
                "task_id": task_id,
            }
            executed_values = executed_actions_np[env_idx] if executed_actions_np is not None and env_idx < len(executed_actions_np) else None
            policy_values = policy_actions_np[env_idx] if policy_actions_np is not None and env_idx < len(policy_actions_np) else None
            row.update(self._action_fields("action", executed_values))
            row.update(self._action_fields("policy_action", policy_values))
            self._writers[env_idx].writerow(row)
            self.env_sample_indices[env_idx] = self.env_sample_indices.get(env_idx, 0) + 1
            self.sample_index += 1
        for csv_file in self._csv_files.values():
            csv_file.flush()

    def close(self):
        for csv_file in self._csv_files.values():
            csv_file.flush()
            csv_file.close()
        self._csv_files = {}
        self._writers = {}


def run_deterministic_evaluation(model, env, num_episodes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    action_log_file = None
    action_log_writer = None
    if args_cli.print_actions:
        action_log_path = os.path.join(output_dir, "printed_actions.csv")
        action_log_file = open(action_log_path, "w", newline="")
        action_dim = int(np.prod(env.action_space.shape))
        fieldnames = ["vec_step", "env_idx"] + [f"action_{idx}" for idx in range(action_dim)]
        action_log_writer = csv.DictWriter(action_log_file, fieldnames=fieldnames)
        action_log_writer.writeheader()
        logging.info("Printing deterministic policy actions and saving them to: %s", action_log_path)

    metric_keys = [
        "raw_step_reward",
        "normalized_step_reward",
        "s_c",
        "s_1",
        "s_2",
        "s_3",
        "s_o",
        "normalized_progress",
        "roi_aligned",
        "lumen_visible",
    ]
    metric_values = {key: [] for key in metric_keys}
    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    obs = env.reset()
    current_rewards = np.zeros(env.num_envs, dtype=np.float64)
    current_lengths = np.zeros(env.num_envs, dtype=np.int64)
    episodes_completed = 0
    total_steps = 0

    while episodes_completed < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        if args_cli.print_actions and total_steps < args_cli.print_action_steps * env.num_envs:
            action_array = np.asarray(action)
            if action_array.ndim == 1:
                action_array = action_array.reshape(1, -1)
            vec_step = total_steps // env.num_envs
            for env_idx, action_values in enumerate(action_array):
                flattened = np.asarray(action_values, dtype=np.float64).reshape(-1)
                print(
                    f"[ACTION] vec_step={vec_step} env={env_idx} "
                    f"policy_action={flattened.tolist()}"
                )
                if action_log_writer is not None:
                    action_log_writer.writerow(
                        {
                            "vec_step": int(vec_step),
                            "env_idx": int(env_idx),
                            **{f"action_{idx}": float(value) for idx, value in enumerate(flattened)},
                        }
                    )
            if action_log_file is not None:
                action_log_file.flush()
        obs, rewards, dones, infos = env.step(action)
        current_rewards += rewards
        current_lengths += 1
        total_steps += env.num_envs

        for info in infos:
            for key in metric_keys:
                value = _float_from_info(info, key)
                if value is not None:
                    metric_values[key].append(value)

        for env_idx, done in enumerate(dones):
            if not done or episodes_completed >= num_episodes:
                continue
            info = infos[env_idx] if env_idx < len(infos) else {}
            success = bool(info.get("goal_reached", False) or info.get("is_success", False) or info.get("success", False))
            episode_rewards.append(float(current_rewards[env_idx]))
            episode_lengths.append(int(current_lengths[env_idx]))
            episode_successes.append(success)
            current_rewards[env_idx] = 0.0
            current_lengths[env_idx] = 0
            episodes_completed += 1

    def mean_or_none(key):
        values = metric_values[key]
        return float(np.mean(values)) if values else None

    results_summary = {
        "model_path": args_cli.model_path,
        "test_episodes": int(num_episodes),
        "episodes_completed": int(episodes_completed),
        "total_env_steps": int(total_steps),
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "normalized_progress": mean_or_none("normalized_progress"),
        "raw_mean_step_reward": mean_or_none("raw_step_reward"),
        "normalized_mean_step_reward": mean_or_none("normalized_step_reward"),
        "mean_s_c": mean_or_none("s_c"),
        "mean_s_1": mean_or_none("s_1"),
        "mean_s_2": mean_or_none("s_2"),
        "mean_s_3": mean_or_none("s_3"),
        "mean_s_o": mean_or_none("s_o"),
        "roi_alignment_rate": mean_or_none("roi_aligned"),
        "lumen_visible_ratio": mean_or_none("lumen_visible"),
        "mean_episode_return": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_episode_return": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "std_episode_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
        "episode_returns": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_successes": episode_successes,
        "reward_variant": reward_variant_label,
        "constrained": bool(args_cli.clip_actions),
        "colon_id": args_cli.colon_id,
        "task_id": args_cli.task_id,
        "seed": args_cli.seed,
        "algorithm": args_cli.algo,
    }
    results_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    if action_log_file is not None:
        action_log_file.close()
    logging.info(f"Evaluation metrics saved to: {results_path}")
    return results_summary

# Select algorithm based on command line argument.
# Continual training and policy-guided data sampling both load a checkpoint.
if continual_training_mode or policy_guided_data_sampling:
    load_context = "policy-guided data sampling" if policy_guided_data_sampling else "continual training"
    logging.info(f"Loading pre-trained model for {load_context} from: {args_cli.model_path}")
    if args_cli.algo == "PPO":
        model = PPO.load(args_cli.model_path, env=env, device=device, tensorboard_log=tensorboard_log)
    elif args_cli.algo == "SAC":
        model = SAC.load(args_cli.model_path, env=env, device=device, tensorboard_log=tensorboard_log)
    elif args_cli.algo == "TD3":
        model = TD3.load(args_cli.model_path, env=env, device=device, tensorboard_log=tensorboard_log)
    elif args_cli.algo == "A2C":
        model = A2C.load(args_cli.model_path, env=env, device=device, tensorboard_log=tensorboard_log)
    elif args_cli.algo == "DDPG":
        model = DDPG.load(args_cli.model_path, env=env, device=device, tensorboard_log=tensorboard_log)
    else:
        raise ValueError(f"Unknown algorithm: {args_cli.algo}")
    logging.info(f"Pre-trained model loaded successfully. Ready for {load_context}.")
elif args_cli.algo == "PPO":
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs={"normalize_images": False},
        tensorboard_log=tensorboard_log,
        device=device,
        seed=args_cli.seed,
        learning_rate=learning_config["learning_rate"],
        batch_size=learning_config["batch_size"],
        n_steps=2048,
        clip_range=0.2,
        ent_coef=0.1,
        n_epochs=64,
        gamma=0.99,
        gae_lambda=0.95,
    )
elif args_cli.algo == "SAC":
    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs={"normalize_images": False},
        tensorboard_log=tensorboard_log,
        device=device,
        seed=args_cli.seed,
        ent_coef=0.5,
        learning_starts=10000,
        learning_rate=learning_config["learning_rate"],
        buffer_size=10000,
        batch_size=learning_config["batch_size"],
        tau=0.005,
        gamma=0.99,
        train_freq=10,
        gradient_steps=2,
    )
elif args_cli.algo == "TD3":
    model = TD3(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs={"normalize_images": False},
        tensorboard_log=tensorboard_log,
        device=device,
        seed=args_cli.seed,
        learning_starts=5000,
        learning_rate=learning_config["learning_rate"],
        buffer_size=5000,
        batch_size=learning_config["batch_size"],
        tau=0.005,
        gamma=0.99,
        train_freq=10,
        gradient_steps=2,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    )
elif args_cli.algo == "A2C":
    model = A2C(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs={"normalize_images": False},
        tensorboard_log=tensorboard_log,
        device=device,
        seed=args_cli.seed,
        learning_rate=learning_config["learning_rate"],
        n_steps=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
elif args_cli.algo == "DDPG":
    model = DDPG(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs={"normalize_images": False},
        tensorboard_log=tensorboard_log,
        device=device,
        seed=args_cli.seed,
        learning_starts=5000,
        learning_rate=learning_config["learning_rate"],
        buffer_size=5000,
        batch_size=learning_config["batch_size"],
        tau=0.005,
        gamma=0.99,
        train_freq=10,
        gradient_steps=2,
    )
else:
    raise ValueError(f"Unknown algorithm: {args_cli.algo}")

# Print the policy network structure
logging.debug("Policy architecture:")
logging.debug(model.policy)

# More detailed view of the policy networks
# PPO has 'action_net' and 'value_net', SAC has 'actor' and 'critic'
if hasattr(model.policy, 'actor'):
    # SAC-specific
    logging.debug("\nActor network:")
    logging.debug(model.policy.actor)
    logging.debug("\nCritic network:")
    logging.debug(model.policy.critic)
elif hasattr(model.policy, 'action_net'):
    # PPO-specific (ActorCriticPolicy)
    logging.debug("\nAction network (actor):")
    logging.debug(model.policy.action_net)
    logging.debug("\nValue network (critic):")
    logging.debug(model.policy.value_net)
    logging.debug("\nMLP extractor:")
    logging.debug(model.policy.mlp_extractor)

logging.debug("\nFeature extractor:")
logging.debug(model.policy.features_extractor)

continual_training_target_episodes = max(1, int(args_cli.continual_training_target_episodes))

if continual_training_mode:
    max_episode_steps = int(getattr(env.unwrapped, "max_episode_length", 0))
    if max_episode_steps > 0:
        min_steps_before_training = continual_training_target_episodes * max_episode_steps + 1
        # Prevent any training update before every env has had time to finish the target episodes.
        # On-policy algorithms train after each rollout of `n_steps` per env.
        if hasattr(model, "n_steps") and hasattr(model, "rollout_buffer"):
            desired_n_steps = max(int(model.n_steps), min_steps_before_training)
            if desired_n_steps != int(model.n_steps):
                logging.info(
                    "continual_training: increasing rollout horizon from %s to %s so no training "
                    "occurs before all envs can finish %s episodes",
                    model.n_steps,
                    desired_n_steps,
                    continual_training_target_episodes,
                )
                model.n_steps = desired_n_steps
                rollout_buffer_cls = model.rollout_buffer.__class__
                model.rollout_buffer = rollout_buffer_cls(
                    model.n_steps,
                    model.observation_space,
                    model.action_space,
                    device=model.device,
                    gamma=model.gamma,
                    gae_lambda=model.gae_lambda,
                    n_envs=model.n_envs,
                )

        # Off-policy algorithms begin gradient updates after `learning_starts` global timesteps.
        if hasattr(model, "learning_starts"):
            desired_learning_starts = max(
                int(model.learning_starts),
                min_steps_before_training * env.num_envs,
            )
            if desired_learning_starts != int(model.learning_starts):
                logging.info(
                    "continual_training: increasing learning_starts from %s to %s so no training "
                    "occurs before all envs can finish %s episodes",
                    model.learning_starts,
                    desired_learning_starts,
                    continual_training_target_episodes,
                )
                model.learning_starts = desired_learning_starts

if continual_training_mode and args_cli.cross_colon_eval_only:
    logging.info(
        "Running cross-colon evaluation only: colon=%s task=%s episodes=%s. No training updates.",
        args_cli.colon_id,
        args_cli.task_id,
        args_cli.test_episodes,
    )
    run_deterministic_evaluation(model, env, args_cli.test_episodes, result_dir)
    with open(os.path.join(result_dir, "run_status.json"), "w") as f:
        json.dump(
            {
                "status": "evaluated",
                "model_path": args_cli.model_path,
                "timesteps": int(getattr(model, "num_timesteps", 0)),
                "cross_colon_eval_only": True,
                "colon_id": args_cli.colon_id,
                "task_id": args_cli.task_id,
            },
            f,
            indent=2,
        )
elif args_cli.train or continual_training_mode:
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, # Save every 10k steps
        save_path=model_save_path,
        name_prefix=f"{algo_name}_arc_checkpoint_{timestamp}"
    )

    log_callback = LogEveryNTimesteps(n_steps=500)

    per_env_reward_callback = PerEnvRewardCallback(
        verbose=1,
        plot_data_file=os.path.join(result_dir, "episode_rewards.npz"),
    )

    reward_ablation_callback = RewardAblationMetricsCallback(
        save_dir=result_dir,
        reward_variant=reward_variant_label,
        constrained=bool(args_cli.clip_actions),
        log_interval_steps=500,
        verbose=1,
    )

    # Create trajectory data saver callback.
    trajectory_callback = TrajectoryDataSaver(
        save_dir=trajectory_save_path,
        save_interval=args_cli.trajectory_save_interval,
        save_images=args_cli.save_trajectory_images,
        verbose=1
    )

    # Create success rate data saver callback (saves to trajectory_data folder).
    success_rate_callback = SuccessRateDataSaver(
        save_dir=trajectory_save_path,
        save_interval_episodes=1,  # Save every episode
        verbose=1
    )

    callbacks = [
        checkpoint_callback,
        log_callback,
        per_env_reward_callback,
        reward_ablation_callback,
        trajectory_callback,
        success_rate_callback,
    ]
    if args_cli.stop_after_episodes is not None:
        callbacks.append(
            StopAfterTotalEpisodesCallback(
                target_episodes=args_cli.stop_after_episodes,
                verbose=1,
            )
        )
    if continual_training_mode:
        callbacks.append(
            FreezeLearningUntilAllEnvsNEpisodesCallback(
                target_episodes=continual_training_target_episodes,
                verbose=1,
            )
        )
        callbacks.append(
            StopAfterAllEnvsNEpisodesCallback(
                target_episodes=continual_training_target_episodes,
                verbose=1,
            )
        )
    interrupted = False
    training_error = None
    learn_completed = False
    try:
        model.learn(learning_config["total_timesteps"], callback=callbacks)
        learn_completed = True
    except KeyboardInterrupt:
        interrupted = True
        logging.warning("Training interrupted by user. Saving current model and flushed metrics before exiting.")
    except Exception as exc:
        training_error = exc
        logging.exception("Training failed. Saving current model and flushed metrics before re-raising.")
    finally:
        # SB3 does not guarantee callback finalizers run when learn() is interrupted.
        # Call them explicitly so CSV summaries, success data, and episode reward
        # arrays are written even after Ctrl+C.
        if not learn_completed:
            for callback in callbacks:
                try:
                    callback._on_training_end()
                except Exception as exc:
                    logging.warning(f"Callback finalization failed for {type(callback).__name__}: {exc}")

    final_suffix = "failed" if training_error is not None else "interrupted" if interrupted else "final"
    final_model_path = os.path.join(model_save_path, f"{algo_name}_arc_{final_suffix}_{timestamp}")
    model.save(final_model_path)
    logging.info(f"{final_suffix.capitalize()} model saved to: {final_model_path}")
    run_status = {
        "status": "failed" if training_error is not None else "interrupted" if interrupted else "completed",
        "model_path": final_model_path,
        "timesteps": int(getattr(model, "num_timesteps", 0)),
        "error": repr(training_error) if training_error is not None else None,
    }
    with open(os.path.join(result_dir, "run_status.json"), "w") as f:
        json.dump(run_status, f, indent=2)
    args_cli.model_path = final_model_path
    if args_cli.eval_after_train and training_error is None:
        run_deterministic_evaluation(model, env, args_cli.test_episodes, result_dir)
    if training_error is not None:
        raise training_error
elif args_cli.test:
    # Test mode: Load a trained model and evaluate its performance
    logging.info(f"Loading model from: {args_cli.model_path}")

    # Determine the algorithm from the model path or use the --algo argument
    if args_cli.algo == "PPO":
        model = PPO.load(args_cli.model_path, env=env, device=device)
    elif args_cli.algo == "SAC":
        model = SAC.load(args_cli.model_path, env=env, device=device)
    elif args_cli.algo == "TD3":
        model = TD3.load(args_cli.model_path, env=env, device=device)
    elif args_cli.algo == "A2C":
        model = A2C.load(args_cli.model_path, env=env, device=device)
    elif args_cli.algo == "DDPG":
        model = DDPG.load(args_cli.model_path, env=env, device=device)

    logging.info(f"Model loaded successfully. Running {args_cli.test_episodes} test episodes...")
    run_deterministic_evaluation(model, env, args_cli.test_episodes, result_dir)
elif args_cli.data_sampling:
    logging.info("Starting data sampling mode...")
    if policy_guided_data_sampling:
        logging.info(
            "Using %s policy actions from %s. No model updates or checkpoint saves will be performed.",
            "deterministic" if policy_sampling_deterministic else "continual-training-style stochastic",
            args_cli.model_path,
        )
    else:
        logging.info("Using random actions generated inside the environment.")
    logging.info("Press 'G' to save the current robot configuration.")
    from arcgym.utils.keyboard import FPVKeyboard
    from arcgym.utils.robot_state_io import save_robot_state

    sampling_keyboard = FPVKeyboard(lin_sensitivity=1, rot_sensitivity=1)
    sampling_keyboard.add_callback("G", lambda: save_robot_state(env))
    sampling_keyboard.reset()

    obs = env.reset()
    zero_actions = np.zeros((env.num_envs,) + env.action_space.shape, dtype=np.float32)
    total_steps = 0
    target_reaches_completed = 0
    target_reach_goal = args_cli.data_sampling_target_reaches
    previous_goal_reached = np.zeros(env.num_envs, dtype=bool)
    sampling_status = "completed"
    medical_writer = None
    action_dim = int(np.prod(env.action_space.shape))
    if not args_cli.disable_medical_data_save:
        medical_writer = MedicalTrainingDataWriter(
            save_root=args_cli.medical_data_dir,
            result_dir=result_dir,
            config=config,
            action_dim=action_dim,
            save_images=args_cli.medical_save_images,
            run_dir=args_cli.medical_run_dir,
        )
        logging.info("Medical training data will be saved to: %s", medical_writer.run_dir)

    def _info_bool(info, key):
        value = info.get(key, False)
        if hasattr(value, "item"):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.reshape(-1)[0] if value.size else False
        return bool(value)

    try:
        while simulation_app.is_running():
            sampling_keyboard.advance()
            vec_step = total_steps // env.num_envs
            if policy_guided_data_sampling:
                actions, _ = model.predict(obs, deterministic=policy_sampling_deterministic)
            else:
                actions = zero_actions
            obs, reward, done, info = env.step(actions)
            if medical_writer is not None:
                base_env = _base_isaac_env(env)
                executed_actions = getattr(base_env, "actions", actions) if base_env is not None else actions
                medical_writer.write_step(
                    vec_step=vec_step,
                    policy_actions=actions,
                    executed_actions=executed_actions,
                    rewards=reward,
                    infos=info,
                    base_env=base_env,
                    colon_id=args_cli.colon_id,
                    task_id=args_cli.task_id,
                )
            total_steps += env.num_envs
            goal_reached = np.asarray([_info_bool(env_info, "goal_reached") for env_info in info], dtype=bool)
            fresh_reaches = goal_reached & ~previous_goal_reached
            if fresh_reaches.any():
                target_reaches_completed += int(fresh_reaches.sum())
                reached_envs = np.where(fresh_reaches)[0].tolist()
                logging.info(
                    "Data sampling target reached in envs %s (%s/%s)",
                    reached_envs,
                    target_reaches_completed,
                    target_reach_goal if target_reach_goal is not None else "unbounded",
                )
            previous_goal_reached = goal_reached
            if target_reach_goal is not None and target_reaches_completed >= target_reach_goal:
                logging.info("Target reach quota met. Exiting data sampling run.")
                break
    except KeyboardInterrupt:
        sampling_status = "interrupted"
        logging.info("\nCtrl+C detected. Exiting data sampling mode.")
    except Exception:
        sampling_status = "failed"
        logging.exception("Data sampling failed.")
        raise
    finally:
        medical_data_run_dir = None
        medical_samples = 0
        if medical_writer is not None:
            medical_data_run_dir = medical_writer.run_dir
            medical_samples = int(medical_writer.sample_index)
            medical_writer.close()
        final_sampling_status = sampling_status
        if (
            final_sampling_status == "completed"
            and target_reach_goal is not None
            and target_reaches_completed < target_reach_goal
        ):
            final_sampling_status = "target_not_reached"
        with open(os.path.join(result_dir, "run_status.json"), "w") as f:
            json.dump(
                {
                    "status": final_sampling_status,
                    "mode": "policy_guided_data_sampling" if policy_guided_data_sampling else "random_data_sampling",
                    "model_path": args_cli.model_path if policy_guided_data_sampling else None,
                    "deterministic": bool(policy_guided_data_sampling and policy_sampling_deterministic),
                    "model_updates": False,
                    "clip_actions": bool(args_cli.clip_actions),
                    "data_sampling_colon1_forward_only": bool(data_sampling_colon1_forward_only),
                    "data_sampling_no_backward_recovery": bool(data_sampling_no_backward_recovery),
                    "init_from_csv": task_startpose_csv,
                    "init_endpose_from_csv": task_endpose_csv,
                    "target_reach_goal": target_reach_goal,
                    "target_reaches": int(target_reaches_completed),
                    "total_env_steps": int(total_steps),
                    "medical_data_dir": medical_data_run_dir,
                    "medical_samples": medical_samples,
                },
                f,
                indent=2,
            )
else:
    from arcgym.utils.teleop_mode import run_teleoperation_mode
    run_teleoperation_mode(env, simulation_app, device=device)

try:
    env.close()
except Exception as exc:
    logging.warning("Environment close failed: %s", exc)

try:
    simulation_app.close()
except Exception as exc:
    logging.warning("Simulation app close failed: %s", exc)
