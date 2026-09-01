import json

import pytest

pytest.importorskip("stable_baselines3")

from arcgym.utils.callbacks import BestAndLastRewardCheckpointCallback


class DummyModel:
    def __init__(self):
        self.saved_paths = []

    def save(self, path):
        self.saved_paths.append(path)


def test_best_and_last_checkpoint_uses_episode_return(tmp_path):
    callback = BestAndLastRewardCheckpointCallback(
        save_dir=tmp_path,
        name_prefix="ppo_arc",
        window_episodes=2,
    )
    callback.model = DummyModel()

    callback.num_timesteps = 100
    callback.locals = {"dones": [True], "infos": [{"episode": {"r": 2.0}}]}
    assert callback._on_step()

    callback.num_timesteps = 200
    callback.locals = {"dones": [True], "infos": [{"episode": {"r": 4.0}}]}
    assert callback._on_step()

    assert str(tmp_path / "ppo_arc_last") in callback.model.saved_paths
    assert str(tmp_path / "ppo_arc_best") in callback.model.saved_paths
    best = json.loads((tmp_path / "ppo_arc_best_metrics.json").read_text())
    assert best["selection_metric"] == "mean_episode_return_recent_2ep"
    assert best["score"] == 3.0
    assert best["episodes_observed"] == 2
