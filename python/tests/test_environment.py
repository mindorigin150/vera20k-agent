"""Long-lived environment contracts owned by vera20k-agent. Require external retail assets."""
import json
import os

import numpy as np
import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

from vera20k_agent import VeraBatch


@pytest.fixture
def assets():
    root = os.environ.get("VERA_ASSETS")
    if root is None:
        pytest.skip("set VERA_ASSETS to the external retail asset directory")
    return root


def diagnostic(batch, slot=0):
    return json.loads(batch.native._diagnostics(slot))


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_single_and_parallel_slots_match_every_frame(assets, seed):
    """One-frame timing and deterministic slot isolation are product requirements."""
    single = VeraBatch(assets, render=False, threads=1)
    batch = VeraBatch(assets, num_slots=2, render=False, threads=2)
    try:
        single.reset_slots({0: seed})
        observations = batch.reset_slots({0: seed, 1: seed + 10})
        assert all("rng" not in state and "state_hash" not in state for state in observations.values())
        mcv = next(e["id"] for e in observations[0]["entities"] if e["owned"] and e["type_name"] == "AMCV")
        for frame in range(1, 1001):
            actions = [{"command": "deploy", "entity_id": mcv}] if frame == 1 else []
            single.step_slots({0: actions})
            batch.step_slots({0: actions, 1: []})
            assert diagnostic(single) == diagnostic(batch)
        assert diagnostic(single)["sim_time_ms"] == 16000
        untouched = diagnostic(batch, 1)
        batch.reset_slots({0: 99})
        assert diagnostic(batch, 1) == untouched
    finally:
        single.close()
        batch.close()
        batch.close()


def test_syntax_failure_is_atomic_and_conditions_use_native_rejection(assets):
    batch = VeraBatch(assets, num_slots=2, render=False, threads=2)
    try:
        batch.reset_slots({0: 0, 1: 1})
        before = [diagnostic(batch, i) for i in range(2)]
        with pytest.raises(ValueError):
            batch.step_slots({0: [], 1: [{"command": "set_game_speed", "speed": 6}]})
        assert [diagnostic(batch, i) for i in range(2)] == before
        with pytest.raises(RuntimeError, match="unknown rules type"):
            batch.step_slots({0: [], 1: [{"command": "produce", "type_name": "MISSING_TYPE"}]})
        assert [diagnostic(batch, i) for i in range(2)] == before
        result = batch.step_slots({0: [{"command": "stop", "entity_id": 999999999}]})[0]
        assert result["frame"] == 1
        assert result["commands"] == [{"submitted_frame": 0, "received_frame": 1}]
    finally:
        batch.close()


def test_gpu_observation_reads_preserve_state_and_match_batch_pixels(assets):
    single = VeraBatch(assets)
    batch = VeraBatch(assets, num_slots=2, threads=2)
    try:
        other = single.reset_slots({0: 0})[0]
        states = batch.reset_slots({0: 42, 1: 0})
        np.testing.assert_array_equal(other["image"], states[1]["image"])
        a = single.reset_slots({0: 42})[0]
        np.testing.assert_array_equal(a["image"], states[0]["image"])
        for frame in range(1, 21):
            single.step_slots({0: []})
            batch.step_slots({0: [], 1: []})
            if frame % 10 == 0:
                before = diagnostic(single)
                one = single.observe_slots([0])[0]
                two = single.observe_slots([0])[0]
                np.testing.assert_array_equal(one["image"], two["image"])
                assert diagnostic(single) == before
                np.testing.assert_array_equal(one["image"], batch.observe_slots([0])[0]["image"])
                assert diagnostic(single) == diagnostic(batch)
        kept = batch.observe_slots([1])[1]
        batch.reset_slots({0: 5})
        np.testing.assert_array_equal(kept["image"], batch.observe_slots([1])[1]["image"])
        output = batch.render_slots([1, 0])
        assert output["images"].dtype == np.uint8 and output["images"].flags.c_contiguous
        assert output["slot_ids"].tolist() == [1, 0]
        assert output["frames"].tolist() == [20, 0]
    finally:
        single.close()
        batch.close()


def test_gpu_credits_display_updates_after_spending_without_input_events(assets):
    """Regression: direct production must update the HUD on committed frames."""
    batch = VeraBatch(assets)
    cpu = VeraBatch(assets, render=False)
    try:
        initial = batch.reset_slots({0: 42})[0]
        cpu.reset_slots({0: 42})
        mcv = next(e["id"] for e in initial["entities"] if e["owned"] and e["type_name"] == "AMCV")
        state = initial
        requested = False
        for frame in range(1, 401):
            actions = [{"command": "deploy", "entity_id": mcv}] if frame == 1 else []
            if not requested and frame % 10 == 0 and any(
                option["type_name"] == "GAPOWR" and option["enabled"] for option in state["build_options"]
            ):
                actions = [{"command": "produce", "type_name": "GAPOWR"}]
                requested = True
            batch.step_slots({0: actions})
            cpu.step_slots({0: actions})
            if frame % 10 == 0:
                state = batch.observe_slots([0])[0]
                assert diagnostic(batch) == diagnostic(cpu)
        assert requested and state["credits"] < initial["credits"]
        # The fixed 800x600 native HUD puts the credits digits inside this strip.
        assert np.any(initial["image"][3:18, 640:790] != state["image"][3:18, 640:790])
        before = diagnostic(batch)
        np.testing.assert_array_equal(state["image"], batch.observe_slots([0])[0]["image"])
        assert diagnostic(batch) == before
    finally:
        batch.close()
        cpu.close()


def test_gym_lifecycle_and_terminal_reset_contract(assets):
    env = gym.make("VERA20k-v0", assets=assets, config={"max_steps": 2}).unwrapped
    try:
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert info["frame"] == 0
        before = diagnostic(env.batch)
        with pytest.raises(RuntimeError):
            env.reset(seed=0, options={"map": "NOT_A_SKIRMISH_MAP.MAP"})
        assert diagnostic(env.batch) == before
        _, reward, terminated, truncated, _ = env.step([])
        assert (reward, terminated, truncated) == (0.0, False, False)
        assert env.render().shape == (600, 800, 3)
        _, reward, terminated, truncated, info = env.step([])
        assert (reward, terminated, truncated) == (0.0, False, True)
        assert info["sim_time_ms"] == 32
        with pytest.raises(Exception, match="reset"):
            env.step([])
        env.reset(seed=1)
        check_env(env)
        env.close()
        env.close()
        with pytest.raises(RuntimeError, match="closed"):
            env.batch.observe_slots([0])
    finally:
        env.close()


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_client_frame_path_matches_headless_logic(assets, seed):
    """The real client frame path and headless path must preserve state and RNG."""
    cpu = VeraBatch(assets, render=False)
    gpu = VeraBatch(assets)
    try:
        cpu.reset_slots({0: seed})
        state = gpu.reset_slots({0: seed})[0]
        client_initial = json.loads(gpu.native._client_diagnostics(0))
        initial = diagnostic(cpu)
        assert client_initial == {key: initial[key] for key in ("state_hash", "rng")}
        mcv = next(e["id"] for e in state["entities"] if e["owned"] and e["type_name"] == "AMCV")
        for frame in range(1, 1001):
            actions = [{"command": "deploy", "entity_id": mcv}] if frame == 1 else []
            cpu.step_slots({0: actions})
            gpu.step_slots({0: actions})
            assert diagnostic(cpu) == diagnostic(gpu), f"first differing frame: {frame}"
    finally:
        cpu.close()
        gpu.close()
