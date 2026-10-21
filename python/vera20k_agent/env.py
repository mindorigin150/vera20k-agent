"""Gymnasium and sparse-slot interfaces. Each step advances one logic frame."""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from ._native import FRAME_MS, NativeBatch


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


class CommandSpace(gym.Space):
    """An ordered list of direct game commands. The native boundary validates syntax."""

    def sample(self, mask=None, probability=None):
        return []

    def contains(self, value):
        return isinstance(value, list) and all(
            isinstance(command, dict) and isinstance(command.get("command"), str)
            for command in value
        )


class PlayerObservationSpace(gym.Space):
    """Variable player knowledge and an optional RGB image."""

    def __init__(self, width: int, height: int):
        super().__init__()
        self.image_space = gym.spaces.Box(0, 255, (height, width, 3), np.uint8)

    def sample(self, mask=None, probability=None):
        raise NotImplementedError("player observations must come from an episode")

    def contains(self, value):
        return (
            isinstance(value, dict)
            and isinstance(value.get("entities"), list)
            and isinstance(value.get("frame"), int)
            and "image" in value
            and self.image_space.contains(value["image"])
        )


class VeraBatch:
    """Own independent Rust slots and a bounded simulation pool on one GPU."""

    env_fps = 1000.0 / FRAME_MS
    policy_interval_steps = 10

    def __init__(
        self, assets: str | Path, num_slots: int = 1, *, threads: int = 1,
        config: Mapping[str, Any] | None = None, render: bool = True,
        width: int = 800, height: int = 600, gpu_index: int = 0,
    ):
        self.assets = Path(assets).resolve(strict=True)
        self.num_slots = num_slots
        self.width, self.height = width, height
        self.render_enabled = render
        self.config = dict(config) if config is not None else {}
        self.native = NativeBatch(
            self.assets, _json(self.config), num_slots, threads,
            render, width, height, gpu_index,
        )
        self.closed = False
        self._episodes = [0] * num_slots
        self._frames: dict[int, dict] = {}

    def reset_slots(
        self, seeds: Mapping[int, int], *, episode_ids: Mapping[int, int] | None = None,
    ) -> dict[int, dict]:
        requests = [
            (slot, seed, self._episodes[slot] if episode_ids is None else episode_ids[slot])
            for slot, seed in seeds.items()
        ]
        rows = json.loads(self.native.reset_slots(_json(requests)))
        for row in rows:
            slot = row["slot"]
            self._episodes[slot] = row["observation"]["episode_id"] + 1
            self._frames[slot] = {key: row["observation"][key] for key in ("episode_id", "frame")}
        return self._with_images({row["slot"]: row["observation"] for row in rows}, list(seeds))

    def step_slots(self, actions: Mapping[int, list[dict]]) -> dict[int, dict]:
        rows = json.loads(self.native.step_slots(_json(list(actions.items()))))
        results = {row["slot"]: row["result"] for row in rows}
        for slot, result in results.items():
            self._frames[slot] = {key: result[key] for key in ("episode_id", "frame")}
        return results

    def observe_slots(self, slots: Sequence[int]) -> dict[int, dict]:
        rows = json.loads(self.native.observe_slots(list(slots)))
        observations = {row["slot"]: row["observation"] for row in rows}
        return self._with_images(observations, slots)

    def _with_images(self, observations, slots):
        if self.render_enabled and slots:
            images = self.render_slots(slots)
            for index, slot in enumerate(slots):
                observations[slot]["image"] = images["images"][index]
        return observations

    def render_slots(self, slots: Sequence[int]) -> dict[str, np.ndarray]:
        rows = self.native.render_slots(list(slots))
        images = np.empty((len(rows), self.height, self.width, 3), dtype=np.uint8)
        for index, (slot, pixels) in enumerate(rows):
            images[index] = np.frombuffer(pixels, np.uint8).reshape(self.height, self.width, 3)
        return {
            "images": images,
            "slot_ids": np.asarray(slots, dtype=np.int64),
            "episode_ids": np.asarray([self._frames[slot]["episode_id"] for slot in slots], dtype=np.uint64),
            "frames": np.asarray([self._frames[slot]["frame"] for slot in slots], dtype=np.uint64),
        }

    def close(self):
        if not self.closed:
            self.native.close()
            self._frames.clear()
            self.closed = True


class VeraEnv(gym.Env):
    """One direct-command skirmish. The caller resets explicitly after episode end."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 1000.0 / FRAME_MS}

    def __init__(
        self, assets: str | Path, *, config: Mapping[str, Any] | None = None,
        width: int = 800, height: int = 600, gpu_index: int = 0,
        threads: int = 1, render_mode: str = "rgb_array",
    ):
        if render_mode != "rgb_array":
            raise ValueError("render_mode must be rgb_array")
        self.render_mode = render_mode
        self.action_space = CommandSpace()
        self.observation_space = PlayerObservationSpace(width, height)
        self._kwargs = dict(assets=assets, threads=threads, width=width, height=height, gpu_index=gpu_index)
        self._config = dict(config) if config is not None else {}
        self.batch = VeraBatch(**self._kwargs, config=self._config)
        self._ready = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        config = {**self._config, **({} if options is None else options)}
        episode_seed = int(self.np_random.integers(0, 2**32)) if seed is None else seed
        if config != self.batch.config:
            candidate = VeraBatch(**self._kwargs, config=config)
            try:
                observation = candidate.reset_slots({0: episode_seed})[0]
            except BaseException:
                candidate.close()
                raise
            previous = self.batch
            self.batch = candidate
            previous.close()
        else:
            observation = self.batch.reset_slots({0: episode_seed})[0]
        observation.pop("episode_id")
        self._ready = True
        return observation, self._info(observation)

    def step(self, action):
        if not self._ready:
            raise gym.error.ResetNeeded("reset the environment before stepping")
        result = dict(self.batch.step_slots({0: action})[0])
        observation = self.batch.observe_slots([0])[0]
        observation.pop("episode_id")
        result.pop("episode_id")
        self._ready = not (result["terminated"] or result["truncated"])
        return observation, float(result["reward"]), result["terminated"], result["truncated"], result

    def render(self):
        return self.batch.render_slots([0])["images"][0]

    def close(self):
        self.batch.close()
        self._ready = False

    @staticmethod
    def _info(observation):
        return {name: observation[name] for name in ("frame", "sim_time_ms")}
