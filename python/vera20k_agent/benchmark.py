"""Optional latency_bench integration. The benchmark owns delay and observation cadence."""
from __future__ import annotations

import os
import time

from latency_bench.core.types import Action, Observation, StepResult
from latency_bench.executors.env_step_backend import EnvStepResponse, RemoteEnvSlotHandle

from .env import CommandSpace, VeraBatch


class VeraEnvStepBackend:
    """Adapt direct commands and sparse Rust slots to EnvStepBackend."""

    def __init__(self, assets, num_slots=1, **kwargs):
        self.batch = VeraBatch(assets, num_slots, **kwargs)
        self.num_slots = num_slots
        self.env_fps = self.batch.env_fps
        self.slot_handles = [RemoteEnvSlotHandle(
            slot_id=slot, env_fps=self.env_fps,
            noop_action=Action([], name="noop", is_noop=True), action_space=CommandSpace(),
        ) for slot in range(num_slots)]
        self.last_step_metadata_by_slot = {}
        self.collection_frames = {}
        self.closed = False

    def worker_pids(self):
        return []

    def reset_slot(self, slot_id, *, episode_id, seed):
        if seed is None:
            raise ValueError("VERA benchmark resets require an explicit seed")
        state = self.batch.reset_slots({slot_id: seed}, episode_ids={slot_id: episode_id})[slot_id]
        observation = self._observation(state)
        self.slot_handles[slot_id].update_observation(observation)
        return observation

    def observe_slots(self, slot_ids, *, capture_rgb=False):
        states = self.batch.observe_slots(slot_ids)
        observations = {slot: self._observation(state) for slot, state in states.items()}
        for slot, observation in observations.items():
            self.slot_handles[slot].update_observation(observation)
            if capture_rgb:
                self.collection_frames[slot] = states[slot]["image"].copy()
        return observations

    def step_slots(self, actions_by_slot):
        started = time.perf_counter()
        results = self.batch.step_slots({slot: action.value for slot, action in actions_by_slot.items()})
        elapsed = time.perf_counter() - started
        responses = {}
        for slot, result in results.items():
            metadata = {"backend": "vera20k", "slot_id": slot, "worker_pid": os.getpid(), "worker_step_wall_sec": elapsed}
            self.last_step_metadata_by_slot[slot] = metadata
            responses[slot] = EnvStepResponse(
                result=StepResult(None, result["reward"], result["terminated"], result["truncated"], result),
                worker_pid=os.getpid(), metadata=metadata,
            )
        return responses

    def close(self):
        self.batch.close()
        self.collection_frames.clear()
        self.closed = True

    @staticmethod
    def _observation(state):
        return Observation(state["image"], state["frame"], state["sim_time_ms"],
                           {key: value for key, value in state.items() if key != "image"})
