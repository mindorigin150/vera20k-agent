"""Run the real benchmark executor with fixed simulated latency and direct commands."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from latency_bench.core.types import Action, PolicyOutput
from latency_bench.executors.factory import build_executor
from latency_bench.policy.base import PolicyRunner

from vera20k_agent.benchmark import VeraEnvStepBackend


class StopPolicy(PolicyRunner):
    """Issue one stop command to an observed owned unit for the scheduler contract test."""

    def predict(self, observation):
        entity = next(item for item in observation.metadata["entities"] if item["owned"])
        return PolicyOutput(Action([{"command": "stop", "entity_id": entity["id"]}], name="stop", is_oneshot=True))

    def predict_batch(self, observations):
        return [self.predict(observation) for observation in observations]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = {
        "experiment": {"name": "vera20k-fixed-delay", "seed": 42},
        "executor": {"mode": "simulated", "parallel_envs": 1, "inference_devices": ["cpu"]},
        "env": {"name": "vera20k", "env_fps": 62.5, "obs_fps": 6.25,
                "noop_action": "noop", "action_map": {"noop": []}},
        "latency": {"method": "fixed", "fixed_latency_ms": 100},
        "scheduler": {"hold_policy": "one_frame_then_noop", "ordering_policy": "issue_order_fifo"},
        "policy": {"type": "constant", "action": "noop", "action_prefix": {"mode": "none"}},
        "evaluation": {"episodes": 1, "max_env_steps": 20},
        "logging": {"output_dir": str(args.out), "save_step_records": True,
                    "save_action_records": True, "save_latency_records": True},
    }
    backend = VeraEnvStepBackend(args.assets)
    policy = StopPolicy()
    executor = build_executor(config, env_backend=backend, policy=policy)
    try:
        metrics = executor.run_episodes(episode_ids=[0], seeds=[42], max_env_steps=20)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metrics.json").write_text(json.dumps([asdict(item) for item in metrics], indent=2))
        events = [json.loads(line) for line in (args.out / "actions.jsonl").read_text().splitlines()]
        assert [event["latency_raw_frames"] for event in events] == [7, 7]
        assert [event["apply_raw_frame"] for event in events] == [7, 17]
        steps = [json.loads(line) for line in (args.out / "steps.jsonl").read_text().splitlines()]
        receipts = [receipt for step in steps for receipt in step["info"]["commands"]]
        assert receipts == [{"submitted_frame": 7, "received_frame": 8}, {"submitted_frame": 17, "received_frame": 18}]
        print("100 ms delay: 7 waiting steps, commands received in frames 8 and 18")
    finally:
        executor.close()


if __name__ == "__main__":
    main()
