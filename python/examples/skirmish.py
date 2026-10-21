"""Scripted acceptance match. Records direct commands, replay hashes and optional video."""
import argparse
import json
from pathlib import Path
import subprocess

from PIL import Image

from vera20k_agent import VeraBatch


def opening(frame, initial, state):
    mcv = next(e for e in initial["entities"] if e["owned"] and e["type_name"] == "AMCV")
    rx, ry = mcv["rx"], mcv["ry"]
    if frame == 1:
        return [{"command": "deploy", "entity_id": mcv["id"]}]
    if frame % 10:
        return []
    owned = {entity["type_name"] for entity in state["entities"] if entity["owned"]}
    for type_name in ("GAPOWR", "GAREFN", "GAPILE", "GAWEAP"):
        if type_name in owned:
            continue
        queued = [item for item in state["production"] if item["type_name"] == type_name]
        if any(item["state"] == "Done" for item in queued):
            # Exercise native placement rejection, then the first legal requested cell.
            return [{"command": "place", "type_name": type_name, "rx": rx + x, "ry": ry + y}
                    for x in range(-9, 10) for y in range(-9, 10)]
        if not queued and any(item["type_name"] == type_name and item["enabled"] for item in state["build_options"]):
            return [{"command": "produce", "type_name": type_name}]
        return []
    if frame % 300 == 0:
        return [{"command": "produce", "type_name": "MTNK"}, {"command": "produce", "type_name": "E1"}]
    if frame == 4000:
        return [{"command": "move", "entity_id": e["id"], "rx": rx + 5, "ry": ry + 5}
                for e in state["entities"] if e["owned"] and e["type_name"] == "MTNK"]
    if frame >= 4500 and frame % 300 == 10:
        # XMP03T4 has public, fixed start positions. This is a diagnostic controller.
        return [{"command": "attack_move", "entity_id": e["id"], "rx": 88, "ry": 31}
                for e in state["entities"] if e["owned"] and e["type_name"] in ("MTNK", "E1")]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    batch = VeraBatch(args.assets, args.slots, render=args.video, threads=4, gpu_index=args.gpu_index)
    initial = batch.reset_slots({slot: args.seed + slot for slot in range(args.slots)})[0]
    active = set(range(args.slots))
    state = initial
    video = None
    if args.video:
        video = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", "800x600", "-framerate", "6.25", "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-threads", "2", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", str(args.out / "replay.mp4"),
        ], stdin=subprocess.PIPE)
        Image.fromarray(initial["image"]).save(args.out / "initial.png")
    replay = args.trace.open() if args.trace else None
    credits = [initial["credits"]]
    observed_types = {e["type_name"] for e in initial["entities"] if e["owned"]}
    enemy_seen = False
    initial_ids = {e["id"] for e in initial["entities"] if e["owned"]}
    new_objects = {}
    health = {e["id"]: e["health"] for e in initial["entities"] if e["owned"]}
    damage_events = 0
    income = []
    result = None
    try:
        with (args.out / "trace.jsonl").open("w") as trace:
            for frame in range(1, 112501):
                reference = json.loads(next(replay)) if replay else None
                actions = reference["actions"] if reference else opening(frame, initial, state)
                results = batch.step_slots({slot: actions if slot == 0 else [] for slot in active})
                result = results[0]
                active.difference_update(slot for slot, value in results.items() if value["terminated"] or value["truncated"])
                diagnostic = json.loads(batch.native._diagnostics(0))
                if reference:
                    if diagnostic != reference["diagnostic"]:
                        (args.out / "mismatch.json").write_text(json.dumps({"frame": frame, "reference": reference["diagnostic"], "actual": diagnostic}, indent=2))
                        raise AssertionError(f"replay differs at frame {frame}")
                trace.write(json.dumps({"frame": frame, "actions": actions, "diagnostic": diagnostic}) + "\n")
                if frame % 10 == 0 or result["terminated"] or result["truncated"]:
                    state = batch.observe_slots([0])[0]
                    if state["credits"] > credits[-1] and len(income) < 20:
                        income.append({"frame": frame, "increase": state["credits"] - credits[-1]})
                    credits.append(state["credits"])
                    for entity in state["entities"]:
                        if not entity["owned"]:
                            continue
                        if entity["id"] not in initial_ids:
                            new_objects[entity["id"]] = entity["type_name"]
                        if entity["health"] < health.get(entity["id"], entity["health"]):
                            damage_events += 1
                        health[entity["id"]] = entity["health"]
                    observed_types.update(e["type_name"] for e in state["entities"] if e["owned"])
                    enemy_seen |= any(e["relation"] == "enemy" for e in state["entities"])
                    if video:
                        video.stdin.write(state["image"].tobytes())
                    if frame in (2000, 4500) and video:
                        Image.fromarray(state["image"]).save(args.out / f"frame-{frame}.png")
                if frame % 1000 == 0:
                    print(json.dumps({"frame": frame, "credits": state["credits"], "types": sorted(observed_types)}), flush=True)
                if result["terminated"] or result["truncated"]:
                    break
        summary = {"evidence": "command_replay" if args.trace else "new_simulation", "reference_trace": str(args.trace) if args.trace else None, "controller": "scripted_acceptance", "seed": args.seed,
                   "map": "XMP03T4.MAP", "game_speed": 1, "result": result, "types": sorted(observed_types),
                   "min_credits": min(credits), "max_credits": max(credits), "enemy_seen": enemy_seen,
                   "observed_credit_increases": income, "observed_damage_events": damage_events,
                   "new_objects": {kind: sum(value == kind for value in new_objects.values()) for kind in sorted(set(new_objects.values()))},
                   "slots": args.slots, "gpu_index": args.gpu_index if args.video else None}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        batch.close()
        if replay:
            replay.close()
        if video:
            video.stdin.close()
            if video.wait() != 0:
                raise RuntimeError("video encoding failed")


if __name__ == "__main__":
    main()
