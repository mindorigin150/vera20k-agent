# VERA20k agent environments

The package runs a complete 1v1 skirmish through VERA's existing rules, AI, command executor and renderer.
The default match uses `XMP03T4.MAP`, America against Russia, Normal AI and GameSpeed 1.
One raw step advances one logic frame and adds 16 ms. The default observation interval is ten raw steps (6.25 Hz).
Rules still convert production and weapon durations using the original 15-frame second.

## Install

Use the repository's Rust `stable` toolchain and Python 3.12.
Linux builds need a C/C++ compiler, pkg-config, ALSA, udev, X11, Wayland and Vulkan development packages.
GPU rendering requires a Vulkan GPU. `gpu_index` selects its index in Vulkan's adapter list.
Software Vulkan adapters produce an error.

```sh
python3.12 -m venv /path/to/vera-venv
. /path/to/vera-venv/bin/activate
pip install maturin numpy gymnasium pillow pytest
maturin develop --release --manifest-path python/Cargo.toml
```

For a distributable wheel, run `maturin build --release --manifest-path python/Cargo.toml`.
Keep game assets outside this repository. This helper downloads the two XWIS archives into an external directory:

```sh
python python/tools/prepare_assets.py /path/to/vera-assets
```

The helper copies nonempty MIX files and records archive URLs, sizes and omitted placeholders.
The explicit `Multiplayer` media mode requires the game archives, rules, maps and images.
It skips campaign movie archives. The ordinary client's media mode keeps its existing requirements.

## Gymnasium

```python
from vera20k_agent import VeraEnv

env = VeraEnv('/path/to/vera-assets/game', threads=4)
observation, info = env.reset(seed=42)
mcv = next(e['id'] for e in observation['entities']
           if e['owned'] and e['type_name'] == 'AMCV')
observation, reward, terminated, truncated, info = env.step([
    {'command': 'deploy', 'entity_id': mcv},
])
image = env.render()  # uint8 RGB, shape (600, 800, 3)
env.close()
```

`step()` returns the current frame's image and player knowledge.
`reset(seed=..., options=...)` accepts episode configuration overrides.
The constructor configuration applies again when a later reset omits those overrides.
Victory gives +1, defeat gives -1, and other steps give zero.
The native House result terminates an episode. The default 112,500-step limit truncates it after 30 simulated minutes.
The caller must reset after either result. `close()` is idempotent.

An observation contains owned objects, visible objects, credits, power, production, build options and visible resource cells.
Enemy objects under shroud or undetected cloak are omitted. Disguised objects do not reveal their true identity.
Object IDs stay stable within an episode. Type names use the loaded rules.
Images include the player's shroud and sidebar. Direct commands keep the OS pointer outside the view.
Gym observations and info omit the episode counter so seeded resets remain reproducible.
The batch interface carries episode counters separately.

`import vera20k_agent` also registers `VERA20k-v0` for `gymnasium.make(..., assets=...)`.

## Commands

An action is an ordered list. An empty list advances one frame without player commands.
The environment binds the local player and the next execution frame.
It preserves native command rejection when a game condition prevents an action.
The player boundary applies the client input lock and ignores unavailable or foreign actors and object targets outside player knowledge.
Unknown and hidden object references return the same no-op behavior.
Malformed commands and unknown type names raise an error before any requested slot advances.

| command | fields |
| --- | --- |
| `move`, `attack_move`, `harvest` | `entity_id`, `rx`, `ry` |
| `attack`, `enter`, `capture` | `entity_id`, `target_id` |
| `stop`, `deploy`, `undeploy`, `unload`, `repair`, `sell` | `entity_id` |
| `produce` | `type_name` |
| `place`, `super_weapon` | `type_name`, `rx`, `ry` |
| `cancel` | `type_name`, `all` |
| `suspend` | `category`: `Building`, `Defense`, `Infantry`, `Vehicle`, `Aircraft` or `Ship` |
| `camera` | world-screen coordinates `x`, `y` for the camera top-left |

`deploy` dispatches the existing infantry or vehicle deploy command for the specified object. `repair` toggles building repair.
`enter` uses the existing Enter mission for transport or building entry.
Game speed is fixed for evaluation and cannot appear in an action.
A command submitted at logical frame f enters the client executor during f+1.
The result records `submitted_frame` and `received_frame`. A received command can still fail its native game conditions.

`python/examples/default.json` records the default episode configuration. Pass its JSON object as the constructor's `config` argument.

Configuration fields are `map`, `player_country`, `opponent_country`, `difficulty`, `player_start`, `opponent_start` and `max_steps`.
Countries accept existing engine names such as `America` and `Russia` or their rule indices.
Difficulty accepts `Easy`, `Normal` and `Hard`. Starts default to 0 and 1.
The `image` field always describes the returned logical frame.
The `camera` field reports the actual camera top-left after the native bounds clamp.

## Batch simulation and images

```python
from vera20k_agent import VeraBatch

batch = VeraBatch('/path/to/vera-assets/game', 8, threads=8, gpu_index=0)
batch.reset_slots({slot: slot for slot in range(8)})
results = batch.step_slots({0: [], 3: [{'command': 'produce', 'type_name': 'GAPOWR'}]})
observations = batch.observe_slots([3, 0])
frames = batch.render_slots([3, 0])
assert frames['images'].shape == (2, 600, 800, 3)
batch.close()
```

Only requested slots advance or reset. Every slot owns its world, RNG, lifecycle, camera and dynamic render resources.
One bounded Rayon pool handles simulation, nested simulation work and client loading.
Compatible static atlases and sprite pipelines share GPU resources.
Image copies use one batch submission and reusable per-slot readback buffers.
`render_slots()` returns contiguous RGB arrays with slot, episode and frame arrays in requested order.

Presentation updates and drawing run once per logic frame. Image reads never update presentation.
The backend submits the requested slots' encoded draws together and batches image copies and readback.
`render=False` measures the GPU-free simulation path and omits images.
That path still runs the native CPU beam updates because laser, bolt and designation geometry consumes the process Main RNG.
Both paths advance presentation once per logic frame, including all three RNG streams.
Use one process per GPU for multiple GPUs. Each process selects its own `gpu_index`.

## Benchmark integration

`vera20k_agent.benchmark.VeraEnvStepBackend` implements the existing `latency_bench` protocol.
Import this optional module only in an environment with `latency_bench` available.

```python
from latency_bench.executors.factory import build_executor
from vera20k_agent.benchmark import VeraEnvStepBackend

backend = VeraEnvStepBackend('/path/to/vera-assets/game', num_slots=8, threads=8)
executor = build_executor(config, env_backend=backend, policy=policy)
```

Set `env.env_fps=62.5`, `env.obs_fps=6.25`, `noop_action=[]` and `hold_policy=one_frame_then_noop`.
The benchmark owns observation cadence, latency and reply ordering.
The environment adds no input delay. A fixed 100 ms delay waits seven raw steps.
A request issued at frame 0 becomes ready at raw frame 7 and enters the next environment step, logical frame 8.
The backend's `step_slots()` does not read images. `observe_slots()` reads them when requested.

## Validation and evidence

```sh
VERA_ASSETS=/path/to/vera-assets/game python -m pytest python/tests -q
python python/examples/benchmark_smoke.py --assets /path/to/vera-assets/game --out /path/to/new-run
python python/examples/skirmish.py --assets /path/to/vera-assets/game --out /path/to/new-match --video
python python/examples/performance.py --assets /path/to/vera-assets/game --out /path/to/perf.json --slots 8 --render
```

The package tests own the environment lifecycle, replay, clock, pixel and isolation contracts.
They require external assets and a Vulkan GPU for rendered cases.
The private native diagnostic methods expose state and RNG fingerprints only to comparison tools.
They never populate policy observations or benchmark metadata.
These comparisons measure consistency between VERA execution paths. They do not certify parity with `gamemd.exe`.
The scripted match and fixed-delay smoke test are new simulations. They contain no model calls or paid API evidence.
Performance reports use release builds and identify the server's shared GPU load.
