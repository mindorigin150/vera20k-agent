//! Python boundary. JSON validation happens here, simulation runs without the GIL.
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::path::PathBuf;
use std::sync::Mutex;
use vera20k::app::cpu_scene::CpuScene;
use vera20k::app::offscreen::OffscreenScene;
use vera20k::render::gpu::GpuContext;
use vera20k_core::agent_env::{AgentCommand, BatchEnv, EnvConfig, EnvSlot};

fn input_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// Own the bounded Rust simulation pool and its independent episodes.
#[pyclass]
struct NativeBatch {
    batch: BatchEnv,
    assets: PathBuf,
    width: u32,
    height: u32,
    cpu_scenes: Vec<Mutex<Option<CpuScene>>>,
    gpu: Option<GpuContext>,
    scenes: Vec<Mutex<Option<OffscreenScene>>>,
}

impl NativeBatch {
    fn submit_draws(&self, ids: impl IntoIterator<Item = usize>) {
        if let Some(gpu) = &self.gpu {
            let draws: Vec<_> = ids
                .into_iter()
                .filter_map(|id| {
                    self.scenes[id]
                        .lock()
                        .unwrap()
                        .as_mut()
                        .unwrap()
                        .take_draw()
                })
                .collect();
            gpu.queue.submit(draws);
        }
    }
}

#[pymethods]
impl NativeBatch {
    #[new]
    fn new(
        assets: PathBuf,
        config: &str,
        slots: usize,
        threads: usize,
        render: bool,
        width: u32,
        height: u32,
        gpu_index: usize,
    ) -> PyResult<Self> {
        let config: EnvConfig = serde_json::from_str(config).map_err(input_error)?;
        let gpu = if render {
            Some(GpuContext::new_offscreen(width, height, gpu_index).map_err(runtime_error)?)
        } else {
            None
        };
        Ok(Self {
            batch: BatchEnv::new(assets.clone(), config, slots, threads).map_err(input_error)?,
            assets,
            gpu,
            width,
            height,
            cpu_scenes: (0..slots).map(|_| Mutex::new(None)).collect(),
            scenes: (0..slots).map(|_| Mutex::new(None)).collect(),
        })
    }

    fn reset_slots(&mut self, py: Python<'_>, requests: &str) -> PyResult<String> {
        let requests: Vec<(usize, u32, u64)> =
            serde_json::from_str(requests).map_err(input_error)?;
        let candidates = py
            .detach(|| self.batch.prepare_reset(&requests))
            .map_err(runtime_error)?;
        let mut prepared: Vec<(usize, EnvSlot, Option<OffscreenScene>, Option<CpuScene>)> =
            Vec::new();
        for (id, mut slot) in candidates {
            let (gpu_scene, cpu_scene) = if let Some(gpu) = &self.gpu {
                let seed = slot.scenario.sim().session().seed() as u32;
                let mut scene = self
                    .batch
                    .in_pool(|| {
                        OffscreenScene::new(
                            &self.assets,
                            gpu.shared_offscreen(),
                            &self.batch.config,
                            seed,
                        )
                    })
                    .map_err(runtime_error)?;
                if let Some((_, _, Some(source), _)) = prepared.first() {
                    scene.share_static_assets_from(source);
                } else {
                    for (other_id, other) in self.scenes.iter().enumerate() {
                        if other_id == id {
                            continue;
                        }
                        let other = other.lock().unwrap();
                        if let Some(other) = other.as_ref() {
                            scene.share_static_assets_from(other);
                            break;
                        }
                    }
                }
                scene.draw_slot(&mut slot).map_err(runtime_error)?;
                (Some(scene), None)
            } else {
                let scene = self
                    .batch
                    .in_pool(|| {
                        CpuScene::new(
                            &self.assets,
                            &mut slot,
                            &self.batch.config,
                            self.width,
                            self.height,
                        )
                    })
                    .map_err(runtime_error)?;
                (None, Some(scene))
            };
            prepared.push((id, slot, gpu_scene, cpu_scene));
        }
        for (id, slot, gpu_scene, cpu_scene) in prepared {
            self.batch.slots[id] = Some(slot);
            *self.scenes[id].lock().unwrap() = gpu_scene;
            *self.cpu_scenes[id].lock().unwrap() = cpu_scene;
        }
        self.submit_draws(requests.iter().map(|r| r.0));
        let result = self
            .batch
            .observe_slots(&requests.iter().map(|r| r.0).collect::<Vec<_>>())
            .map_err(runtime_error)?;
        serde_json::to_string(&result).map_err(runtime_error)
    }

    fn step_slots(&mut self, py: Python<'_>, requests: &str) -> PyResult<String> {
        let requests: Vec<(usize, Vec<AgentCommand>)> =
            serde_json::from_str(requests).map_err(input_error)?;
        let result = if self.gpu.is_some() {
            let scenes = &self.scenes;
            py.detach(|| {
                self.batch.step_slots_using(&requests, |id, slot, actions| {
                    scenes[id]
                        .lock()
                        .unwrap()
                        .as_mut()
                        .unwrap()
                        .step(slot, actions)
                })
            })
        } else {
            let scenes = &self.cpu_scenes;
            py.detach(|| {
                self.batch.step_slots_using(&requests, |id, slot, actions| {
                    scenes[id]
                        .lock()
                        .unwrap()
                        .as_mut()
                        .unwrap()
                        .step(slot, actions)
                })
            })
        }
        .map_err(runtime_error)?;
        self.submit_draws(requests.iter().map(|r| r.0));
        serde_json::to_string(&result).map_err(runtime_error)
    }

    fn observe_slots(&self, ids: Vec<usize>) -> PyResult<String> {
        let result = self.batch.observe_slots(&ids).map_err(runtime_error)?;
        serde_json::to_string(&result).map_err(runtime_error)
    }

    /// Internal comparison output. This method is never used to construct observations.
    fn _diagnostics(&self, id: usize) -> PyResult<String> {
        let slot = self
            .batch
            .slots
            .get(id)
            .and_then(Option::as_ref)
            .ok_or_else(|| input_error("slot is not initialized"))?;
        let sim = slot.scenario.sim();
        Ok(self.batch.in_pool(|| serde_json::json!({"state_hash": sim.state_hash(), "rng": serde_json::json!({"scenario": sim.scenario_rng().state(), "main": sim.main_rng().state(), "mapgen": sim.mapgen_rng().state()}), "frame": sim.session().tick(), "sim_time_ms": sim.session().total_sim_ms()}).to_string()))
    }

    fn render_slots<'py>(
        &mut self,
        py: Python<'py>,
        ids: Vec<usize>,
    ) -> PyResult<Vec<(usize, Bound<'py, PyBytes>)>> {
        self.batch.validate_ids(&ids, true).map_err(runtime_error)?;
        if self.gpu.is_none() {
            return Err(runtime_error("RGB rendering is disabled"));
        }
        let gpu = self.gpu.as_ref().unwrap();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for &id in &ids {
            self.scenes[id]
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .encode_readback(&mut encoder)
                .map_err(runtime_error)?;
        }
        let submission = gpu.queue.submit([encoder.finish()]);
        let mut images = Vec::new();
        for id in ids {
            let bytes = self.scenes[id]
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .finish_readback(submission.clone())
                .map_err(runtime_error)?;
            images.push((id, PyBytes::new(py, &bytes)));
        }
        Ok(images)
    }

    fn _client_diagnostics(&self, id: usize) -> PyResult<String> {
        let scene = self
            .scenes
            .get(id)
            .ok_or_else(|| input_error("invalid slot"))?;
        self.batch.in_pool(|| {
            let scene = scene.lock().unwrap();
            let scene = scene
                .as_ref()
                .ok_or_else(|| input_error("scene is not initialized"))?;
            Ok(scene.initial_diagnostics().to_string())
        })
    }

    fn close(&mut self) {
        self.batch.close();
        self.scenes.clear();
        self.cpu_scenes.clear();
        self.gpu = None;
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeBatch>()?;
    m.add("FRAME_MS", vera20k_core::util::fixed_math::SIM_TICK_MS)?;
    Ok(())
}
