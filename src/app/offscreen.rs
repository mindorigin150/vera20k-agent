//! Window-free access to the existing client loader, frame effects and tactical renderer.
//! Rendering runs once per logic frame. Image reads never advance presentation.
use crate::agent_env::{AgentCommand, EnvConfig, EnvSlot};
use crate::app::match_bootstrap::{MatchSeedClock, MatchSeedSource};
use crate::app::{App, AppState};
use crate::render::frame_readback::PendingBgra8Readback;
use crate::render::gpu::GpuContext;
use anyhow::Result;
use std::path::Path;
use std::time::Duration;

struct EpisodeSeed(u32);
impl MatchSeedClock for EpisodeSeed {
    fn low_u32(&mut self) -> u32 {
        self.0
    }
    fn source(&self) -> MatchSeedSource {
        MatchSeedSource::NonWindowsDevelopmentFallback
    }
    fn seed_authority_certifying(&self) -> bool {
        false
    }
}

/// One slot's client presentation and offscreen targets on a shared GPU device.
pub struct OffscreenScene {
    state: AppState,
    readback: Option<PendingBgra8Readback>,
    pending_draw: Option<wgpu::CommandBuffer>,
    client_initial: serde_json::Value,
    rules_fingerprint: u64,
}

impl OffscreenScene {
    pub fn new(assets: &Path, gpu: GpuContext, config: &EnvConfig, seed: u32) -> Result<Self> {
        let mut state = App::initialize_offscreen(assets, gpu)?;
        if let Some(error) = &state.frontend.main_menu_shell_error {
            anyhow::bail!("{error}");
        }
        state.diag.use_map_presentation_clock()?;
        let seed = crate::app::match_bootstrap::read_match_seed(&mut EpisodeSeed(seed));
        crate::app::loading::pump::load_offscreen_episode(
            &mut state,
            config.session().map_err(anyhow::Error::msg)?,
            seed,
        )?;
        // Direct commands do not move an OS pointer. Keep it outside the game view.
        state.platform.pointer_in_client = false;
        state.match_state.input.cursor_x = -1024.0;
        state.match_state.input.cursor_y = -1024.0;
        let runtime = state.match_state.sim_runtime.as_ref().unwrap();
        let sim = &runtime.simulation;
        let client_initial = serde_json::json!({"state_hash": sim.state_hash(), "rng": {"scenario": sim.scenario_rng().state(), "main": sim.main_rng().state(), "mapgen": sim.mapgen_rng().state()}});
        let rules_fingerprint = runtime.resources.rules.simulation_config_hash();
        // The slot supplies the live world during each visit. Keep only a small swap placeholder.
        state.match_state.sim_runtime = Some(crate::sim::runtime::SimRuntime {
            simulation: crate::sim::world::Simulation::from_descriptor(&Default::default()),
            resources: crate::sim::runtime::SimResources::empty(),
        });
        Ok(Self {
            state,
            readback: None,
            pending_draw: None,
            client_initial,
            rules_fingerprint,
        })
    }

    /// Compare the independently loaded client world before attaching a player slot.
    pub fn initial_diagnostics(&self) -> serde_json::Value {
        self.client_initial.clone()
    }

    /// Share compatible immutable atlases and pipelines. Dynamic atlases remain slot-owned.
    pub fn share_static_assets_from(&mut self, source: &Self) {
        let compatible = self.state.match_state.loaded_map_hash
            == source.state.match_state.loaded_map_hash
            && self.state.match_state.match_presentation.house_color_map
                == source.state.match_state.match_presentation.house_color_map
            && self.rules_fingerprint == source.rules_fingerprint;
        if !compatible {
            return;
        }
        self.state
            .renderer
            .batch_renderer
            .share_pipelines_from(&source.state.renderer.batch_renderer);
        let src = &source.state.match_state.match_presentation;
        let dst = &mut self.state.match_state.match_presentation;
        dst.tile_atlas = src.tile_atlas.clone();
        dst.bridge_atlas = src.bridge_atlas.clone();
        dst.overlay_atlas = src.overlay_atlas.clone();
        dst.sidebar_cameo_atlas = src.sidebar_cameo_atlas.clone();
        dst.sidebar_chrome = src.sidebar_chrome.clone();
        dst.building_zshape = src.building_zshape.clone();
        dst.palette_set = src.palette_set.clone();
    }

    /// Draw the authoritative slot state after reset without advancing it.
    pub fn draw_slot(&mut self, slot: &mut EnvSlot) -> Result<()> {
        std::mem::swap(
            self.state.match_state.sim_runtime.as_mut().unwrap(),
            &mut slot.scenario.runtime,
        );
        let result = self.draw();
        std::mem::swap(
            self.state.match_state.sim_runtime.as_mut().unwrap(),
            &mut slot.scenario.runtime,
        );
        if result.is_ok() {
            slot.publish_camera([
                self.state.match_state.input.camera_x,
                self.state.match_state.input.camera_y,
            ]);
        }
        result
    }

    /// Run the actual client command and frame-effects path on the slot's runtime.
    pub fn step(
        &mut self,
        slot: &mut EnvSlot,
        actions: &[AgentCommand],
    ) -> Result<serde_json::Value, String> {
        for action in actions {
            if let AgentCommand::Camera { x, y } = *action {
                self.state.match_state.input.camera_x = x;
                self.state.match_state.input.camera_y = y;
            }
        }
        let result = slot.step_using(actions, |runtime| {
            let retained = self.state.match_state.sim_runtime.as_mut().unwrap();
            std::mem::swap(retained, runtime);
            let committed = crate::app::match_runtime::sim_tick::advance_one_simulation_frame(
                &mut self.state,
                crate::sim::world::TickLane::Ordinary,
                None,
            );
            let sim = &self
                .state
                .match_state
                .sim_runtime
                .as_ref()
                .unwrap()
                .simulation;
            let ended = sim
                .session()
                .current_house()
                .and_then(|owner| sim.houses().get(&owner))
                .and_then(|house| house.outcome())
                .is_some();
            let drawn = if committed || ended {
                self.draw().map_err(|e| e.to_string())
            } else {
                Err("client frame did not advance".into())
            };
            std::mem::swap(
                self.state.match_state.sim_runtime.as_mut().unwrap(),
                runtime,
            );
            drawn
        });
        if result.is_ok() {
            slot.publish_camera([
                self.state.match_state.input.camera_x,
                self.state.match_state.input.camera_y,
            ]);
        }
        result
    }

    fn draw(&mut self) -> Result<()> {
        let tick = self
            .state
            .match_state
            .sim_runtime
            .as_ref()
            .unwrap()
            .simulation
            .session()
            .tick();
        let time = tick * u64::from(crate::util::fixed_math::SIM_TICK_MS);
        let mut encoder = self.state.renderer.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("VERA offscreen frame"),
            },
        );
        let output = crate::app::presentation::render::render_game(
            &mut self.state,
            &mut encoder,
            crate::app::presentation::render::GameRenderTimes {
                radar_ms: time,
                tooltip_ms: time,
                message_ms: Some(time),
            },
        )?;
        let _ = output;
        self.pending_draw = Some(encoder.finish());
        Ok(())
    }

    /// Transfer one encoded draw to the batch queue owner.
    pub fn take_draw(&mut self) -> Option<wgpu::CommandBuffer> {
        self.pending_draw.take()
    }

    /// Add this slot's image copy to a batch submission, reusing its readback buffer.
    pub fn encode_readback(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        let gpu = &self.state.renderer.gpu;
        let texture = self
            .state
            .renderer
            .combat_light_renderer
            .composition_texture();
        if let Some(pending) = &self.readback {
            pending.encode_copy(encoder, texture);
        } else {
            self.readback = Some(PendingBgra8Readback::encode(
                &gpu.device,
                encoder,
                texture,
                texture.format(),
                gpu.config.width,
                gpu.config.height,
            )?);
        }
        Ok(())
    }

    /// Map one completed batch copy and return packed RGB8 pixels.
    pub fn finish_readback(&self, submission: wgpu::SubmissionIndex) -> Result<Vec<u8>> {
        let bgra = self
            .readback
            .as_ref()
            .expect("readback was encoded")
            .finish(
                &self.state.renderer.gpu.device,
                submission,
                Duration::from_secs(60),
            )?;
        let mut rgb = Vec::with_capacity(bgra.len() / 4 * 3);
        for pixel in bgra.chunks_exact(4) {
            rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
        }
        Ok(rgb)
    }
}
