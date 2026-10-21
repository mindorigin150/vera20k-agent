//! GPU-free presentation cadence for the agent's headless execution path.
//! Native laser, bolt and designation geometry consumes Main RNG. Reuse those
//! producers every frame so CPU simulation and GPU drawing retain the same cursor.
use crate::agent_env::{AgentCommand, EnvConfig, EnvSlot};
use crate::app::presentation::{airstrike_lines, electric_bolts, lasers};
use crate::render::electric_bolt::ElectricBoltPalette;
use crate::render::surface_line::SurfaceLineViewport;
use crate::sim::world::{FrameEffects, LifecycleOutput, TickLane};
use anyhow::{Context, Result};
use std::path::Path;

/// One headless slot's native beam lifetimes, camera and Main RNG consumers.
pub struct CpuScene {
    lasers: lasers::Lasers,
    bolts: electric_bolts::ElectricBolts,
    palette: ElectricBoltPalette,
    colors: crate::map::houses::HouseColorMap,
    camera: (f32, f32),
    viewport: (u32, u32),
}

impl CpuScene {
    pub fn new(
        assets: &Path,
        slot: &mut EnvSlot,
        config: &EnvConfig,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let runtime = &slot.scenario.runtime;
        let sim = &runtime.simulation;
        let rules = &runtime.resources.rules;
        let manager = crate::assets::asset_manager::AssetManager::new(
            assets,
            crate::assets::asset_manager::MediaArchiveMode::Multiplayer,
        )?;
        let palette = crate::assets::pal_file::Palette::from_bytes(
            manager
                .get_ref("PALETTE.PAL")
                .context("PALETTE.PAL is required")?,
        )?;
        let roster = crate::map::houses::parse_house_roster(
            &slot.scenario.map.ini,
            &rules.color_schemes,
            Some(rules),
        );
        let colors = crate::app::frontend::skirmish::house_color_map_for_launch_session(
            &config.session().map_err(anyhow::Error::msg)?,
            &roster,
            &rules.color_schemes,
        );
        let viewport = crate::ui::sidebar::SidebarFrame::native(width, height).tactical_size();
        let owner = sim
            .session()
            .current_house()
            .map(|owner| sim.interner().resolve(owner));
        let anchor = crate::app::input::camera::initial_camera_anchor(
            Some(sim),
            owner,
            slot.scenario.opening_view_cell,
            sim.resolved_terrain()
                .context("resolved terrain is required")?,
            &slot.scenario.map,
        );
        let mut camera = crate::app::input::camera::tactical_camera_top_left(
            anchor,
            viewport.0 as f32,
            viewport.1 as f32,
            1.0,
        );
        if let Some(bounds) = runtime.view().playfield_authority().0 {
            camera = crate::app::input::camera::clamp_camera_point_to_local_bounds(
                camera,
                bounds,
                (viewport.0 as f32, viewport.1 as f32),
                1.0,
            );
        }
        slot.publish_camera([camera.0, camera.1]);
        Ok(Self {
            lasers: Default::default(),
            bolts: Default::default(),
            palette: ElectricBoltPalette::from_palette(&palette),
            colors,
            camera,
            viewport,
        })
    }

    pub fn step(
        &mut self,
        slot: &mut EnvSlot,
        actions: &[AgentCommand],
    ) -> Result<serde_json::Value, String> {
        for action in actions {
            if let AgentCommand::Camera { x, y } = *action {
                self.camera = (x, y);
            }
        }
        let result = slot.step_using(actions, |runtime| {
            let due = runtime.simulation.take_due_commands();
            let output = runtime
                .advance_frame(
                    &due,
                    crate::util::fixed_math::SIM_TICK_MS,
                    TickLane::Ordinary,
                    FrameEffects::default(),
                )
                .map_err(|e| e.to_string())?;
            let sim = &runtime.simulation;
            let rules = &runtime.resources.rules;
            for event in output.lifecycle_outputs {
                match event {
                    LifecycleOutput::LaserUpdate { frame } => self.lasers.update(frame),
                    LifecycleOutput::LaserCreated(birth) => self.lasers.create(birth, |owner| {
                        lasers::house_rgb(sim, rules, &self.colors, owner)
                    }),
                    LifecycleOutput::ElectricBoltCreated(birth) => self.bolts.create(birth),
                    _ => {}
                }
            }
            if let Some(bounds) = runtime.view().playfield_authority().0 {
                self.camera = crate::app::input::camera::clamp_camera_point_to_local_bounds(
                    self.camera,
                    bounds,
                    (self.viewport.0 as f32, self.viewport.1 as f32),
                    1.0,
                );
            }
            self.bolts.follow_owners(|owner| {
                crate::sim::combat::electric_bolt::owner_source(sim, rules, owner)
            });
            let view = SurfaceLineViewport {
                camera: [self.camera.0.floor() as i32, self.camera.1.floor() as i32],
                clip: [0, 0, self.viewport.0 as i32, self.viewport.1 as i32],
                z_origin_y: 0,
                zoom: 1.0,
            };
            let mut main = sim.presentation_main_draws();
            self.lasers.composite(&mut main);
            self.bolts.composite(view, Some(self.palette), &mut main);
            let designations: Vec<_> = sim
                .entities()
                .iter_sorted()
                .filter(|(_, entity)| entity.health.current() != 0)
                .filter_map(|(id, _)| {
                    sim.airstrike_designation_coordinates(id, rules)
                        .map(airstrike_lines::TacticalLineCall::Designation)
                })
                .collect();
            let _ = airstrike_lines::composite(
                &designations,
                sim.session().total_sim_ms() as u32,
                &main,
                view,
            );
            Ok(())
        });
        if result.is_ok() {
            slot.publish_camera([self.camera.0, self.camera.1]);
        }
        result
    }
}
