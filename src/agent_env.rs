//! Player-bound, one-frame environment over the production simulation runtime.
//! Policy observations contain only player knowledge. Diagnostics remain separate.

use crate::headless_scenario::{self, HeadlessScenario};
use crate::sim::combat::TargetKind;
use crate::sim::command::{Command, CommandEnvelope};
use crate::sim::house_state::HouseOutcomeKind;
use crate::sim::intern::InternedId;
use crate::sim::mission::MissionType;
use crate::sim::production::{self, ProductionCategory};
use crate::skirmish_launch::{
    AiDifficulty, LaunchCountry, LaunchStartPosition, SkirmishLaunchSession,
};
use crate::util::fixed_math::SIM_TICK_MS;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Resolved episode setup. The environment does not accept speed-changing actions.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EnvConfig {
    pub map: String,
    pub player_country: LaunchCountry,
    pub opponent_country: LaunchCountry,
    pub difficulty: AiDifficulty,
    pub player_start: u8,
    pub opponent_start: u8,
    pub max_steps: u64,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            map: "XMP03T4.MAP".into(),
            player_country: LaunchCountry::America,
            opponent_country: LaunchCountry::Russia,
            difficulty: AiDifficulty::Normal,
            player_start: 0,
            opponent_start: 1,
            max_steps: 30 * 60 * 1000 / SIM_TICK_MS as u64,
        }
    }
}

impl EnvConfig {
    pub fn session(&self) -> Result<SkirmishLaunchSession, String> {
        if self.player_start == self.opponent_start || self.max_steps == 0 {
            return Err("starts must differ and max_steps must be positive".into());
        }
        let mut session = headless_scenario::battle_session(
            &self.map,
            &[(self.opponent_country, self.difficulty)],
        );
        session.humans[0].slot.country = self.player_country;
        session.humans[0].slot.start_position = LaunchStartPosition::Position(self.player_start);
        session.opponents[0].start_position = LaunchStartPosition::Position(self.opponent_start);
        session.options.game_speed = 1;
        Ok(session)
    }
}

/// Ordered direct commands. IDs are stable for one episode, names come from rules.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case", deny_unknown_fields)]
pub enum AgentCommand {
    Move { entity_id: u64, rx: u16, ry: u16 },
    Attack { entity_id: u64, target_id: u64 },
    AttackMove { entity_id: u64, rx: u16, ry: u16 },
    Stop { entity_id: u64 },
    Deploy { entity_id: u64 },
    Undeploy { entity_id: u64 },
    Harvest { entity_id: u64, rx: u16, ry: u16 },
    Enter { entity_id: u64, target_id: u64 },
    Unload { entity_id: u64 },
    Capture { entity_id: u64, target_id: u64 },
    Repair { entity_id: u64 },
    Sell { entity_id: u64 },
    Produce { type_name: String },
    Place { type_name: String, rx: u16, ry: u16 },
    Cancel { type_name: String, all: bool },
    Suspend { category: ProductionCategory },
    SuperWeapon { type_name: String, rx: u16, ry: u16 },
    Camera { x: f32, y: f32 },
}

impl AgentCommand {
    fn actor_id(&self) -> Option<u64> {
        match *self {
            Self::Move { entity_id, .. }
            | Self::Attack { entity_id, .. }
            | Self::AttackMove { entity_id, .. }
            | Self::Stop { entity_id }
            | Self::Deploy { entity_id }
            | Self::Undeploy { entity_id }
            | Self::Harvest { entity_id, .. }
            | Self::Enter { entity_id, .. }
            | Self::Unload { entity_id }
            | Self::Capture { entity_id, .. }
            | Self::Repair { entity_id }
            | Self::Sell { entity_id } => Some(entity_id),
            _ => None,
        }
    }

    fn target_id(&self) -> Option<u64> {
        match *self {
            Self::Attack { target_id, .. }
            | Self::Enter { target_id, .. }
            | Self::Capture { target_id, .. } => Some(target_id),
            _ => None,
        }
    }

    fn payload(&self, scenario: &HeadlessScenario) -> Result<Option<Command>, String> {
        let name_id = |name: &str| {
            scenario
                .sim()
                .interner()
                .get(name)
                .ok_or_else(|| format!("unknown rules type: {name}"))
        };
        Ok(Some(match *self {
            Self::Move { entity_id, rx, ry } => Command::Move {
                entity_id,
                target_rx: rx,
                target_ry: ry,
            },
            Self::Attack {
                entity_id,
                target_id,
            } => Command::Attack {
                attacker_id: entity_id,
                target_id,
            },
            Self::AttackMove { entity_id, rx, ry } => Command::AttackMove {
                entity_id,
                target: None,
                destination: Some(TargetKind::Cell(rx, ry)),
            },
            Self::Stop { entity_id } => Command::Stop { entity_id },
            Self::Deploy { entity_id } => {
                if scenario
                    .sim()
                    .entities()
                    .get(entity_id)
                    .is_some_and(|entity| {
                        entity.category() == crate::map::entities::EntityCategory::Infantry
                    })
                {
                    Command::ToggleInfantryDeploy { entity_id }
                } else {
                    Command::DeployMcv { entity_id }
                }
            }
            Self::Undeploy { entity_id } => Command::UndeployBuilding { entity_id },
            Self::Harvest { entity_id, rx, ry } => Command::MegaMission {
                entity_id,
                mission: MissionType::Harvest,
                target: None,
                destination: Some(TargetKind::Cell(rx, ry)),
            },
            Self::Enter {
                entity_id,
                target_id,
            } => Command::MegaMission {
                entity_id,
                mission: MissionType::Enter,
                target: None,
                destination: Some(TargetKind::Entity(target_id)),
            },
            Self::Unload { entity_id } => Command::UnloadPassengers {
                transport_id: entity_id,
            },
            Self::Capture {
                entity_id,
                target_id,
            } => Command::CaptureBuilding {
                engineer_id: entity_id,
                target_building_id: target_id,
            },
            Self::Repair { entity_id } => Command::ToggleRepair { entity_id },
            Self::Sell { entity_id } => Command::SellBuilding { entity_id },
            Self::Produce { ref type_name } => Command::QueueProduction {
                type_id: name_id(type_name)?,
            },
            Self::Place {
                ref type_name,
                rx,
                ry,
            } => Command::PlaceReadyBuilding {
                type_id: name_id(type_name)?,
                rx,
                ry,
            },
            Self::Cancel { ref type_name, all } => Command::CancelProductionByType {
                type_id: name_id(type_name)?,
                all,
            },
            Self::Suspend { category } => Command::SuspendProduction { category },
            Self::SuperWeapon {
                ref type_name,
                rx,
                ry,
            } => Command::LaunchSuperWeapon {
                sw_type_id: name_id(type_name)?,
                target_rx: rx,
                target_ry: ry,
            },
            Self::Camera { x, y } => {
                if !x.is_finite() || !y.is_finite() {
                    return Err("camera coordinates must be finite".into());
                }
                return Ok(None);
            }
        }))
    }
}

/// One episode's runtime, lifecycle and player camera. Contains no latency scheduler.
pub struct EnvSlot {
    pub scenario: HeadlessScenario,
    owner: InternedId,
    camera: Option<[f32; 2]>,
    episode_id: u64,
    max_steps: u64,
    ended: bool,
}

impl EnvSlot {
    pub fn load(
        assets: &Path,
        config: &EnvConfig,
        seed: u32,
        episode_id: u64,
    ) -> Result<Self, String> {
        let launch =
            crate::sim::scenario_bootstrap::MatchLaunchDescriptor::from_resolved(config.session()?)
                .map_err(|e| e.to_string())?;
        let scenario = headless_scenario::load_with_launch_and_media(
            assets,
            &config.map,
            seed,
            launch,
            &Default::default(),
            crate::assets::asset_manager::MediaArchiveMode::Multiplayer,
        )?;
        let owner = scenario
            .sim()
            .session()
            .current_house()
            .ok_or("launch has no current player")?;
        Ok(Self {
            scenario,
            owner,
            camera: None,
            episode_id,
            max_steps: config.max_steps,
            ended: false,
        })
    }

    /// Convert the whole action before mutating the episode. Invalid syntax is atomic.
    fn prepare(&self, actions: &[AgentCommand]) -> Result<Vec<CommandEnvelope>, String> {
        if self.ended {
            return Err("episode ended; reset the slot before stepping".into());
        }
        let frame = self.scenario.sim().session().tick() + 1;
        actions
            .iter()
            .filter_map(|action| match action.payload(&self.scenario) {
                Ok(Some(payload)) => {
                    // Native EventClass trusts the selected actor. This boundary replaces the UI's selection/visibility gates.
                    let actor_unavailable = action.actor_id().is_some_and(|id| {
                        !self
                            .scenario
                            .sim()
                            .entities()
                            .get(id)
                            .is_some_and(|entity| entity.owner() == self.owner)
                    });
                    let target_unavailable = action.target_id().is_some_and(|id| {
                        !self
                            .scenario
                            .sim()
                            .entities()
                            .get(id)
                            .is_some_and(|entity| self.observable(entity))
                    });
                    if actor_unavailable
                        || target_unavailable
                        || self.scenario.sim().session().input_locked()
                    {
                        None
                    } else {
                        Some(Ok(CommandEnvelope::new(self.owner, frame, payload)))
                    }
                }
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            })
            .collect()
    }

    /// Share lifecycle and command binding with the client presentation path.
    pub fn step_using(
        &mut self,
        actions: &[AgentCommand],
        advance: impl FnOnce(&mut crate::sim::runtime::SimRuntime) -> Result<(), String>,
    ) -> Result<Value, String> {
        let commands = self.prepare(actions)?;
        let submitted_frame = self.scenario.sim().session().tick();
        let runtime = &mut self.scenario.runtime;
        runtime.simulation.queue_commands(commands.clone());
        advance(runtime)?;
        let sim = self.scenario.sim();
        let outcome = sim.houses().get(&self.owner).and_then(|h| h.outcome());
        let terminated = outcome.is_some();
        let truncated = !terminated && sim.session().tick() >= self.max_steps;
        self.ended = terminated || truncated;
        let reward = outcome.map_or(0, |o| match o.kind {
            HouseOutcomeKind::Victory => 1,
            HouseOutcomeKind::Defeat => -1,
        });
        Ok(json!({
            "episode_id": self.episode_id, "frame": sim.session().tick(),
            "sim_time_ms": sim.session().total_sim_ms(), "reward": reward,
            "terminated": terminated, "truncated": truncated,
            "commands": actions.iter().filter(|action| !matches!(action, AgentCommand::Camera { .. })).map(|_| json!({"submitted_frame": submitted_frame, "received_frame": submitted_frame + 1})).collect::<Vec<_>>()
        }))
    }

    /// Publish the camera resolved by the presentation owner, after native bounds clamping.
    pub fn publish_camera(&mut self, camera: [f32; 2]) {
        self.camera = Some(camera);
    }

    fn observable(&self, entity: &crate::sim::game_entity::GameEntity) -> bool {
        let sim = self.scenario.sim();
        entity.is_alive()
            && !entity.lifecycle.in_limbo()
            && (entity.owner() == self.owner
                || (sim.fog().is_cell_visible(
                    self.owner,
                    entity.position.rx(),
                    entity.position.ry(),
                ) && !entity.cloak.as_ref().is_some_and(|cloak| {
                    cloak.is_fully_cloaked()
                        && !sim.fog().has_sensor_for_house(
                            self.owner,
                            entity.position.rx(),
                            entity.position.ry(),
                        )
                }) && !entity.is_disguised_to(
                    self.owner,
                    Some(sim.fog()),
                    Some(sim.house_alliances()),
                    sim.interner(),
                )))
    }

    /// Read player knowledge without advancing logic, animation or RNG.
    pub fn observe(&self) -> Value {
        let sim = self.scenario.sim();
        let owner_name = sim.interner().resolve(self.owner);
        let rules = &self.scenario.runtime.resources.rules;
        let entities: Vec<_> = sim.entities().values().filter(|entity| self.observable(entity))        .map(|entity| {
            let relation = if entity.owner() == self.owner { "owned" }
                else if crate::map::houses::is_allied_with(sim.house_alliances(), owner_name, sim.interner().resolve(entity.owner())) { "ally" }
                else if sim.houses().get(&entity.owner()).is_some_and(|house| house.multiplay_passive()) { "neutral" }
                else { "enemy" };
            json!({
            "relation": relation,
            "id": entity.stable_id(), "type_name": sim.interner().resolve(entity.type_ref()),
            "owned": entity.owner() == self.owner,
            "rx": entity.position.rx(), "ry": entity.position.ry(),
            "health": entity.health.current(),
        })}).collect();
        let build_options: Vec<_> = production::build_options_for_owner(sim, rules, owner_name).iter().map(|o| json!({
            "type_name": sim.interner().resolve(o.type_id), "cost": o.cost, "enabled": o.enabled,
            "category": o.queue_category.label(),
        })).collect();
        let queue: Vec<_> = production::queue_view_for_owner(sim, rules, owner_name).iter().map(|o| json!({
            "type_name": sim.interner().resolve(o.type_id), "category": o.queue_category.label(), "state": format!("{:?}", o.state),
        })).collect();
        let resources: Vec<_> = sim
            .overlay_grid()
            .into_iter()
            .flat_map(|grid| grid.iter_occupied())
            .filter_map(|(rx, ry, cell)| {
                if !sim.fog().is_cell_visible(self.owner, rx, ry) {
                    return None;
                }
                let overlay = cell.overlay_id?;
                self.scenario
                    .runtime
                    .resources
                    .overlay_registry
                    .tiberium_type_for_overlay(&rules.tiberium_types, overlay)?;
                Some(json!({"rx": rx, "ry": ry, "density": cell.overlay_data}))
            })
            .collect();
        json!({"episode_id": self.episode_id, "frame": sim.session().tick(),
            "sim_time_ms": sim.session().total_sim_ms(), "entities": entities,
            "credits": production::credits_for_owner(sim, owner_name),
            "power": production::power_balance_for_owner(sim, rules, owner_name),
            "build_options": build_options, "production": queue, "resources": resources, "camera": self.camera})
    }
}

/// Independent slots advanced on one bounded Rayon pool, including nested sim work.
pub struct BatchEnv {
    assets: PathBuf,
    pub config: EnvConfig,
    pub slots: Vec<Option<EnvSlot>>,
    pool: Option<rayon::ThreadPool>,
    closed: bool,
}

impl BatchEnv {
    pub fn new(
        assets: PathBuf,
        config: EnvConfig,
        slots: usize,
        threads: usize,
    ) -> Result<Self, String> {
        config.session()?;
        if slots == 0 || threads == 0 {
            return Err("slots and threads must be positive".into());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            assets,
            config,
            slots: (0..slots).map(|_| None).collect(),
            pool: Some(pool),
            closed: false,
        })
    }

    /// Keep client loading and diagnostic work inside the same bounded CPU pool.
    pub fn in_pool<T: Send>(&self, work: impl FnOnce() -> T + Send) -> T {
        self.pool
            .as_ref()
            .expect("environment is open")
            .install(work)
    }

    pub fn validate_ids(&self, ids: &[usize], require_reset: bool) -> Result<(), String> {
        if self.closed {
            return Err("environment is closed".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for &id in ids {
            if id >= self.slots.len() || !seen.insert(id) {
                return Err(format!("invalid or duplicate slot: {id}"));
            }
            if require_reset && self.slots[id].is_none() {
                return Err(format!("reset slot {id} before use"));
            }
        }
        Ok(())
    }

    /// Prepare all replacement worlds before the host commits their presentation resources.
    pub fn prepare_reset(
        &self,
        requests: &[(usize, u32, u64)],
    ) -> Result<Vec<(usize, EnvSlot)>, String> {
        self.validate_ids(&requests.iter().map(|r| r.0).collect::<Vec<_>>(), false)?;
        self.pool.as_ref().unwrap().install(|| {
            requests
                .par_iter()
                .map(|&(id, seed, episode)| {
                    EnvSlot::load(&self.assets, &self.config, seed, episode).map(|slot| (id, slot))
                })
                .collect()
        })
    }

    /// Advance slots with either the GPU-free runtime or the client presentation owner.
    pub fn step_slots_using(
        &mut self,
        requests: &[(usize, Vec<AgentCommand>)],
        advance: impl Fn(usize, &mut EnvSlot, &[AgentCommand]) -> Result<Value, String> + Sync,
    ) -> Result<Vec<Value>, String> {
        self.validate_ids(&requests.iter().map(|r| r.0).collect::<Vec<_>>(), true)?;
        for (id, actions) in requests {
            self.slots[*id].as_ref().unwrap().prepare(actions)?;
        }
        let actions: std::collections::BTreeMap<_, _> = requests
            .iter()
            .map(|(id, actions)| (*id, actions))
            .collect();
        let results: Result<Vec<_>, _> = self.pool.as_ref().unwrap().install(|| {
            self.slots
                .par_iter_mut()
                .enumerate()
                .filter_map(|(id, slot)| {
                    actions.get(&id).map(|actions| {
                        advance(id, slot.as_mut().unwrap(), actions)
                            .map(|result| json!({"slot": id, "result": result}))
                    })
                })
                .collect()
        });
        results
    }

    pub fn observe_slots(&self, ids: &[usize]) -> Result<Vec<Value>, String> {
        self.validate_ids(ids, true)?;
        Ok(ids.iter().map(|&id| json!({"slot": id, "observation": self.slots[id].as_ref().unwrap().observe()})).collect())
    }

    pub fn close(&mut self) {
        self.slots.clear();
        self.pool.take();
        self.closed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Long-lived agent boundary: actors stay player-owned and object targets stay observable.
    #[test]
    fn player_binding_rejects_foreign_actors_and_hidden_targets() {
        let mut own =
            crate::sim::game_entity::GameEntity::test_default(1, "AMCV", "Player", 10, 10);
        let mut enemy =
            crate::sim::game_entity::GameEntity::test_default(2, "SMCV", "Enemy", 12, 12);
        own.lifecycle.set_object_alive_for_test(true);
        own.lifecycle.set_in_limbo_for_test(false);
        enemy.lifecycle.set_object_alive_for_test(true);
        enemy.lifecycle.set_in_limbo_for_test(false);
        let owner = own.owner();
        let mut sim = crate::sim::world::Simulation::new();
        sim.interner = crate::sim::intern::test_interner();
        sim.entities_mut_for_test().insert(own);
        sim.entities_mut_for_test().insert(enemy);
        sim.fog = crate::sim::vision::FogState::new(40, 40);
        let map = crate::map::map_file::MapFile::from_bytes(b"[Map]\nTheater=TEMPERATE\nSize=0,0,40,40\nLocalSize=2,2,36,32\n[IsoMapPack5]\n1=CAAEABUAAAAAEQAA\n").unwrap();
        let mut slot = EnvSlot {
            scenario: HeadlessScenario {
                runtime: crate::sim::runtime::SimRuntime::from_simulation(sim),
                map,
                opening_view_cell: None,
            },
            owner,
            camera: None,
            episode_id: 0,
            max_steps: 100,
            ended: false,
        };
        assert_eq!(
            slot.prepare(&[AgentCommand::Move {
                entity_id: 1,
                rx: 11,
                ry: 11
            }])
            .unwrap()
            .len(),
            1
        );
        for id in [2, 9999] {
            assert!(
                slot.prepare(&[AgentCommand::Move {
                    entity_id: id,
                    rx: 11,
                    ry: 11
                }])
                .unwrap()
                .is_empty()
            );
            assert!(
                slot.prepare(&[AgentCommand::Attack {
                    entity_id: 1,
                    target_id: id
                }])
                .unwrap()
                .is_empty()
            );
        }
        slot.scenario
            .runtime
            .simulation
            .fog
            .mark_visible_for_owner(owner, 12, 12);
        assert_eq!(
            slot.prepare(&[AgentCommand::Attack {
                entity_id: 1,
                target_id: 2
            }])
            .unwrap()
            .len(),
            1
        );
    }
}
