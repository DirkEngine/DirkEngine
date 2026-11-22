//! A simple [`Subsystem`] that creates a demo world.

use std::f32::consts::PI;

use dirk_engine::{EngineHandle, EnginePlugin, Subsystem};
use dirk_universe::{CommandBuffer, Entity, UniverseHandle, World};

/// An [`EnginePlugin`] with a basic demo world.
pub struct DemoPlugin;

impl EnginePlugin for DemoPlugin {
    fn name(&self) -> &'static str {
        "demo"
    }
    fn build(&self, builder: &mut dirk_engine::EngineBuilder) -> anyhow::Result<()> {
        builder.with_app_name(self.name());

        builder.with_plugin(dirk_player::PlayerPlugin)?;
        builder.with_plugin(dirk_platform::PlatformPlugin)?;
        builder.add_subsystem(|ctx| {
            Ok(Demo {
                players: ctx.resource::<dirk_player::PlayerRegistry>()?,
                universe: ctx.resource::<dirk_universe::UniverseHandle>()?,
                started: false,
            })
        });
        Ok(())
    }
}

/// A basic demo [`Subsystem`].
struct Demo {
    players: dirk_player::PlayerRegistry,
    universe: UniverseHandle,
    started: bool,
}

impl Subsystem for Demo {
    fn name(&self) -> &'static str {
        "demo-startup"
    }

    fn start(&mut self, _engine: &EngineHandle) -> anyhow::Result<()> {
        if self.started {
            return Ok(());
        }

        let mut cmd = self.universe.command_buffer();

        let world_id = create_test_world(&mut cmd);
        let player = self.players.new_player();

        cmd.spawn(
            world_id,
            Entity::builder().with_component(player).with_component(
                dirk_world::components::Transform {
                    location: glam::vec3(0.0, 500.0, 500.0),
                    rotation: glam::Quat::from_rotation_x(-PI / 4.0),
                    scale: glam::Vec3::ONE,
                },
            ),
        );

        cmd.submit();

        self.started = true;
        Ok(())
    }
}

fn create_test_world(cmd: &mut CommandBuffer) -> dirk_universe::WorldId {
    use dirk_world::components::{Renderable, Transform};

    let duck_model = dirk_assets::AssetHandle::from_raw(
        "models/Duck/Duck.dirkasset",
        dirk_assets::AssetType::Model,
    );
    let shrek_model = dirk_assets::AssetHandle::from_raw(
        "models/Shrek/Shrek.dirkasset",
        dirk_assets::AssetType::Model,
    );

    let shrek_builder = Entity::builder()
        .with_component(Transform {
            location: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::splat(1.),
        })
        .with_component(Renderable::new(shrek_model));

    let duck_builder = Entity::builder()
        .with_component(Transform {
            location: glam::vec3(100., 0., 0.),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::splat(1.),
        })
        .with_component(Renderable::new(duck_model));

    let world_builder = World::builder("test world")
        .with_entity(shrek_builder)
        .with_entity(duck_builder);

    cmd.create_world(world_builder)
}
