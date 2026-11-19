//! The renderer's passes executed under validation, read back by the host.

use std::time::{Duration, Instant};

use dirk_assets::{AssetHandle, AssetRegistry, AssetType};
use dirk_events::EventManager;
use dirk_render_utils::{Handle, RenderGraph, Transients, Uploader};
use dirk_rhi::{
    Buffer, BufferAccess, BufferDesc, BufferImageCopy, BufferUsage, Completion, Extent3d, Format,
    Frame, Graphics, Image, ImageAccess, ImageDesc, ImageDimension, ImageUsage, Memory, Rhi,
    SampleCount,
};
use dirk_rhi_test_shaders::support::with_rhi;
use dirk_threads::WorkerPool;
use dirk_universe::{Entity, Universe, World};
use glam::{Mat4, Vec3, vec3};

use crate::{
    models::ModelRegistry,
    proxy::scene::SceneManager,
    viewport::{SHOWN, Viewport},
    window::Presenter,
};

/// Errors of the renderer, the RHI, and test setup.
type Result<T = ()> = anyhow::Result<T>;

/// An image presented to, like a window's, and read back by the host.
struct Target {
    image: Image,
    readback: Buffer,
}

impl Target {
    /// A `width` by `height` target of `format`.
    fn new(rhi: &Rhi, format: Format, (width, height): (u32, u32)) -> Result<Self> {
        Ok(Self {
            image: rhi.create_image(&ImageDesc {
                label: "target",
                dimension: ImageDimension::D2,
                extent: Extent3d::new_2d(width, height),
                format,
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::COPY_SRC,
                mip_levels: 1,
                array_layers: 1,
                samples: SampleCount::X1,
            })?,
            readback: rhi.create_buffer(&BufferDesc {
                label: "readback",
                size: u64::from(width * height * format.bytes_per_texel()),
                usage: BufferUsage::COPY_DST,
                memory: Memory::Readback,
            })?,
        })
    }

    /// Imports the target into `graph`, discarding its contents like a
    /// swapchain image's, and adds `record`'s passes, then a pass copying
    /// it into the readback buffer. Executes and submits the graph.
    fn render<'a>(
        &'a self,
        frame: &Frame<'_>,
        transients: &mut Transients,
        record: impl FnOnce(&mut RenderGraph<'a>, Handle<Image>) -> Result,
    ) -> Result<Completion> {
        let mut graph = RenderGraph::new();
        let target =
            graph.import_image(&self.image, ImageAccess::Undefined, ImageAccess::CopySource)?;
        record(&mut graph, target)?;
        let buffer =
            graph.import_buffer(&self.readback, BufferAccess::None, BufferAccess::HostRead)?;
        let info = *self.image.info();
        graph
            .pass("read back")
            .image(target, ImageAccess::CopySource)
            .buffer(buffer, BufferAccess::CopyDestination)
            .record(move |pass| {
                let (image, buffer) = (pass.image(target)?, pass.buffer(buffer)?);
                let copy = BufferImageCopy::packed(info.format, 0, info.mip_region(0));
                pass.copy_image_to_buffer(image, buffer, &[copy])
            })?;
        let mut encoder = frame.encoder::<Graphics>("test")?;
        graph.execute(frame, &mut encoder, transients)?;
        Ok(frame.submit(encoder)?)
    }

    /// The texels once `done` is reached, row by row.
    fn read(&self, rhi: &Rhi, done: Completion) -> Result<Vec<[u8; 4]>> {
        rhi.wait(done, Duration::from_secs(5))?;
        let mut bytes = vec![0; usize::try_from(self.readback.info().size)?];
        #[expect(unsafe_code, reason = "reading GPU results needs the work to be done")]
        // SAFETY: the copy completed, and the graph made it visible to the
        // host.
        unsafe {
            self.readback.read(0, &mut bytes)?;
        }
        Ok(bytes.as_chunks::<4>().0.to_vec())
    }
}

/// Whether each channel of `actual` is within one of `expected`.
fn close(actual: [u8; 4], expected: [u8; 4]) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(&actual, expected)| actual.abs_diff(expected) <= 1)
}

/// Draws the fixture models at `objects`' paths, translated by their
/// offsets, into a `width` by `height` image seen from three units in front
/// of the origin, and returns its texels row by row.
fn draw_models(
    rhi: &mut Rhi,
    extent: (u32, u32),
    objects: &[(&str, Vec3)],
) -> Result<Vec<[u8; 4]>> {
    let workers = WorkerPool::new("renderer tests");
    let events = EventManager::new(workers.clone());
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let assets = AssetRegistry::init_at(&events, workers, fixtures)?;

    let mut uploader = Uploader::new(rhi, 1 << 16)?;
    let mut models = {
        let mut frame = rhi.begin_frame()?;
        let mut uploads = uploader.begin(&mut frame);
        let models = ModelRegistry::new(&mut uploads, &events)?;
        uploads.submit()?;
        models
    };
    let handles: Vec<_> = objects
        .iter()
        .map(|&(path, _)| AssetHandle::from_raw(path, AssetType::Model))
        .collect();
    let mut assets_loaded = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    for handle in &handles {
        let mut load = assets.load_asset::<dirk_assets::Model>(handle);
        loop {
            if let Some(asset) = load.try_poll().transpose()? {
                assets_loaded.push(asset);
                break;
            }
            anyhow::ensure!(Instant::now() < deadline, "{handle:?} did not load in time");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    // Load events reach the registry asynchronously.
    while handles
        .iter()
        .any(|handle| models.primitives(handle).is_none())
    {
        anyhow::ensure!(
            Instant::now() < deadline,
            "the models did not reach the renderer in time"
        );
        std::thread::sleep(Duration::from_millis(1));
        let mut frame = rhi.begin_frame()?;
        let mut uploads = uploader.begin(&mut frame);
        models.update(uploads.rhi(), &mut uploads)?;
        uploads.submit()?;
    }

    let universe = Universe::builder().build()?;
    let mut commands = universe.handle().command_buffer();
    let world = commands.create_world(World::builder("test"));
    let camera = commands.spawn(world, Entity::builder());
    let mut scene = SceneManager::new(rhi)?;
    scene.create_scene(world);
    for (handle, (_, offset)) in handles.into_iter().zip(objects) {
        let entity = commands.spawn(world, Entity::builder());
        scene.create_proxy(entity, world)?;
        let proxy = scene.get_proxy_mut(entity).expect("just created");
        proxy.set_model(Some(handle));
        proxy.set_model_matrix(Some(Mat4::from_translation(*offset)));
    }
    scene.create_proxy(camera, world)?;
    let view = glam::camera::rh::view::look_at_mat4(vec3(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
    scene
        .get_proxy_mut(camera)
        .expect("just created")
        .set_view(Some(view));
    let mut viewport = Viewport::new(rhi, scene.targets())?;
    viewport.resize(rhi, extent)?;
    (viewport.world, viewport.camera) = (Some(world), Some(camera));

    let presenter = Presenter::new(rhi)?;
    let pipeline = presenter.pipeline(rhi, Format::Rgba8Unorm)?;
    let target = Target::new(rhi, Format::Rgba8Unorm, extent)?;
    let mut transients = Transients::default();
    let frame = rhi.begin_frame()?;
    scene.write_objects(&frame)?;
    let mut graph = RenderGraph::new();
    viewport.record(&mut graph, &frame, &scene, &models)?;
    let mut encoder = frame.encoder::<Graphics>("viewport")?;
    graph.execute(&frame, &mut encoder, &mut transients)?;
    frame.submit(encoder)?;
    let done = target.render(&frame, &mut transients, |graph, image| {
        let output = graph.import_image(viewport.output(), SHOWN, SHOWN)?;
        Ok(presenter.record(graph, &pipeline, image, Some(output))?)
    })?;
    frame.finish();
    target.read(rhi, done)
}

/// Asserts that each `(x, y, color)` texel of the `width` wide `texels` is
/// close to `color`.
fn assert_texels(texels: &[[u8; 4]], width: u32, expected: &[(u32, u32, [u8; 4])]) {
    for &(x, y, expected) in expected {
        let texel = texels[(y * width + x) as usize];
        assert!(
            close(texel, expected),
            "texel ({x}, {y}) is {texel:?}, not {expected:?}"
        );
    }
}

/// A glTF quad, textured with sRGB (200, 100, 50), drawn as two objects of
/// one model: each reads its own transform by instance index.
#[test]
fn gltf_models_draw_each_object_with_its_own_data() -> Result {
    const EXTENT: (u32, u32) = (16, 8);
    with_rhi(|rhi| {
        let quad = "quad/quad.dirkasset";
        let objects = [(quad, vec3(-1.2, 0.0, 0.0)), (quad, vec3(1.2, 0.0, 0.0))];
        let texels = draw_models(rhi, EXTENT, &objects)?;
        let (textured, black) = ([200, 100, 50, 255], [0, 0, 0, 255]);
        assert_texels(
            &texels,
            EXTENT.0,
            &[
                (4, 4, textured),
                (11, 4, textured),
                (8, 4, black),
                (0, 0, black),
            ],
        );
        Ok(())
    })
}

/// A quad textured with a 32x32 black and white checkerboard covers a few
/// texels, so it samples small mips, which average the squares in linear
/// space: sRGB 188, not the 128 of averaging encoded values.
#[test]
fn model_textures_sample_mips_averaged_in_linear_space() -> Result {
    const EXTENT: (u32, u32) = (16, 8);
    with_rhi(|rhi| {
        let objects = [("checker/checker.dirkasset", Vec3::ZERO)];
        let texels = draw_models(rhi, EXTENT, &objects)?;
        assert_texels(&texels, EXTENT.0, &[(8, 4, [188, 188, 188, 255])]);
        Ok(())
    })
}

/// A unit quad from the origin, without normals or material, drawn by a
/// triangle list covering its lower right half and a strip and a fan
/// covering all of it; only their second triangles cover its top. It is
/// white and lit head-on wherever converted triangles keep their winding,
/// so culling keeps them, and flat normals face the viewer.
#[test]
fn converted_topologies_draw_with_flat_normals() -> Result {
    const EXTENT: (u32, u32) = (32, 16);
    with_rhi(|rhi| {
        let objects = [("gltf/multi_primitive.dirkasset", Vec3::ZERO)];
        let texels = draw_models(rhi, EXTENT, &objects)?;
        let (white, black) = ([255; 4], [0, 0, 0, 255]);
        // The top of the quad, its bottom, then outside it.
        assert_texels(
            &texels,
            EXTENT.0,
            &[(19, 2, white), (19, 6, white), (13, 4, black)],
        );
        Ok(())
    })
}

/// egui's sRGB-encoded, premultiplied colors are composited in linear space
/// and encoded once, whether the target encodes them or not.
#[cfg(feature = "editor")]
#[test]
fn ui_colors_are_composited_linearly_and_encoded_once() -> Result {
    use egui::{
        Color32, Rgba, TextureId, TextureOptions, TexturesDelta,
        epaint::{ClippedPrimitive, ColorImage, ImageDelta, Mesh, Primitive, pos2},
    };

    use crate::egui_integration::EguiRenderer;

    const EXTENT: (u32, u32) = (12, 4);
    with_rhi(|rhi| {
        let gray = Color32::from_gray(128);
        let half_white = Color32::from_white_alpha(128);
        let additive_red = Color32::from_rgba_premultiplied(128, 0, 0, 0);
        // Gray, half-transparent white over black, and red added to gray.
        let mut mesh = Mesh::default();
        let rect = |x: f32| egui::Rect::from_min_max(pos2(x, 0.0), pos2(x + 4.0, 4.0));
        mesh.add_colored_rect(rect(0.0), gray);
        mesh.add_colored_rect(rect(4.0), half_white);
        mesh.add_colored_rect(rect(8.0), gray);
        mesh.add_colored_rect(rect(8.0), additive_red);
        let primitives = [ClippedPrimitive {
            clip_rect: egui::Rect::EVERYTHING,
            primitive: Primitive::Mesh(mesh),
        }];
        let mut delta = TexturesDelta::default();
        let white = ColorImage::filled([1, 1], Color32::WHITE);
        delta.set.insert(
            TextureId::default(),
            [ImageDelta::full(white, TextureOptions::LINEAR)].into(),
        );

        let linear = |color: Color32| Rgba::from(color).to_array();
        let encode = egui::ecolor::gamma_u8_from_linear_f32;
        let gray_linear = linear(gray)[0];
        let expected = [
            [128, 128, 128, 255],
            [
                encode(linear(half_white)[0]),
                encode(linear(half_white)[1]),
                encode(linear(half_white)[2]),
                255,
            ],
            [encode(gray_linear + linear(additive_red)[0]), 128, 128, 255],
        ];

        let mut uploader = Uploader::new(rhi, 1 << 16)?;
        let mut egui = EguiRenderer::new(rhi)?;
        let presenter = Presenter::new(rhi)?;
        let mut transients = Transients::default();
        for format in [Format::Rgba8Srgb, Format::Rgba8Unorm] {
            let pipeline = presenter.pipeline(rhi, format)?;
            let target = Target::new(rhi, format, EXTENT)?;
            let mut frame = rhi.begin_frame()?;
            let mut uploads = uploader.begin(&mut frame);
            egui.set_textures(uploads.rhi(), &mut uploads, &mut delta)?;
            uploads.submit()?;
            egui.prepare(&frame, &primitives, 1.0, EXTENT, EXTENT)?;
            let done = target.render(&frame, &mut transients, |graph, image| {
                let ui = egui.record(graph, &frame, EXTENT, [])?;
                Ok(presenter.record(graph, &pipeline, image, Some(ui))?)
            })?;
            frame.finish();

            let texels = target.read(rhi, done)?;
            for (column, expected) in expected.iter().enumerate() {
                let texel = texels[column * 4 + EXTENT.0 as usize + 1];
                assert!(
                    close(texel, *expected),
                    "{format:?}: column {column} is {texel:?}, not {expected:?}"
                );
            }
        }
        Ok(())
    })
}
