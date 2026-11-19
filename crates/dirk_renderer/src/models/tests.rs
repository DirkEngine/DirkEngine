use dirk_rhi::{AddressMode, Filter};
use glam::Vec3;

use super::{Geometry, SamplerKey, Unsupported};

/// Reads every primitive of the first mesh of fixture `name`, mapping
/// texture coordinates of `uv_set`.
fn read<const N: usize>(
    name: &str,
    uv_set: u32,
) -> anyhow::Result<[Result<Geometry, Unsupported>; N]> {
    let path = format!("{}/tests/fixtures/gltf/{name}", env!("CARGO_MANIFEST_DIR"));
    let (gltf, buffers, _) = gltf::import(path)?;
    let mesh = gltf.meshes().next().expect("fixtures have a mesh");
    let primitives: Vec<_> = mesh
        .primitives()
        .map(|primitive| Geometry::read(&primitive, &buffers, uv_set))
        .collect();
    primitives
        .try_into()
        .map_err(|primitives: Vec<_>| anyhow::anyhow!("{} primitives", primitives.len()))
}

/// Each index's vertex position.
fn positions(geometry: &Geometry) -> Vec<[f32; 3]> {
    geometry
        .indices
        .iter()
        .map(|&index| geometry.vertices[index as usize].position)
        .collect()
}

/// Whether every vertex has the normal +Z.
fn faces_viewer(geometry: &Geometry) -> bool {
    geometry
        .vertices
        .iter()
        .all(|vertex| Vec3::from(vertex.normal) == Vec3::Z)
}

#[test]
fn primitives_without_indices_or_normals_get_both() -> anyhow::Result<()> {
    let [Ok(triangle)] = read("non_indexed.gltf", 0)? else {
        panic!("the primitive should load");
    };
    assert_eq!(triangle.indices, [0, 1, 2]);
    assert_eq!(
        positions(&triangle),
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    );
    assert!(faces_viewer(&triangle));
    Ok(())
}

#[test]
fn byte_indices_and_sparse_positions_are_read() -> anyhow::Result<()> {
    let ([Ok(quad)], [Ok(sparse)]) = (read("u8_indices.gltf", 0)?, read("sparse.gltf", 0)?) else {
        panic!("the primitives should load");
    };
    let corners = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    assert_eq!(positions(&quad), [0, 1, 2, 0, 2, 3].map(|i| corners[i]));
    // The sparse accessor replaces the third zero position.
    assert_eq!(positions(&sparse), [0, 1, 3].map(|i| corners[i]));
    Ok(())
}

#[test]
fn strips_and_fans_become_lists_and_other_topologies_are_skipped() -> anyhow::Result<()> {
    let [Ok(list), Ok(strip), Ok(fan), Err(lines), Err(partial)] = read("multi_primitive.gltf", 0)?
    else {
        panic!("only the triangle primitives should load");
    };
    assert_eq!(list.index_count, 3);
    for quad in [&strip, &fan] {
        assert_eq!(quad.index_count, 6);
        // Front faces stay front faces, facing the viewer.
        assert!(faces_viewer(quad));
    }
    assert!(matches!(
        lines,
        Unsupported::Topology(gltf::mesh::Mode::Lines)
    ));
    assert!(matches!(partial, Unsupported::PartialTriangle(4)));
    Ok(())
}

#[test]
fn vertex_colors_are_multiplied_by_the_base_color_factor() -> anyhow::Result<()> {
    let [Ok(triangle)] = read("vertex_colors.gltf", 0)? else {
        panic!("the primitive should load");
    };
    let colors: Vec<_> = triangle
        .vertices
        .iter()
        .map(|vertex| vertex.color)
        .collect();
    assert_eq!(
        colors,
        [
            [0.5, 0.0, 0.0, 0.25],
            [0.0, 1.0, 0.0, 0.25],
            [0.0, 0.0, 1.0, 0.25]
        ]
    );
    // Without vertex colors or a material, colors are white.
    let [Ok(white)] = read("non_indexed.gltf", 0)? else {
        panic!("the primitive should load");
    };
    let colors: Vec<_> = white.vertices.iter().map(|vertex| vertex.color).collect();
    assert_eq!(colors, [[1.0; 4]; 3]);
    Ok(())
}

#[test]
fn sampler_filters_wrapping_and_other_uv_sets_are_read() -> anyhow::Result<()> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/gltf/sampler_uv_set.gltf"
    );
    let (gltf, _, _) = gltf::import(path)?;
    let texture = gltf.textures().next().expect("the fixture has a texture");
    assert_eq!(
        SamplerKey::from(&texture.sampler()),
        SamplerKey {
            mag_filter: Filter::Nearest,
            min_filter: Filter::Nearest,
            mipmap_filter: Some(Filter::Nearest),
            address_modes: [AddressMode::ClampToEdge, AddressMode::MirrorRepeat],
        }
    );
    let [Ok(triangle)] = read("sampler_uv_set.gltf", 1)? else {
        panic!("the primitive should load");
    };
    let uvs: Vec<_> = triangle.vertices.iter().map(|vertex| vertex.uv).collect();
    assert_eq!(uvs, [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]);
    Ok(())
}
