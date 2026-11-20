//! Finds the push data type each entry point reads in the shader crate's
//! source, as host code names it, for the generated `Shader`
//! implementations.
//!
//! Types are named by the paths their entry point's module reaches them by:
//! its `use` declarations, `crate`, `self` and `super`, or items it
//! declares. Push data types the shader crate defines are named through it,
//! so host crates including the generated code depend on it.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, bail};
use quote::ToTokens;
use syn::{Attribute, FnArg, Item, Meta, Token, Type, UseTree, punctuated::Punctuated};

/// Primitive types, which need no path.
const PRIMITIVES: [&str; 13] = [
    "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128", "f32", "f64", "bool",
];

/// The source of a shader crate.
pub(super) struct Source {
    /// The crate's name, which host crates depending on it reach it by.
    name: String,
    /// The crate root.
    root: PathBuf,
}

/// A module of a shader crate.
struct Module {
    /// Its path from the crate root.
    path: Vec<String>,
    items: Vec<Item>,
    /// The directory its out-of-line child modules are in.
    dir: PathBuf,
}

impl Source {
    /// Finds the library of the shader crate at `shader_crate`.
    pub(super) fn new(shader_crate: &Path) -> anyhow::Result<Self> {
        let manifest = fs::canonicalize(shader_crate.join("Cargo.toml"))
            .with_context(|| format!("finding the manifest of {}", shader_crate.display()))?;
        let metadata = cargo_metadata::MetadataCommand::new()
            .manifest_path(&manifest)
            .no_deps()
            .exec()
            .with_context(|| format!("reading the metadata of {}", manifest.display()))?;
        let library = metadata
            .packages
            .iter()
            .filter(|package| package.manifest_path == manifest)
            .flat_map(|package| &package.targets)
            .find(|target| target.is_lib() || target.is_dylib())
            .with_context(|| format!("{} has no library", manifest.display()))?;
        Ok(Self {
            name: library.name.replace('-', "_"),
            root: library.src_path.clone().into(),
        })
    }

    /// The push data type `entry_point`, a function path from the crate
    /// root, reads, as a path from another crate; `()` if it reads none.
    pub(super) fn push_type(&self, entry_point: &str) -> anyhow::Result<String> {
        let mut path: Vec<_> = entry_point.split("::").collect();
        let function = path.pop().context("empty entry point")?;
        let mut module = Module::load(&self.root, Vec::new())?;
        for name in path {
            module = module.child(name)?;
        }
        module.push_type(&self.name, function)
    }
}

impl Module {
    /// Parses the module in `file`, at `path` from the crate root.
    fn load(file: &Path, path: Vec<String>) -> anyhow::Result<Self> {
        let code =
            fs::read_to_string(file).with_context(|| format!("reading {}", file.display()))?;
        let items = syn::parse_file(&code)
            .with_context(|| format!("parsing {}", file.display()))?
            .items;
        // Root and `mod.rs` modules own their directory; others own a
        // directory named after them.
        let parent = file.parent().unwrap_or(Path::new("."));
        let dir = match file.file_stem().and_then(|stem| stem.to_str()) {
            Some("mod") | None => parent.to_owned(),
            Some(_) if path.is_empty() => parent.to_owned(),
            Some(stem) => parent.join(stem),
        };
        Ok(Self { path, items, dir })
    }

    /// The child module `name`, inline or in its own file.
    fn child(self, name: &str) -> anyhow::Result<Self> {
        let declared = self.items.into_iter().find_map(|item| match item {
            Item::Mod(module) if module.ident == name => Some(module),
            _ => None,
        });
        let module = declared.with_context(|| format!("no module `{name}`"))?;
        let mut path = self.path;
        path.push(name.to_owned());
        if let Some((_, items)) = module.content {
            return Ok(Self {
                path,
                items,
                dir: self.dir.join(name),
            });
        }
        let file = self.dir.join(format!("{name}.rs"));
        let file = if file.exists() {
            file
        } else {
            self.dir.join(name).join("mod.rs")
        };
        Self::load(&file, path)
    }

    /// The push data type `function`, an entry point of this module of
    /// crate `krate`, reads: the type its `#[spirv(push_constant)]`
    /// parameter refers to, or `()` without one.
    fn push_type(&self, krate: &str, function: &str) -> anyhow::Result<String> {
        let function = self
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(item) if item.sig.ident == function => Some(item),
                _ => None,
            })
            .with_context(|| format!("no function `{function}` in `{}`", self.path.join("::")))?;
        let push = function.sig.inputs.iter().find_map(|input| match input {
            FnArg::Typed(input) if input.attrs.iter().any(Self::is_push_constant) => {
                Some(&*input.ty)
            }
            _ => None,
        });
        match push {
            None => Ok("()".to_owned()),
            Some(Type::Reference(reference)) => self.resolve(krate, &reference.elem),
            Some(other) => bail!(
                "the push constant parameter is a {}, not a reference",
                other.to_token_stream()
            ),
        }
    }

    /// Whether `attribute` is `#[spirv(push_constant)]`, possibly among
    /// other arguments.
    fn is_push_constant(attribute: &Attribute) -> bool {
        attribute.path().is_ident("spirv")
            && attribute
                .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                .is_ok_and(|metas| {
                    metas
                        .iter()
                        .any(|meta| meta.path().is_ident("push_constant"))
                })
    }

    /// `ty`, written in this module of crate `krate`, as another crate
    /// names it.
    fn resolve(&self, krate: &str, ty: &Type) -> anyhow::Result<String> {
        Ok(match ty {
            Type::Tuple(tuple) if tuple.elems.is_empty() => "()".to_owned(),
            Type::Array(array) => format!(
                "[{}; {}]",
                self.resolve(krate, &array.elem)?,
                array.len.to_token_stream()
            ),
            Type::Path(path) if path.qself.is_none() => {
                let segments: Vec<_> = path
                    .path
                    .segments
                    .iter()
                    .map(|segment| segment.to_token_stream().to_string())
                    .collect();
                match segments.as_slice() {
                    [primitive] if PRIMITIVES.contains(&primitive.as_str()) => primitive.clone(),
                    _ if path.path.leading_colon.is_some() => format!("::{}", segments.join("::")),
                    _ => format!("::{}", self.absolute(krate, &segments)?.join("::")),
                }
            }
            _ => bail!("cannot name the push data type {}", ty.to_token_stream()),
        })
    }

    /// `segments`, a path written in this module of crate `krate`, from
    /// the root of the crate it starts in.
    fn absolute(&self, krate: &str, segments: &[String]) -> anyhow::Result<Vec<String>> {
        let (first, rest) = segments.split_first().context("empty path")?;
        let within = |module: &[String]| {
            std::iter::once(krate.to_owned())
                .chain(module.iter().cloned())
                .chain(rest.iter().cloned())
                .collect()
        };
        Ok(match first.as_str() {
            "crate" => within(&[]),
            "self" => within(&self.path),
            "super" => within(self.path.split_last().map_or(&[][..], |(_, parent)| parent)),
            _ => match self.uses().get(first) {
                // `use name;` of an extern crate names itself.
                Some(used) if used != std::slice::from_ref(first) => {
                    self.absolute(krate, &[used.as_slice(), rest].concat())?
                }
                _ if self.declares(first) => std::iter::once(krate.to_owned())
                    .chain(self.path.iter().cloned())
                    .chain(segments.iter().cloned())
                    .collect(),
                // An extern crate.
                _ => segments.to_vec(),
            },
        })
    }

    /// The paths this module's `use` declarations bring into scope, by the
    /// name they bring them in as.
    fn uses(&self) -> HashMap<String, Vec<String>> {
        /// Adds the names `tree`, below `prefix`, brings into scope.
        fn add(tree: &UseTree, prefix: &[String], uses: &mut HashMap<String, Vec<String>>) {
            let path = |name: String| [prefix, &[name]].concat();
            match tree {
                UseTree::Path(tree) => add(&tree.tree, &path(tree.ident.to_string()), uses),
                UseTree::Name(name) if name.ident == "self" => {
                    if let Some(last) = prefix.last() {
                        uses.insert(last.clone(), prefix.to_vec());
                    }
                }
                UseTree::Name(name) => {
                    uses.insert(name.ident.to_string(), path(name.ident.to_string()));
                }
                UseTree::Rename(rename) => {
                    uses.insert(rename.rename.to_string(), path(rename.ident.to_string()));
                }
                UseTree::Group(group) => {
                    for tree in &group.items {
                        add(tree, prefix, uses);
                    }
                }
                UseTree::Glob(_) => {}
            }
        }
        let mut uses = HashMap::new();
        for item in &self.items {
            if let Item::Use(item) = item {
                add(&item.tree, &[], &mut uses);
            }
        }
        uses
    }

    /// Whether this module declares an item or module named `name`.
    fn declares(&self, name: &str) -> bool {
        self.items.iter().any(|item| match item {
            Item::Struct(item) => item.ident == name,
            Item::Enum(item) => item.ident == name,
            Item::Union(item) => item.ident == name,
            Item::Type(item) => item.ident == name,
            Item::Mod(item) => item.ident == name,
            _ => false,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::Module;

    /// The push data type `function` reads in a module of crate `shaders`
    /// at `path`, holding `code`.
    fn push_type(path: &[&str], code: &str, function: &str) -> String {
        let module = Module {
            path: path.iter().map(ToString::to_string).collect(),
            items: syn::parse_file(code).expect("valid code").items,
            dir: PathBuf::new(),
        };
        module.push_type("shaders", function).expect("resolvable")
    }

    #[test]
    fn push_types_are_named_as_their_module_reaches_them() {
        let code = r"
            use dirk_shaders::types::{MeshPush, PresentPush as Present};
            use crate::common;

            pub struct Local { value: u32 }

            #[spirv(vertex)]
            pub fn imported(#[spirv(push_constant)] push: &MeshPush) {}
            #[spirv(fragment)]
            pub fn renamed(uv: Vec2, #[spirv(push_constant)] push: &Present) {}
            #[spirv(fragment)]
            pub fn local(#[spirv(push_constant)] push: &Local) {}
            #[spirv(fragment)]
            pub fn through_module(#[spirv(push_constant)] push: &common::Shared) {}
            #[spirv(fragment)]
            pub fn parent(#[spirv(push_constant)] push: &super::Outer) {}
            #[spirv(compute(threads(64)))]
            pub fn array(#[spirv(push_constant)] push: &[u32; 4]) {}
            #[spirv(vertex)]
            pub fn none(#[spirv(vertex_index)] index: i32) {}
        ";
        let resolve = |function| push_type(&["meshes"], code, function);
        assert_eq!(resolve("imported"), "::dirk_shaders::types::MeshPush");
        assert_eq!(resolve("renamed"), "::dirk_shaders::types::PresentPush");
        assert_eq!(resolve("local"), "::shaders::meshes::Local");
        assert_eq!(resolve("through_module"), "::shaders::common::Shared");
        assert_eq!(resolve("parent"), "::shaders::Outer");
        assert_eq!(resolve("array"), "[u32; 4]");
        assert_eq!(resolve("none"), "()");
    }
}
