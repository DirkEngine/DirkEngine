//! Registers the custom Rust-GPU target for cfg checking.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(target_arch, values(\"spirv\"))");
}
