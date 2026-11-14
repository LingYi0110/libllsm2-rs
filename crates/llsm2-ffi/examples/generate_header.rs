fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = cbindgen::Config::from_file(root.join("cbindgen.toml")).unwrap();
    let bindings = cbindgen::Builder::new()
        .with_crate(root)
        .with_config(config)
        .generate()
        .expect("generate C ABI header");
    std::fs::create_dir_all(root.join("include")).unwrap();
    bindings.write_to_file(root.join("include/llsm2.h"));
}
