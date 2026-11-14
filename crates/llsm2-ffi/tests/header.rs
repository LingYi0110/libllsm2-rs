#[test]
fn checked_in_c99_header_matches_the_exported_abi() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let config = cbindgen::Config::from_file(root.join("cbindgen.toml")).unwrap();
    let bindings = cbindgen::Builder::new()
        .with_crate(root)
        .with_config(config)
        .generate()
        .unwrap();
    let mut generated = Vec::new();
    bindings.write(&mut generated);
    assert_eq!(
        generated,
        std::fs::read(root.join("include/llsm2.h")).unwrap(),
        "regenerate with cargo run -p llsm2-ffi --example generate_header"
    );
}
