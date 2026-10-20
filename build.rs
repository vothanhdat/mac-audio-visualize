fn main() {
    println!("cargo:rerun-if-changed=native/system_audio.m");
    println!("cargo:rerun-if-changed=macos/Info.plist");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let mut native = cc::Build::new();
    native
        // cc otherwise defaults to the installed SDK's version (e.g. macOS 27),
        // which would defeat the runtime 14.2 availability guard on older Macs.
        .env(
            "MACOSX_DEPLOYMENT_TARGET",
            std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "11.0".into()),
        )
        .file("native/system_audio.m")
        .flag("-fobjc-arc")
        .flag("-Wall")
        .flag("-Wextra")
        .compile("visual_audio_capture");

    // Objective-C @available checks use Clang's OS-version runtime. Rust links
    // with -nodefaultlibs, so Clang does not add this library automatically.
    let runtime = native
        .get_compiler()
        .to_command()
        .arg("--print-file-name=libclang_rt.osx.a")
        .output()
        .expect("could not locate the Clang macOS runtime");
    assert!(
        runtime.status.success(),
        "could not locate the Clang macOS runtime"
    );
    let runtime_path = String::from_utf8(runtime.stdout).unwrap();
    let runtime_path = std::path::Path::new(runtime_path.trim());
    assert!(runtime_path.is_file(), "Clang macOS runtime was not found");
    println!(
        "cargo:rustc-link-search=native={}",
        runtime_path.parent().unwrap().display()
    );
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
    println!("cargo:rustc-link-lib=framework=CoreAudio");
    println!("cargo:rustc-link-lib=framework=Foundation");
    // Give `cargo run` the same audio-capture usage description as the app bundle.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-arg=-Wl,-sectcreate,__TEXT,__info_plist,{manifest}/macos/Info.plist");
}
