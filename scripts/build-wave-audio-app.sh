#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
cargo build --release --bin wave-audio
app="target/WaveAudio.app"
mkdir -p "$app/Contents/MacOS"
cp target/release/wave-audio "$app/Contents/MacOS/wave-audio"
cp macos/WaveAudio-Info.plist "$app/Contents/Info.plist"
codesign --force --sign - "$app"
printf 'Built %s/target/WaveAudio.app\n' "$PWD"
