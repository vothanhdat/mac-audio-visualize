#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
cargo build --release
app="target/VisualAudio.app"
mkdir -p "$app/Contents/MacOS"
cp target/release/visual-audio "$app/Contents/MacOS/visual-audio"
cp macos/Info.plist "$app/Contents/Info.plist"
codesign --force --sign - "$app"
printf 'Built %s/target/VisualAudio.app\n' "$PWD"
