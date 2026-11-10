#!/bin/sh
# Build VisualAudio.app and package it as dist/VisualAudio-VERSION.zip for a
# GitHub release and the vothanhdat/tap Homebrew cask. Apple Silicon only.
set -eu

cd "$(dirname "$0")/.."
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)
plist_version=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' macos/Info.plist)
if [ "$version" != "$plist_version" ]; then
    echo "Cargo.toml version $version does not match macos/Info.plist $plist_version" >&2
    exit 1
fi
if [ "$(uname -m)" != "arm64" ]; then
    echo "Build the release on an Apple Silicon Mac; the cask is arm64 only" >&2
    exit 1
fi

# Start from a fresh bundle so no files from an earlier build are packaged.
rm -rf target/VisualAudio.app
sh scripts/build-macos-app.sh
codesign --verify --strict target/VisualAudio.app

mkdir -p dist
zip="dist/VisualAudio-$version.zip"
rm -f "$zip"
ditto -c -k --norsrc --noextattr --noacl --keepParent target/VisualAudio.app "$zip"
printf 'Packaged %s\nsha256 %s\n' "$zip" "$(shasum -a 256 "$zip" | cut -d ' ' -f 1)"
