#!/usr/bin/env bash
# Build lanty-pet in release mode and atomically install to ~/.lantern/bin/lanty-pet
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PET_DIR="$PROJECT_ROOT/pet"
DEST="$HOME/.lantern/bin/lanty-pet"

echo "Building lanty-pet (release)..."
cargo build --release --manifest-path "$PET_DIR/Cargo.toml"

echo "Installing -> $DEST"
mkdir -p "$(dirname "$DEST")"
cp "$PET_DIR/target/release/lanty-pet" "/tmp/lanty-pet-new.$$"
mv -f "/tmp/lanty-pet-new.$$" "$DEST"
chmod +x "$DEST"

echo "Done. Run: $DEST"
