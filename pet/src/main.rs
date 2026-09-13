//! Lanty desktop pet — a small sentient mushroom who wanders along the
//! bottom of your screen and opens the chat window when you click him.

use std::path::PathBuf;

use anyhow::{anyhow, Result};

mod animator;
mod shm;
mod sprite;
mod wayland;

const ICONS_ENV: &str = "LANTY_ICONS_DIR";
const LAUNCH_ENV: &str = "LANTY_LAUNCH_CMD";
const DEFAULT_ICONS_DIR: &str = "/home/alva/Pictures/Icons/SVG/Lanty and Friends/lanty";
const DEFAULT_LAUNCH_CMD: &str = "/home/alva/.lantern/bin/lanty-gui";

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("lanty_pet=warn")),
        )
        .init();

    let icons_dir: PathBuf = std::env::var(ICONS_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_ICONS_DIR));
    if !icons_dir.is_dir() {
        return Err(anyhow!(
            "icons dir not found: {} (set {} to override)",
            icons_dir.display(),
            ICONS_ENV
        ));
    }
    let launch_cmd: PathBuf = std::env::var(LAUNCH_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_LAUNCH_CMD));

    let app = wayland::App { icons_dir, launch_cmd };
    app.run()
}
