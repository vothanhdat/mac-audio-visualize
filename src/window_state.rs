//! Remembers where the visualizer and controls windows were, per profile, in
//! a small file next to the settings. Kept apart from the visual settings so
//! moving a window is not a settings change and **Reset** does not move it.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use winit::{monitor::MonitorHandle, window::Window};

/// A window's outer top-left corner and inner size, in logical points.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Frame {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Frame {
    pub fn of(window: &Window) -> Option<Self> {
        let scale = window.scale_factor();
        let position = window.outer_position().ok()?.to_logical::<f64>(scale);
        let size = window.inner_size().to_logical::<f64>(scale);
        Some(Self {
            x: position.x,
            y: position.y,
            width: size.width,
            height: size.height,
        })
    }

    /// True when the saved frame is usable on the current displays: its
    /// numbers are sane and enough of it is on one screen to grab and move it.
    /// A frame on a disconnected display is ignored rather than opened off screen.
    pub fn restorable(&self, screens: &[Frame]) -> bool {
        let finite = [self.x, self.y, self.width, self.height]
            .iter()
            .all(|value| value.is_finite());
        if !finite || !(1.0..=100_000.0).contains(&self.width) || !(1.0..=100_000.0).contains(&self.height) {
            return false;
        }
        let (need_width, need_height) = (self.width.min(40.0), self.height.min(20.0));
        screens.iter().any(|screen| {
            let overlap_width =
                (self.x + self.width).min(screen.x + screen.width) - self.x.max(screen.x);
            let overlap_height =
                (self.y + self.height).min(screen.y + screen.height) - self.y.max(screen.y);
            overlap_width >= need_width && overlap_height >= need_height
        })
    }
}

/// The current displays as logical rectangles in the shared screen space.
pub fn screens(monitors: impl IntoIterator<Item = MonitorHandle>) -> Vec<Frame> {
    monitors
        .into_iter()
        .map(|monitor| {
            let scale = monitor.scale_factor();
            let position = monitor.position().to_logical::<f64>(scale);
            let size = monitor.size().to_logical::<f64>(scale);
            Frame {
                x: position.x,
                y: position.y,
                width: size.width,
                height: size.height,
            }
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WindowState {
    pub visualizer: Option<Frame>,
    pub controls: Option<Frame>,
}

impl WindowState {
    /// A missing or unreadable file means "use the default placement".
    pub fn load(path: &Path) -> Self {
        let source = match fs::read_to_string(path) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(error) => {
                eprintln!("could not read window positions from {}: {error}", path.display());
                return Self::default();
            }
        };
        serde_json::from_str(&source).unwrap_or_else(|error| {
            eprintln!("could not parse window positions from {}: {error}", path.display());
            Self::default()
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let parent = path.parent().context("window position path has no parent")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
        let temporary = path.with_extension("json.tmp");
        let json =
            serde_json::to_string_pretty(self).context("could not serialize window positions")?;
        fs::write(&temporary, json)
            .with_context(|| format!("could not write {}", temporary.display()))?;
        fs::rename(&temporary, path)
            .with_context(|| format!("could not replace {}", path.display()))?;
        Ok(())
    }
}

/// `window.json` for the default profile, `window-NAME.json` for `--profile NAME`,
/// next to that profile's settings file.
pub fn window_state_path(profile: Option<&str>) -> Option<PathBuf> {
    let file_name = profile
        .map(|profile| format!("window-{profile}.json"))
        .unwrap_or_else(|| "window.json".to_owned());
    Some(
        PathBuf::from(std::env::var_os("HOME")?)
            .join("Library/Application Support/VisualAudio")
            .join(file_name),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAPTOP: Frame = Frame {
        x: 0.0,
        y: 0.0,
        width: 1728.0,
        height: 1117.0,
    };
    // An external display to the right of the laptop.
    const EXTERNAL: Frame = Frame {
        x: 1728.0,
        y: -300.0,
        width: 2560.0,
        height: 1440.0,
    };

    fn frame(x: f64, y: f64, width: f64, height: f64) -> Frame {
        Frame {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn frames_restore_only_where_they_can_still_be_grabbed() {
        let both = [LAPTOP, EXTERNAL];
        assert!(frame(100.0, 200.0, 640.0, 120.0).restorable(&[LAPTOP]));
        // Saved on the external display, which is now disconnected.
        let external = frame(2500.0, 100.0, 640.0, 120.0);
        assert!(external.restorable(&both));
        assert!(!external.restorable(&[LAPTOP]));
        // Mostly off the right edge, but a 50-point strip is still on screen.
        assert!(frame(1678.0, 300.0, 640.0, 120.0).restorable(&[LAPTOP]));
        // Only a 10-point sliver remains: too little to grab.
        assert!(!frame(1718.0, 300.0, 640.0, 120.0).restorable(&[LAPTOP]));
        assert!(!frame(100.0, 1110.0, 640.0, 120.0).restorable(&[LAPTOP]));
        // Small windows need only their own size on screen.
        assert!(frame(1700.0, 10.0, 28.0, 12.0).restorable(&[LAPTOP]));
        assert!(!frame(f64::NAN, 0.0, 640.0, 120.0).restorable(&both));
        assert!(!frame(0.0, 0.0, 0.0, 120.0).restorable(&both));
        assert!(!frame(0.0, 0.0, 640.0, 120.0).restorable(&[]));
    }

    #[test]
    fn window_state_round_trips_and_tolerates_bad_files() {
        let directory = std::env::temp_dir().join(format!(
            "visual-audio-window-state-{}",
            std::process::id()
        ));
        let path = directory.join("window.json");
        assert_eq!(WindowState::load(&path), WindowState::default());
        let state = WindowState {
            visualizer: Some(frame(120.5, 80.0, 900.0, 90.0)),
            controls: None,
        };
        state.save(&path).unwrap();
        assert_eq!(WindowState::load(&path), state);
        fs::write(&path, "{ not json").unwrap();
        assert_eq!(WindowState::load(&path), WindowState::default());
        // Unknown or missing fields keep the rest usable.
        fs::write(&path, r#"{"controls":{"x":1,"y":2,"width":3,"height":4},"extra":1}"#).unwrap();
        assert_eq!(
            WindowState::load(&path),
            WindowState {
                visualizer: None,
                controls: Some(frame(1.0, 2.0, 3.0, 4.0)),
            }
        );
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn each_profile_has_its_own_window_file() {
        let default = window_state_path(None).unwrap();
        assert!(default.ends_with("Library/Application Support/VisualAudio/window.json"));
        let named = window_state_path(Some("hybrid")).unwrap();
        assert!(named.ends_with("Library/Application Support/VisualAudio/window-hybrid.json"));
    }
}
