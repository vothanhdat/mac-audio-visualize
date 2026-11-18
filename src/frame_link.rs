use std::{
    ffi::c_void,
    ptr::NonNull,
    time::{Duration, Instant},
};

use winit::{
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::Window,
};

unsafe extern "C" {
    fn va_frame_link_create(ns_view: *mut c_void) -> *mut c_void;
    fn va_frame_link_set_rate(link: *mut c_void, frames_per_second: u32);
    fn va_frame_link_take_tick() -> bool;
}

/// Paces frames below the display refresh, in step with it.
#[derive(Default)]
pub struct FrameLink {
    link: Option<NonNull<c_void>>,
    rate: u32,
    last_frame: Option<Instant>,
}

impl FrameLink {
    pub fn attach(&mut self, window: &Window) {
        let Ok(handle) = window.window_handle() else {
            return;
        };
        if let RawWindowHandle::AppKit(handle) = handle.as_raw() {
            // SAFETY: the view belongs to a live window and this runs on the main thread.
            self.link = NonNull::new(unsafe { va_frame_link_create(handle.ns_view.as_ptr()) });
        }
    }

    pub fn is_attached(&self) -> bool {
        self.link.is_some()
    }

    /// Runs the link at this rate, or pauses it for 0 so that it stops waking the app.
    pub fn set_rate(&mut self, frames_per_second: u32) {
        let Some(link) = self.link else {
            return;
        };
        if self.rate != frames_per_second {
            self.rate = frames_per_second;
            // SAFETY: the link stays valid for the life of the process.
            unsafe { va_frame_link_set_rate(link.as_ptr(), frames_per_second) };
        }
    }

    /// Whether the link has asked for a frame that the limit allows.
    pub fn frame_due(&mut self, now: Instant) -> bool {
        // SAFETY: this only reads and clears an atomic flag.
        if self.rate == 0 || !unsafe { va_frame_link_take_tick() } {
            return false;
        }
        let due = self
            .last_frame
            .is_none_or(|last| limit_allows_frame(now - last, self.rate));
        if due {
            self.last_frame = Some(now);
        }
        due
    }
}

/// A display that cannot run at the limit may tick faster than it. The margin
/// keeps normal timing jitter from dropping a frame that is on time.
fn limit_allows_frame(since_last_frame: Duration, frames_per_second: u32) -> bool {
    since_last_frame.as_secs_f64() >= 0.9 / f64::from(frames_per_second)
}

#[cfg(test)]
mod tests {
    use super::limit_allows_frame;
    use std::time::Duration;

    fn frames_shown(tick_hz: f64, limit: u32, ticks: u32) -> u32 {
        let tick = Duration::from_secs_f64(1.0 / tick_hz);
        let mut since_last_frame = Duration::from_secs(1);
        let mut shown = 0;
        for _ in 0..ticks {
            if limit_allows_frame(since_last_frame, limit) {
                shown += 1;
                since_last_frame = Duration::ZERO;
            }
            since_last_frame += tick;
        }
        shown
    }

    #[test]
    fn ticks_at_the_limit_all_draw() {
        assert_eq!(frames_shown(40.0, 40, 400), 400);
        assert_eq!(frames_shown(60.0, 60, 600), 600);
    }

    #[test]
    fn a_60_hz_display_never_exceeds_a_40_fps_limit() {
        assert_eq!(frames_shown(60.0, 40, 600), 300);
    }

    #[test]
    fn a_slightly_early_tick_still_draws() {
        assert!(limit_allows_frame(Duration::from_secs_f64(0.024), 40));
    }
}
