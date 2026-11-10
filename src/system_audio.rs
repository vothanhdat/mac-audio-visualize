use std::{
    ffi::{CStr, c_char, c_void},
    ptr::NonNull,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};

use super::SampleRing;

type SamplesCallback = unsafe extern "C" fn(*mut c_void, *const f32, *const f32, u32, u32);
type SampleRateCallback = unsafe extern "C" fn(*mut c_void, u32);

unsafe extern "C" {
    fn va_capture_open(
        samples: SamplesCallback,
        sample_rate: SampleRateCallback,
        context: *mut c_void,
        channels: *mut u16,
        error: *mut c_char,
        error_size: usize,
    ) -> *mut c_void;
    #[cfg(test)]
    fn va_capture_restart(capture: *mut c_void);
    fn va_capture_close(capture: *mut c_void);
    fn va_output_latency_ms() -> f64;
}

/// Native system capture. It rebuilds its process tap when the default output
/// device or the tap's format changes, publishing each new sample rate to the
/// ring before samples at that rate arrive.
pub(super) struct SystemAudio {
    capture: NonNull<c_void>,
    // The stable Arc allocation stays alive until native capture has closed.
    _ring: Arc<SampleRing>,
    latency_checked_at: Option<Instant>,
    output_latency_ms: Option<f32>,
}

impl SystemAudio {
    /// On success the ring already holds the capture's sample rate.
    pub(super) fn open(ring: Arc<SampleRing>) -> Result<(Self, u16)> {
        let mut channels = 0;
        let mut error = [0 as c_char; 512];
        // SAFETY: all output pointers are valid for the call. `ring` remains alive
        // during startup and, on success, for the entire native capture lifetime.
        let capture = unsafe {
            va_capture_open(
                receive_samples,
                receive_sample_rate,
                Arc::as_ptr(&ring).cast_mut().cast(),
                &mut channels,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let Some(capture) = NonNull::new(capture) else {
            // SAFETY: the bridge always null-terminates this fixed-size error buffer.
            let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            bail!("{message}");
        };
        Ok((
            Self {
                capture,
                _ring: ring,
                latency_checked_at: None,
                output_latency_ms: None,
            },
            channels,
        ))
    }

    /// Rebuild the tap as an output change would. Returns immediately.
    #[cfg(test)]
    fn restart(&self) {
        // SAFETY: the handle is open; the restart runs on the capture's own queue.
        unsafe { va_capture_restart(self.capture.as_ptr()) };
    }

    pub(super) fn output_latency_ms(&mut self) -> Option<f32> {
        let now = Instant::now();
        if self
            .latency_checked_at
            .is_none_or(|last| now.duration_since(last) >= Duration::from_secs(1))
        {
            // SAFETY: this queries Core Audio properties and retains no Rust pointers.
            let latency = unsafe { va_output_latency_ms() };
            self.output_latency_ms = if latency.is_finite() && latency >= 0.0 {
                Some(latency.min(1000.0) as f32)
            } else {
                None
            };
            self.latency_checked_at = Some(now);
        }
        self.output_latency_ms
    }
}

impl Drop for SystemAudio {
    fn drop(&mut self) {
        // SAFETY: this owns the handle; close waits for any rebuild in progress and
        // stops/removes callbacks synchronously before `_ring` is released,
        // including on event-loop startup errors.
        unsafe { va_capture_close(self.capture.as_ptr()) };
    }
}

unsafe extern "C" fn receive_samples(
    context: *mut c_void,
    left: *const f32,
    right: *const f32,
    frames: u32,
    stride: u32,
) {
    // SAFETY: the native bridge supplies valid Float32 buffers of at least
    // `frames` frames with this stride. The Arc outlives every callback.
    let ring = unsafe { &*context.cast::<SampleRing>() };
    for frame in 0..frames as usize {
        let offset = frame * stride as usize;
        unsafe { ring.push(*left.add(offset), *right.add(offset)) };
    }
}

/// Called on the native capture queue while no samples are delivered: before
/// capture starts and between stopping and restarting it.
unsafe extern "C" fn receive_sample_rate(context: *mut c_void, sample_rate: u32) {
    // SAFETY: the Arc outlives every callback (see `receive_samples`).
    let ring = unsafe { &*context.cast::<SampleRing>() };
    ring.begin_sample_rate(sample_rate);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(
        left: *const f32,
        right: *const f32,
        frames: u32,
        stride: u32,
    ) -> ([f32; 3], [f32; 3]) {
        let ring = SampleRing::new(8);
        unsafe {
            receive_samples(
                (&ring as *const SampleRing).cast_mut().cast(),
                left,
                right,
                frames,
                stride,
            );
        }
        let mut left = [0.0; 3];
        let mut right = [0.0; 3];
        assert_eq!(ring.drain_latest(&mut left, &mut right), frames as usize);
        (left, right)
    }

    #[test]
    fn native_interleaved_stereo_preserves_channels() {
        let data = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let (left, right) = capture(data.as_ptr(), unsafe { data.as_ptr().add(1) }, 3, 2);
        assert_eq!(left, [0.1, 0.2, 0.3]);
        assert_eq!(right, [-0.1, -0.2, -0.3]);
    }

    #[test]
    fn native_planar_stereo_preserves_channels() {
        let left = [0.1, 0.2, 0.3];
        let right = [-0.1, -0.2, -0.3];
        assert_eq!(capture(left.as_ptr(), right.as_ptr(), 3, 1), (left, right));
    }

    #[test]
    fn native_rate_callback_separates_samples_by_rate() {
        let ring = SampleRing::new(8);
        let context = (&ring as *const SampleRing).cast_mut().cast();
        let data = [0.1, -0.1];
        unsafe {
            receive_sample_rate(context, 44_100);
            receive_samples(context, data.as_ptr(), data.as_ptr().add(1), 1, 2);
            receive_sample_rate(context, 48_000);
        }
        assert_eq!(ring.take_sample_rate_change(), Some(48_000));
        let (mut left, mut right) = ([0.0; 2], [0.0; 2]);
        assert_eq!(
            ring.drain_latest(&mut left, &mut right),
            0,
            "44.1 kHz samples must not be analyzed at 48 kHz"
        );
    }

    fn drain_for(ring: &SampleRing, duration: Duration) -> usize {
        let (mut left, mut right) = (vec![0.0; 1 << 17], vec![0.0; 1 << 17]);
        let deadline = Instant::now() + duration;
        let mut total = 0;
        while Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
            total += ring.drain_latest(&mut left, &mut right);
        }
        total
    }

    /// Exercises the native rebuild path that an output device or format
    /// change takes, without changing any system setting or playing sound.
    #[test]
    #[ignore = "opens a system audio tap; set VISUAL_AUDIO_CAPTURE_TEST=1 (delivery is checked only while an app plays audio)"]
    fn system_capture_keeps_delivering_across_rebuilds() {
        if std::env::var_os("VISUAL_AUDIO_CAPTURE_TEST").is_none() {
            eprintln!("skipped: set VISUAL_AUDIO_CAPTURE_TEST=1 to open a system audio tap");
            return;
        }
        let ring = Arc::new(SampleRing::new(1 << 17));
        // Closing must wait for a pending or running rebuild (each takes a few ms
        // after the 200 ms settle delay) and never hang or call back afterwards.
        for delay_ms in [0, 150, 195, 200, 202, 205, 210, 250, 400] {
            let (capture, channels) = SystemAudio::open(Arc::clone(&ring)).unwrap();
            assert_eq!(channels, 2);
            assert!(ring.sample_rate() > 0);
            capture.restart();
            std::thread::sleep(Duration::from_millis(delay_ms));
            let closing = Instant::now();
            drop(capture);
            assert!(closing.elapsed() < Duration::from_secs(2), "close hung");
        }
        let leftover = drain_for(&ring, Duration::from_millis(100));
        assert_eq!(
            drain_for(&ring, Duration::from_millis(300)),
            0,
            "samples arrived after close ({leftover} were still queued)"
        );

        let (capture, _) = SystemAudio::open(Arc::clone(&ring)).unwrap();
        let rate = ring.sample_rate();
        let before = drain_for(&ring, Duration::from_secs(1));
        if before == 0 {
            eprintln!("delivery not checked: no app is playing audio, so the tap delivers nothing");
            return;
        }
        eprintln!("capturing at {rate} Hz: {before} frames in 1 s before rebuilds");
        for attempt in 0..3 {
            capture.restart();
            // The rebuild waits 200 ms for a notification burst to settle.
            drain_for(&ring, Duration::from_millis(400));
            let after = drain_for(&ring, Duration::from_secs(1));
            eprintln!("rebuild {attempt}: {after} frames in 1 s at {} Hz", ring.sample_rate());
            assert!(
                after as f64 > 0.8 * f64::from(ring.sample_rate()),
                "capture did not resume after rebuild {attempt}: {after} frames"
            );
        }
    }
}
