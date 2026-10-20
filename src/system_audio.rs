use std::{
    ffi::{CStr, c_char, c_void},
    ptr::NonNull,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};

use super::SampleRing;

type SamplesCallback = unsafe extern "C" fn(*mut c_void, *const f32, *const f32, u32, u32);

unsafe extern "C" {
    fn va_capture_open(
        callback: SamplesCallback,
        context: *mut c_void,
        sample_rate: *mut u32,
        channels: *mut u16,
        error: *mut c_char,
        error_size: usize,
    ) -> *mut c_void;
    fn va_capture_close(capture: *mut c_void);
    fn va_output_latency_ms() -> f64;
}

pub(super) struct SystemAudio {
    capture: NonNull<c_void>,
    // The stable Arc allocation stays alive until the native IOProc has stopped.
    _ring: Arc<SampleRing>,
    latency_checked_at: Option<Instant>,
    output_latency_ms: Option<f32>,
}

impl SystemAudio {
    pub(super) fn open(ring: Arc<SampleRing>) -> Result<(Self, u32, u16)> {
        let mut sample_rate = 0;
        let mut channels = 0;
        let mut error = [0 as c_char; 512];
        // SAFETY: all output pointers are valid for the call. `ring` remains alive
        // during startup and, on success, for the entire native capture lifetime.
        let capture = unsafe {
            va_capture_open(
                receive_samples,
                Arc::as_ptr(&ring).cast_mut().cast(),
                &mut sample_rate,
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
            sample_rate,
            channels,
        ))
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
        // SAFETY: this owns the handle; close stops/removes callbacks synchronously
        // before `_ring` is released, including on event-loop startup errors.
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
}
