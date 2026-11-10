use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use cpal::{
    SampleFormat, Stream, StreamConfig,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

#[cfg(target_os = "macos")]
#[path = "system_audio.rs"]
mod system_audio;

/// A bounded single-producer/single-consumer queue with no allocation or locks
/// in the CoreAudio callback. Synchronized stereo frames are packed into one
/// atomic word so the renderer can never pair samples from different times.
///
/// The ring also carries the samples' rate, which system capture changes when
/// it follows a new output device or format.
pub struct SampleRing {
    slots: Box<[AtomicU64]>,
    capacity: usize,
    read: AtomicUsize,
    write: AtomicUsize,
    // `generation << 32 | sample_rate` of the samples written from `rate_start`.
    rate: AtomicU64,
    rate_start: AtomicUsize,
    // Consumer side: the rate generation last returned by `take_sample_rate_change`.
    seen_rate: AtomicU32,
}

impl SampleRing {
    pub fn new(capacity: usize) -> Self {
        let slots = (0..capacity)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            slots,
            capacity,
            read: AtomicUsize::new(0),
            write: AtomicUsize::new(0),
            rate: AtomicU64::new(0),
            rate_start: AtomicUsize::new(0),
            seen_rate: AtomicU32::new(0),
        }
    }

    /// Producer side: samples pushed from now on have this rate. Call only
    /// while nothing is pushing, before capture starts or while it is stopped.
    fn begin_sample_rate(&self, sample_rate: u32) {
        let current = self.rate.load(Ordering::Relaxed);
        if current as u32 == sample_rate {
            return; // A restart at the same rate continues the same stream.
        }
        let generation = ((current >> 32) as u32).wrapping_add(1);
        self.rate_start
            .store(self.write.load(Ordering::Acquire), Ordering::Relaxed);
        // Release: a consumer that sees the new rate also sees where it starts.
        self.rate.store(
            u64::from(generation) << 32 | u64::from(sample_rate),
            Ordering::Release,
        );
    }

    /// Rate of the newest samples; 0 before capture has started.
    pub fn sample_rate(&self) -> u32 {
        self.rate.load(Ordering::Acquire) as u32
    }

    /// Consumer side: the new rate if it changed since the previous call.
    /// Samples still queued from before the change are discarded.
    pub fn take_sample_rate_change(&self) -> Option<u32> {
        let rate = self.rate.load(Ordering::Acquire);
        let generation = (rate >> 32) as u32;
        if generation == self.seen_rate.load(Ordering::Relaxed) {
            return None;
        }
        self.seen_rate.store(generation, Ordering::Relaxed);
        let start = self.rate_start.load(Ordering::Relaxed);
        let read = self.read.load(Ordering::Relaxed);
        let write = self.write.load(Ordering::Acquire);
        // Only skip forward over written samples. A later change may already
        // have moved `start`; the next call then reports that change.
        if start.wrapping_sub(read) <= write.wrapping_sub(read) {
            self.read.store(start, Ordering::Release);
        }
        Some(rate as u32)
    }

    #[inline]
    fn push(&self, left: f32, right: f32) {
        let write = self.write.load(Ordering::Relaxed);
        let read = self.read.load(Ordering::Acquire);
        if write.wrapping_sub(read) >= self.capacity {
            return; // Rendering fell behind: keep the callback real-time safe.
        }
        let packed = u64::from(left.to_bits()) | (u64::from(right.to_bits()) << 32);
        self.slots[write % self.capacity].store(packed, Ordering::Relaxed);
        self.write.store(write.wrapping_add(1), Ordering::Release);
    }

    pub fn drain_latest(&self, left_history: &mut [f32], right_history: &mut [f32]) -> usize {
        debug_assert_eq!(left_history.len(), right_history.len());
        let write = self.write.load(Ordering::Acquire);
        let mut read = self.read.load(Ordering::Relaxed);
        let available = write.wrapping_sub(read).min(self.capacity);
        let skip = available.saturating_sub(left_history.len());
        read = read.wrapping_add(skip);

        let count = available - skip;
        if count > 0 {
            left_history.copy_within(count.., 0);
            right_history.copy_within(count.., 0);
            let start = left_history.len() - count;
            for index in start..left_history.len() {
                let packed = self.slots[read % self.capacity].load(Ordering::Relaxed);
                left_history[index] = f32::from_bits(packed as u32);
                right_history[index] = f32::from_bits((packed >> 32) as u32);
                read = read.wrapping_add(1);
            }
        }
        // Consume everything, including samples skipped to reduce latency.
        self.read.store(write, Ordering::Release);
        count
    }
}

/// How long capture may deliver nothing before the missing time counts as
/// silence. Frames also fall between audio callbacks, so shorter gaps hold.
pub const CAPTURE_STALL: Duration = Duration::from_millis(250);

/// Longest part of a stall filled with silence: enough for every sample-driven
/// level to decay and for the history to flush at any rate. Longer stalls
/// then hold the decayed levels without further analysis work.
pub const MAX_STALL_SILENCE: Duration = Duration::from_secs(10);

/// System capture delivers no samples at all while no app plays, and none
/// while it restarts after an output change. Analyzers hold their last levels
/// on frames without new samples, so once a stall outlasts `CAPTURE_STALL`
/// its remaining time is filled with silence and levels decay, instead of
/// freezing when capture stops mid-sound.
#[derive(Debug)]
pub struct CaptureStall {
    last_samples: Instant,
    filled: u64,
}

impl CaptureStall {
    pub fn new(now: Instant) -> Self {
        Self {
            last_samples: now,
            filled: 0,
        }
    }

    /// Silent samples to append after a drain that returned `drained` samples.
    pub fn silent_samples(&mut self, drained: usize, now: Instant, sample_rate: f32) -> usize {
        if drained > 0 {
            *self = Self::new(now);
            return 0;
        }
        let Some(stalled) = now
            .saturating_duration_since(self.last_samples)
            .checked_sub(CAPTURE_STALL)
        else {
            return 0;
        };
        let stalled = stalled.min(MAX_STALL_SILENCE);
        let due = (stalled.as_secs_f64() * f64::from(sample_rate.max(0.0))) as u64;
        let count = due.saturating_sub(self.filled);
        self.filled = self.filled.max(due);
        count.try_into().unwrap_or(usize::MAX)
    }

    /// Silent samples appended since capture last delivered audio.
    pub fn filled(&self) -> u64 {
        self.filled
    }
}

pub struct AudioInput {
    // Kept before `ring` so callbacks stop before the consumer's ownership ends.
    _capture: Capture,
    pub ring: Arc<SampleRing>,
    pub device_name: String,
    pub channels: u16,
}

enum Capture {
    Device { _stream: Stream },
    #[cfg(target_os = "macos")]
    System { _stream: system_audio::SystemAudio },
}

impl AudioInput {
    /// Rate of the newest captured samples. System capture follows output
    /// device and format changes; `SampleRing::take_sample_rate_change`
    /// reports them to the consumer.
    pub fn sample_rate(&self) -> u32 {
        self.ring.sample_rate()
    }

    pub fn output_latency_ms(&mut self) -> Option<f32> {
        match &mut self._capture {
            #[cfg(target_os = "macos")]
            Capture::System { _stream } => _stream.output_latency_ms(),
            // Input devices may already be routed/synchronized externally.
            Capture::Device { .. } => None,
        }
    }

    pub fn system() -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let ring = Arc::new(SampleRing::new(131_072));
            let (stream, channels) = system_audio::SystemAudio::open(Arc::clone(&ring))?;
            // The consumer starts at the current rate; only later changes are news.
            ring.take_sample_rate_change();
            if ring.sample_rate() == 0 {
                bail!("system audio capture did not report a sample rate");
            }
            Ok(Self {
                _capture: Capture::System { _stream: stream },
                ring,
                device_name: "System Audio".to_owned(),
                channels,
            })
        }
        #[cfg(not(target_os = "macos"))]
        bail!("system audio capture requires macOS 14.2 or newer; use --device NAME")
    }

    pub fn open(name_fragment: &str) -> Result<Self> {
        let host = cpal::default_host();
        let wanted = name_fragment.to_ascii_lowercase();
        let mut discovered = Vec::new();
        let mut selected = None;

        for device in host
            .input_devices()
            .context("could not enumerate input devices")?
        {
            let name = device.to_string();
            discovered.push(name.clone());
            if selected.is_none() && name.to_ascii_lowercase().contains(&wanted) {
                selected = Some((device, name));
            }
        }

        let Some((device, device_name)) = selected else {
            bail!(
                "no input device matched {name_fragment:?}. Available inputs: {}",
                discovered.join(", ")
            );
        };

        let supported = device
            .default_input_config()
            .with_context(|| format!("{device_name} has no default input format"))?;
        let sample_format = supported.sample_format();
        let config: StreamConfig = supported.into();
        let channels = config.channels as usize;
        let ring = Arc::new(SampleRing::new(131_072));
        ring.begin_sample_rate(config.sample_rate);
        ring.take_sample_rate_change();
        let stream = match sample_format {
            SampleFormat::F32 => {
                let callback_ring = Arc::clone(&ring);
                let error_device = device_name.clone();
                device.build_input_stream(
                    config,
                    move |data: &[f32], _| {
                        push_stereo(data.iter().copied(), channels, &callback_ring)
                    },
                    move |error| eprintln!("audio stream error on {error_device}: {error}"),
                    None,
                )
            }
            SampleFormat::I16 => {
                let callback_ring = Arc::clone(&ring);
                let error_device = device_name.clone();
                device.build_input_stream(
                    config,
                    move |data: &[i16], _| {
                        push_stereo(
                            data.iter().map(|&v| v as f32 / i16::MAX as f32),
                            channels,
                            &callback_ring,
                        )
                    },
                    move |error| eprintln!("audio stream error on {error_device}: {error}"),
                    None,
                )
            }
            SampleFormat::U16 => {
                let callback_ring = Arc::clone(&ring);
                let error_device = device_name.clone();
                device.build_input_stream(
                    config,
                    move |data: &[u16], _| {
                        push_stereo(
                            data.iter().map(|&v| v as f32 / u16::MAX as f32 * 2.0 - 1.0),
                            channels,
                            &callback_ring,
                        )
                    },
                    move |error| eprintln!("audio stream error on {error_device}: {error}"),
                    None,
                )
            }
            other => bail!("unsupported input sample format: {other:?}"),
        }
        .with_context(|| format!("could not open input stream on {device_name}"))?;

        stream.play().context("could not start the input stream")?;
        Ok(Self {
            _capture: Capture::Device { _stream: stream },
            ring,
            device_name,
            channels: config.channels,
        })
    }
}

#[inline]
fn push_stereo(samples: impl Iterator<Item = f32>, channels: usize, ring: &SampleRing) {
    if channels == 0 {
        return;
    }
    let mut left = 0.0;
    let mut right = 0.0;
    let mut channel = 0;
    for sample in samples {
        if channel == 0 {
            left = sample;
            right = sample;
        } else if channel == 1 {
            right = sample;
        }
        channel += 1;
        if channel == channels {
            ring.push(left, right);
            channel = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_frames_remain_synchronized() {
        let ring = SampleRing::new(8);
        ring.push(1.0, -1.0);
        ring.push(2.0, -2.0);
        ring.push(3.0, -3.0);
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];

        assert_eq!(ring.drain_latest(&mut left, &mut right), 3);
        assert_eq!(left, [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(right, [0.0, -1.0, -2.0, -3.0]);
    }

    #[test]
    fn sample_rate_change_discards_samples_from_the_previous_rate() {
        let ring = SampleRing::new(8);
        assert_eq!((ring.sample_rate(), ring.take_sample_rate_change()), (0, None));
        ring.begin_sample_rate(44_100);
        assert_eq!(ring.take_sample_rate_change(), Some(44_100));
        assert_eq!(ring.take_sample_rate_change(), None);
        ring.push(1.0, -1.0);
        ring.push(2.0, -2.0);
        // Capture restarts at a new rate before the old samples were drained.
        ring.begin_sample_rate(48_000);
        ring.push(3.0, -3.0);
        assert_eq!(ring.sample_rate(), 48_000);
        assert_eq!(ring.take_sample_rate_change(), Some(48_000));
        assert_eq!(ring.take_sample_rate_change(), None);
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        assert_eq!(ring.drain_latest(&mut left, &mut right), 1);
        assert_eq!((left, right), ([0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, -3.0]));
    }

    #[test]
    fn restart_at_the_same_rate_keeps_queued_samples() {
        let ring = SampleRing::new(8);
        ring.begin_sample_rate(48_000);
        ring.take_sample_rate_change();
        ring.push(1.0, -1.0);
        ring.begin_sample_rate(48_000);
        ring.push(2.0, -2.0);
        assert_eq!(ring.take_sample_rate_change(), None);
        let mut left = [0.0; 2];
        let mut right = [0.0; 2];
        assert_eq!(ring.drain_latest(&mut left, &mut right), 2);
        assert_eq!(left, [1.0, 2.0]);
    }

    #[test]
    fn late_sample_rate_check_never_skips_past_new_samples() {
        let ring = SampleRing::new(4);
        ring.begin_sample_rate(44_100);
        ring.take_sample_rate_change();
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        // Indices wrap the ring several times before the change.
        for index in 0..10 {
            ring.push(index as f32, 0.0);
            ring.drain_latest(&mut left, &mut right);
        }
        ring.begin_sample_rate(48_000);
        ring.push(10.0, 0.0);
        // The consumer drained a new-rate sample before checking the rate:
        // the start of the new rate is behind `read` and nothing more is skipped.
        assert_eq!(ring.drain_latest(&mut left, &mut right), 1);
        ring.push(11.0, 0.0);
        assert_eq!(ring.take_sample_rate_change(), Some(48_000));
        assert_eq!(ring.drain_latest(&mut left, &mut right), 1);
        assert_eq!(left[3], 11.0);
        // Two changes before a check: only samples at the newest rate remain.
        ring.begin_sample_rate(96_000);
        ring.push(12.0, 0.0);
        ring.begin_sample_rate(44_100);
        ring.push(13.0, 0.0);
        assert_eq!(ring.take_sample_rate_change(), Some(44_100));
        assert_eq!(ring.drain_latest(&mut left, &mut right), 1);
        assert_eq!(left[3], 13.0);
    }

    #[test]
    fn capture_stall_fills_silence_only_after_the_stall_time() {
        let start = Instant::now();
        let at = |ms: u64| start + Duration::from_millis(ms);
        let mut stall = CaptureStall::new(start);
        // Frames between audio callbacks hold the last level.
        assert_eq!(stall.silent_samples(0, at(250), 48_000.0), 0);
        // Beyond the stall time, elapsed time becomes silence.
        assert_eq!(stall.silent_samples(0, at(260), 48_000.0), 480);
        assert_eq!(stall.silent_samples(0, at(270), 48_000.0), 480);
        assert_eq!(stall.filled(), 960);
        // Audio resumes: nothing is added and the stall time starts again.
        assert_eq!(stall.silent_samples(128, at(280), 48_000.0), 0);
        assert_eq!(stall.filled(), 0);
        assert_eq!(stall.silent_samples(0, at(530), 44_100.0), 0);
        // Many small frames add up to elapsed time without rounding drift.
        let total: usize = (1..=1000)
            .map(|ms| stall.silent_samples(0, at(530 + ms), 44_100.0))
            .sum();
        assert_eq!(total, 44_100);
        assert_eq!(stall.silent_samples(0, at(1531), 0.0), 0);
        // A long stall is filled only until every level has decayed.
        assert_eq!(
            stall.silent_samples(0, at(530 + 60_000), 44_100.0),
            9 * 44_100
        );
        assert_eq!(stall.filled(), 10 * 44_100);
        assert_eq!(stall.silent_samples(0, at(530 + 70_000), 44_100.0), 0);
    }

    #[test]
    fn mono_input_is_duplicated_to_both_channels() {
        let ring = SampleRing::new(8);
        push_stereo([0.25, -0.5].into_iter(), 1, &ring);
        let mut left = [0.0; 2];
        let mut right = [0.0; 2];

        assert_eq!(ring.drain_latest(&mut left, &mut right), 2);
        assert_eq!(left, right);
        assert_eq!(left, [0.25, -0.5]);
    }
}
