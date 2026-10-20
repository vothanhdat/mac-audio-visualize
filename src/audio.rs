use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
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
pub struct SampleRing {
    slots: Box<[AtomicU64]>,
    capacity: usize,
    read: AtomicUsize,
    write: AtomicUsize,
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
        }
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

pub struct AudioInput {
    // Kept before `ring` so callbacks stop before the consumer's ownership ends.
    _capture: Capture,
    pub ring: Arc<SampleRing>,
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: u16,
}

enum Capture {
    Device { _stream: Stream },
    #[cfg(target_os = "macos")]
    System { _stream: system_audio::SystemAudio },
}

impl AudioInput {
    pub fn system() -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            let ring = Arc::new(SampleRing::new(131_072));
            let (stream, sample_rate, channels) = system_audio::SystemAudio::open(Arc::clone(&ring))?;
            Ok(Self {
                _capture: Capture::System { _stream: stream },
                ring,
                device_name: "System Audio".to_owned(),
                sample_rate,
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
            sample_rate: config.sample_rate,
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
