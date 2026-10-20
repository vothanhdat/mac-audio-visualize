use std::f32::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultirateConfig {
    pub band_count: usize,
    pub fft_sizes: [usize; 5],
    pub decimations: [usize; 5],
    pub window_kind: u32,
    pub kaiser_beta: f32,
    pub flanagan_alpha: f32,
    pub analysis_rate_hz: f32,
}

pub struct MultirateFft {
    config: Option<MultirateConfig>,
    bands: [Band; 5],
    work: Vec<[f32; 2]>,
    packed_magnitudes: Vec<f32>,
    samples_since_analysis: usize,
}

impl Default for MultirateFft {
    fn default() -> Self {
        Self {
            config: None,
            bands: std::array::from_fn(|_| Band::empty()),
            work: Vec::new(),
            packed_magnitudes: Vec::new(),
            samples_since_analysis: 0,
        }
    }
}

impl MultirateFft {
    pub fn reset(&mut self, config: MultirateConfig, history: &[f32]) {
        self.config = Some(config);
        self.bands = std::array::from_fn(|index| {
            if index < config.band_count {
                Band::new(
                    config.fft_sizes[index],
                    config.decimations[index],
                    config.window_kind,
                    config.kaiser_beta,
                    config.flanagan_alpha,
                )
            } else {
                Band::empty()
            }
        });
        let maximum_size = config.fft_sizes[..config.band_count]
            .iter()
            .copied()
            .max()
            .unwrap_or(2);
        self.work.resize(maximum_size, [0.0; 2]);
        let packed_len = config.fft_sizes[..config.band_count]
            .iter()
            .map(|size| size / 2 + 1)
            .sum();
        self.packed_magnitudes.resize(packed_len, 0.0);
        for &sample in history {
            for band in &mut self.bands[..config.band_count] {
                band.push(sample);
            }
        }
        self.samples_since_analysis = usize::MAX / 2;
    }

    pub fn ingest(&mut self, samples: &[f32]) {
        let band_count = self.config.map_or(0, |config| config.band_count);
        for &sample in samples {
            for band in &mut self.bands[..band_count] {
                band.push(sample);
            }
        }
        self.samples_since_analysis = self.samples_since_analysis.saturating_add(samples.len());
    }

    pub fn requires_reset(&self, config: MultirateConfig) -> bool {
        let Some(current) = self.config else {
            return true;
        };
        current.band_count != config.band_count
            || current.fft_sizes != config.fft_sizes
            || current.decimations != config.decimations
            || current.window_kind != config.window_kind
            || current.kaiser_beta != config.kaiser_beta
            || current.flanagan_alpha != config.flanagan_alpha
    }

    pub fn set_analysis_rate(&mut self, analysis_rate_hz: f32) {
        if let Some(config) = &mut self.config {
            config.analysis_rate_hz = analysis_rate_hz;
        }
    }

    pub fn analyze_if_due(
        &mut self,
        sample_rate: f32,
        force: bool,
        sync_to_display: bool,
    ) -> Option<&[f32]> {
        let config = self.config?;
        if sync_to_display {
            if !force && self.samples_since_analysis == 0 {
                return None;
            }
            self.samples_since_analysis = 0;
        } else {
            let hop = (sample_rate / config.analysis_rate_hz.max(1.0))
                .round()
                .max(1.0) as usize;
            if !force && self.samples_since_analysis < hop {
                return None;
            }
            self.samples_since_analysis %= hop;
        }

        let mut offset = 0;
        for band in &self.bands[..config.band_count] {
            let count = band.fft_size;
            band.copy_windowed(&mut self.work[..count]);
            fft_in_place(&mut self.work[..count]);
            let magnitude_count = count / 2 + 1;
            let normalization = 2.0 / band.window_sum.max(0.000_001);
            for index in 0..magnitude_count {
                let value = self.work[index];
                self.packed_magnitudes[offset + index] =
                    (value[0] * value[0] + value[1] * value[1]).sqrt() * normalization;
            }
            offset += magnitude_count;
        }
        Some(&self.packed_magnitudes)
    }
}

struct Band {
    fft_size: usize,
    decimation: usize,
    taps: Vec<f32>,
    delay: Vec<f32>,
    delay_position: usize,
    decimation_phase: usize,
    ring: Vec<f32>,
    ring_position: usize,
    ring_filled: usize,
    window: Vec<f32>,
    window_sum: f32,
}

impl Band {
    fn empty() -> Self {
        Self {
            fft_size: 2,
            decimation: 1,
            taps: Vec::new(),
            delay: Vec::new(),
            delay_position: 0,
            decimation_phase: 0,
            ring: vec![0.0; 2],
            ring_position: 0,
            ring_filled: 0,
            window: vec![1.0; 2],
            window_sum: 2.0,
        }
    }

    fn new(
        fft_size: usize,
        decimation: usize,
        window_kind: u32,
        kaiser_beta: f32,
        flanagan_alpha: f32,
    ) -> Self {
        let taps = design_decimation_filter(decimation);
        let delay = vec![0.0; taps.len()];
        let window = make_window(fft_size, window_kind, kaiser_beta, flanagan_alpha);
        let window_sum = window.iter().sum();
        Self {
            fft_size,
            decimation,
            taps,
            delay,
            delay_position: 0,
            decimation_phase: 0,
            ring: vec![0.0; fft_size],
            ring_position: 0,
            ring_filled: 0,
            window,
            window_sum,
        }
    }

    fn push(&mut self, sample: f32) {
        if self.decimation == 1 {
            self.push_decimated(sample);
            return;
        }

        self.delay[self.delay_position] = sample;
        self.delay_position = (self.delay_position + 1) % self.delay.len();
        self.decimation_phase += 1;
        if self.decimation_phase < self.decimation {
            return;
        }
        self.decimation_phase = 0;

        let mut filtered = 0.0;
        let length = self.delay.len();
        for (tap, &weight) in self.taps.iter().enumerate() {
            let index = (self.delay_position + length - 1 - tap) % length;
            filtered += self.delay[index] * weight;
        }
        self.push_decimated(filtered);
    }

    fn push_decimated(&mut self, sample: f32) {
        self.ring[self.ring_position] = sample;
        self.ring_position = (self.ring_position + 1) % self.fft_size;
        self.ring_filled = (self.ring_filled + 1).min(self.fft_size);
    }

    fn copy_windowed(&self, output: &mut [[f32; 2]]) {
        let missing = self.fft_size - self.ring_filled;
        for (index, value) in output.iter_mut().enumerate() {
            let sample = if index < missing {
                0.0
            } else {
                let valid_index = index - missing;
                let oldest =
                    (self.ring_position + self.fft_size - self.ring_filled) % self.fft_size;
                self.ring[(oldest + valid_index) % self.fft_size]
            };
            *value = [sample * self.window[index], 0.0];
        }
    }
}

fn design_decimation_filter(factor: usize) -> Vec<f32> {
    if factor == 1 {
        return Vec::new();
    }
    let length = (factor * 8 + 1).min(1025);
    let center = (length - 1) as f32 * 0.5;
    let cutoff = 0.45 / factor as f32;
    let mut taps = Vec::with_capacity(length);
    for index in 0..length {
        let distance = index as f32 - center;
        let sinc = if distance.abs() < 0.000_001 {
            1.0
        } else {
            let x = 2.0 * cutoff * distance;
            (PI * x).sin() / (PI * x)
        };
        let hamming = 0.54 - 0.46 * (2.0 * PI * index as f32 / (length - 1) as f32).cos();
        taps.push(2.0 * cutoff * sinc * hamming);
    }
    let sum: f32 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= sum;
    }
    taps
}

fn make_window(size: usize, kind: u32, kaiser_beta: f32, flanagan_alpha: f32) -> Vec<f32> {
    let denominator = (size - 1).max(1) as f32;
    (0..size)
        .map(|index| {
            let position = index as f32 / denominator;
            match kind {
                0 => 0.5 - 0.5 * (2.0 * PI * position).cos(),
                1 => 0.5 - 0.5 * (PI * position).cos(),
                2 => 0.42 - 0.5 * (PI * position).cos() + 0.08 * (2.0 * PI * position).cos(),
                3 => {
                    let x = position - 1.0;
                    bessel_i0(kaiser_beta * (1.0 - x * x).max(0.0).sqrt()) / bessel_i0(kaiser_beta)
                }
                _ => {
                    let age = 1.0 - position;
                    let peak_age = 2.0 / flanagan_alpha;
                    let peak = peak_age * peak_age * (-flanagan_alpha * peak_age).exp();
                    age * age * (-flanagan_alpha * age).exp() / peak.max(0.000_001)
                }
            }
        })
        .collect()
}

fn bessel_i0(value: f32) -> f32 {
    let y = value * value * 0.25;
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..=12 {
        term *= y / (k * k) as f32;
        sum += term;
    }
    sum
}

fn fft_in_place(values: &mut [[f32; 2]]) {
    let size = values.len();
    let mut target = 0;
    for source in 1..size {
        let mut bit = size >> 1;
        while target & bit != 0 {
            target ^= bit;
            bit >>= 1;
        }
        target ^= bit;
        if source < target {
            values.swap(source, target);
        }
    }

    let mut length = 2;
    while length <= size {
        let angle = -2.0 * PI / length as f32;
        let rotation = [angle.cos(), angle.sin()];
        for start in (0..size).step_by(length) {
            let mut twiddle = [1.0, 0.0];
            for offset in 0..length / 2 {
                let even = values[start + offset];
                let odd = values[start + offset + length / 2];
                let rotated = [
                    odd[0] * twiddle[0] - odd[1] * twiddle[1],
                    odd[0] * twiddle[1] + odd[1] * twiddle[0],
                ];
                values[start + offset] = [even[0] + rotated[0], even[1] + rotated[1]];
                values[start + offset + length / 2] = [even[0] - rotated[0], even[1] - rotated[1]];
                twiddle = [
                    twiddle[0] * rotation[0] - twiddle[1] * rotation[1],
                    twiddle[0] * rotation[1] + twiddle[1] * rotation[0],
                ];
            }
        }
        length *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::{MultirateConfig, MultirateFft, fft_in_place};

    #[test]
    fn fft_places_exact_tone_in_expected_bin() {
        let size = 128;
        let bin = 7;
        let mut values = (0..size)
            .map(|index| {
                let phase = 2.0 * std::f32::consts::PI * bin as f32 * index as f32 / size as f32;
                [phase.cos(), 0.0]
            })
            .collect::<Vec<_>>();
        fft_in_place(&mut values);
        let strongest = values[..=size / 2]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                let left = left[0] * left[0] + left[1] * left[1];
                let right = right[0] * right[0] + right[1] * right[1];
                left.total_cmp(&right)
            })
            .map(|(index, _)| index)
            .unwrap();
        assert_eq!(strongest, bin);
    }

    #[test]
    fn display_sync_analyzes_once_for_each_new_audio_batch() {
        let config = MultirateConfig {
            band_count: 1,
            fft_sizes: [128, 0, 0, 0, 0],
            decimations: [1, 0, 0, 0, 0],
            window_kind: 0,
            kaiser_beta: 8.0,
            flanagan_alpha: 8.0,
            analysis_rate_hz: 1.0,
        };
        let mut fft = MultirateFft::default();
        fft.reset(config, &[0.0; 128]);

        assert!(fft.analyze_if_due(44_100.0, true, true).is_some());
        assert!(fft.analyze_if_due(44_100.0, false, true).is_none());
        fft.ingest(&[0.25]);
        assert!(fft.analyze_if_due(44_100.0, false, true).is_some());
        assert!(fft.analyze_if_due(44_100.0, false, true).is_none());
    }
}
