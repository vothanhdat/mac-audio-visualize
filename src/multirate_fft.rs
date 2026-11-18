use std::f32::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MultirateConfig {
    pub band_count: usize,
    pub fft_sizes: [usize; 5],
    pub decimations: [usize; 5],
    pub window_kind: u32,
    pub kaiser_beta: f32,
    pub flanagan_alpha: f32,
}

pub struct MultirateFft {
    config: Option<MultirateConfig>,
    bands: [Band; 5],
    windowed: Vec<f32>,
    fft_real: Vec<f32>,
    fft_imaginary: Vec<f32>,
    packed_magnitudes: Vec<f32>,
    samples_since_analysis: usize,
    frames_until_analysis: u32,
    display_divisor: u32,
}

impl Default for MultirateFft {
    fn default() -> Self {
        Self {
            config: None,
            bands: std::array::from_fn(|_| Band::empty()),
            windowed: Vec::new(),
            fft_real: Vec::new(),
            fft_imaginary: Vec::new(),
            packed_magnitudes: Vec::new(),
            samples_since_analysis: 0,
            frames_until_analysis: 0,
            display_divisor: 1,
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
        self.windowed.resize(maximum_size, 0.0);
        self.fft_real.resize(maximum_size / 2, 0.0);
        self.fft_imaginary.resize(maximum_size / 2, 0.0);
        let packed_len = config.fft_sizes[..config.band_count]
            .iter()
            .map(|size| size / 2 + 1)
            .sum();
        self.packed_magnitudes.resize(packed_len, 0.0);
        for band in &mut self.bands[..config.band_count] {
            band.ingest(history);
        }
        self.samples_since_analysis = usize::MAX / 2;
        self.frames_until_analysis = 0;
    }

    pub fn ingest(&mut self, samples: &[f32]) {
        let band_count = self.config.map_or(0, |config| config.band_count);
        for band in &mut self.bands[..band_count] {
            band.ingest(samples);
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

    pub fn analyze_if_due(&mut self, force: bool, display_divisor: u32) -> Option<&[f32]> {
        let config = self.config?;
        let divisor = display_divisor.clamp(1, 4);
        if self.display_divisor != divisor {
            self.display_divisor = divisor;
            self.frames_until_analysis = 0;
        }
        if !force && self.frames_until_analysis > 0 {
            self.frames_until_analysis -= 1;
            return None;
        }
        if !force && self.samples_since_analysis == 0 {
            return None;
        }
        self.frames_until_analysis = divisor - 1;
        self.samples_since_analysis = 0;

        let mut offset = 0;
        for band in &self.bands[..config.band_count] {
            let count = band.fft_size;
            band.copy_windowed(&mut self.windowed[..count]);
            let magnitude_count = count / 2 + 1;
            let magnitudes = &mut self.packed_magnitudes[offset..offset + magnitude_count];
            band.fft.magnitudes(
                &self.windowed[..count],
                &mut self.fft_real[..count / 2],
                &mut self.fft_imaginary[..count / 2],
                magnitudes,
            );
            let normalization = 2.0 / band.window_sum.max(0.000_001);
            for magnitude in magnitudes {
                *magnitude *= normalization;
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
    /// Two copies of the delay line, so the newest `taps.len()` samples are
    /// always one contiguous slice starting at `delay_position`.
    delay: Vec<f32>,
    delay_position: usize,
    decimation_phase: usize,
    ring: Vec<f32>,
    ring_position: usize,
    ring_filled: usize,
    window: Vec<f32>,
    window_sum: f32,
    fft: RealFft,
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
            fft: RealFft::new(2),
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
        let delay = vec![0.0; taps.len() * 2];
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
            fft: RealFft::new(fft_size),
        }
    }

    fn ingest(&mut self, samples: &[f32]) {
        if self.decimation == 1 {
            for &sample in samples {
                self.push_decimated(sample);
            }
            return;
        }

        let length = self.taps.len();
        for &sample in samples {
            self.delay_position = self.delay_position.checked_sub(1).unwrap_or(length - 1);
            self.delay[self.delay_position] = sample;
            self.delay[self.delay_position + length] = sample;
            self.decimation_phase += 1;
            if self.decimation_phase < self.decimation {
                continue;
            }
            self.decimation_phase = 0;

            let newest_first = &self.delay[self.delay_position..self.delay_position + length];
            let filtered = newest_first
                .iter()
                .zip(&self.taps)
                .fold(0.0, |sum, (sample, weight)| sum + sample * weight);
            self.push_decimated(filtered);
        }
    }

    fn push_decimated(&mut self, sample: f32) {
        self.ring[self.ring_position] = sample;
        self.ring_position += 1;
        if self.ring_position == self.fft_size {
            self.ring_position = 0;
        }
        self.ring_filled = (self.ring_filled + 1).min(self.fft_size);
    }

    fn copy_windowed(&self, output: &mut [f32]) {
        let missing = self.fft_size - self.ring_filled;
        output[..missing].fill(0.0);
        let oldest = (self.ring_position + self.fft_size - self.ring_filled) % self.fft_size;
        let (wrapped, unwrapped) = self.ring.split_at(oldest);
        let valid = &mut output[missing..];
        let first = unwrapped.len().min(valid.len());
        let (start, end) = valid.split_at_mut(first);
        start.copy_from_slice(&unwrapped[..first]);
        end.copy_from_slice(&wrapped[..end.len()]);
        for (sample, weight) in output.iter_mut().zip(&self.window) {
            *sample *= weight;
        }
    }
}

/// Spectrum magnitudes of real input, computed with a complex FFT of half the
/// size. Real and imaginary parts are kept apart so the butterflies vectorize.
struct RealFft {
    bit_reversed: Vec<u32>,
    /// Entries `h..2 * h` hold the rotations for butterflies spanning `2 * h`.
    rotation_real: Vec<f32>,
    rotation_imaginary: Vec<f32>,
    /// Rotations that separate the half-size result into the real input's spectrum.
    split_real: Vec<f32>,
    split_imaginary: Vec<f32>,
}

impl RealFft {
    fn new(size: usize) -> Self {
        debug_assert!(size.is_power_of_two() && size >= 2);
        let half = size / 2;
        let bits = half.trailing_zeros();
        let bit_reversed = (0..half as u32)
            .map(|index| index.reverse_bits().checked_shr(32 - bits).unwrap_or(0))
            .collect();
        let rotation = |turn: f64| {
            let angle = -std::f64::consts::TAU * turn;
            (angle.cos() as f32, angle.sin() as f32)
        };
        let (mut rotation_real, mut rotation_imaginary) = (vec![1.0; half], vec![0.0; half]);
        let mut span = 1;
        while span < half {
            for index in 0..span {
                let (real, imaginary) = rotation(index as f64 / (2 * span) as f64);
                rotation_real[span + index] = real;
                rotation_imaginary[span + index] = imaginary;
            }
            span *= 2;
        }
        let (split_real, split_imaginary) = (0..half)
            .map(|index| rotation(index as f64 / size as f64))
            .unzip();
        Self {
            bit_reversed,
            rotation_real,
            rotation_imaginary,
            split_real,
            split_imaginary,
        }
    }

    /// Writes `input.len() / 2 + 1` magnitudes. `real` and `imaginary` are scratch space.
    fn magnitudes(
        &self,
        input: &[f32],
        real: &mut [f32],
        imaginary: &mut [f32],
        output: &mut [f32],
    ) {
        let half = input.len() / 2;
        for (index, &source) in self.bit_reversed.iter().enumerate() {
            real[index] = input[source as usize * 2];
            imaginary[index] = input[source as usize * 2 + 1];
        }

        let mut span = 1;
        while span < half {
            let turn_real = &self.rotation_real[span..span * 2];
            let turn_imaginary = &self.rotation_imaginary[span..span * 2];
            for (block_real, block_imaginary) in real
                .chunks_exact_mut(span * 2)
                .zip(imaginary.chunks_exact_mut(span * 2))
            {
                let (even_real, odd_real) = block_real.split_at_mut(span);
                let (even_imaginary, odd_imaginary) = block_imaginary.split_at_mut(span);
                for index in 0..span {
                    let turned_real = odd_real[index] * turn_real[index]
                        - odd_imaginary[index] * turn_imaginary[index];
                    let turned_imaginary = odd_real[index] * turn_imaginary[index]
                        + odd_imaginary[index] * turn_real[index];
                    odd_real[index] = even_real[index] - turned_real;
                    odd_imaginary[index] = even_imaginary[index] - turned_imaginary;
                    even_real[index] += turned_real;
                    even_imaginary[index] += turned_imaginary;
                }
            }
            span *= 2;
        }

        output[0] = (real[0] + imaginary[0]).abs();
        output[half] = (real[0] - imaginary[0]).abs();
        for bin in 1..half {
            let mirror = half - bin;
            let even_real = (real[bin] + real[mirror]) * 0.5;
            let even_imaginary = (imaginary[bin] - imaginary[mirror]) * 0.5;
            let odd_real = (imaginary[bin] + imaginary[mirror]) * 0.5;
            let odd_imaginary = (real[mirror] - real[bin]) * 0.5;
            let result_real = even_real + odd_real * self.split_real[bin]
                - odd_imaginary * self.split_imaginary[bin];
            let result_imaginary = even_imaginary
                + odd_real * self.split_imaginary[bin]
                + odd_imaginary * self.split_real[bin];
            output[bin] = (result_real * result_real + result_imaginary * result_imaginary).sqrt();
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

#[cfg(test)]
mod tests {
    use super::{Band, MultirateConfig, MultirateFft, RealFft};

    fn test_signal(count: usize) -> Vec<f32> {
        let mut noise = 0x2545_f491u32;
        (0..count)
            .map(|index| {
                noise ^= noise << 13;
                noise ^= noise >> 17;
                noise ^= noise << 5;
                let time = index as f32 / 48_000.0;
                0.5 * (std::f32::consts::TAU * 997.0 * time).sin()
                    + 0.2 * (std::f32::consts::TAU * 83.0 * time).sin()
                    + 0.1 * (noise as f32 / u32::MAX as f32 - 0.5)
            })
            .collect()
    }

    fn fft_magnitudes(input: &[f32]) -> Vec<f32> {
        let mut real = vec![0.0; input.len() / 2];
        let mut imaginary = vec![0.0; input.len() / 2];
        let mut output = vec![0.0; input.len() / 2 + 1];
        RealFft::new(input.len()).magnitudes(input, &mut real, &mut imaginary, &mut output);
        output
    }

    #[test]
    fn fft_places_exact_tone_in_expected_bin() {
        let size = 128;
        let bin = 7;
        let input = (0..size)
            .map(|index| {
                (2.0 * std::f32::consts::PI * bin as f32 * index as f32 / size as f32).cos()
            })
            .collect::<Vec<_>>();
        let magnitudes = fft_magnitudes(&input);
        let strongest = (0..magnitudes.len())
            .max_by(|&left, &right| magnitudes[left].total_cmp(&magnitudes[right]))
            .unwrap();
        assert_eq!(strongest, bin);
        assert!((magnitudes[bin] - size as f32 / 2.0).abs() < 0.001);
    }

    #[test]
    fn fft_matches_a_direct_transform_at_every_bin() {
        for size in [2, 4, 8, 64, 512, 2048] {
            let input = test_signal(size);
            let magnitudes = fft_magnitudes(&input);
            let peak = magnitudes.iter().copied().fold(0.0, f32::max);
            for (bin, &magnitude) in magnitudes.iter().enumerate() {
                let (mut real, mut imaginary) = (0.0f64, 0.0f64);
                for (index, &sample) in input.iter().enumerate() {
                    let angle = std::f64::consts::TAU * (bin * index) as f64 / size as f64;
                    real += f64::from(sample) * angle.cos();
                    imaginary -= f64::from(sample) * angle.sin();
                }
                let expected = real.hypot(imaginary) as f32;
                assert!(
                    (magnitude - expected).abs() <= peak * 2.0e-5 + 1.0e-6,
                    "size {size}, bin {bin}: {magnitude} instead of {expected}"
                );
            }
        }
    }

    #[test]
    fn filtering_does_not_depend_on_how_audio_is_split_into_blocks() {
        let input = test_signal(6000);
        for decimation in [1, 2, 4, 16, 64, 128] {
            let mut whole = Band::new(256, decimation, 0, 8.0, 8.0);
            whole.ingest(&input);
            let mut split = Band::new(256, decimation, 0, 8.0, 8.0);
            let mut rest = input.as_slice();
            let mut block = 1;
            while !rest.is_empty() {
                let (now, later) = rest.split_at(block.min(rest.len()));
                split.ingest(now);
                rest = later;
                block = block % 397 + 13;
            }
            let (mut expected, mut actual) = (vec![0.0; 256], vec![0.0; 256]);
            whole.copy_windowed(&mut expected);
            split.copy_windowed(&mut actual);
            assert_eq!(expected, actual, "decimation {decimation}");
        }
    }

    #[test]
    fn decimated_samples_equal_the_filter_applied_directly() {
        let input = test_signal(2000);
        let decimation = 4;
        let mut band = Band::new(64, decimation, 0, 8.0, 8.0);
        band.ingest(&input);
        let expected: Vec<f32> = (1..=input.len() / decimation)
            .map(|output| {
                let newest = output * decimation - 1;
                band.taps
                    .iter()
                    .enumerate()
                    .fold(0.0, |sum, (tap, weight)| {
                        sum + newest.checked_sub(tap).map_or(0.0, |index| input[index]) * weight
                    })
            })
            .collect();
        let start = expected.len() - 64;
        let newest_first_position = band.ring_position;
        let actual: Vec<f32> = (0..64)
            .map(|index| band.ring[(newest_first_position + index) % 64])
            .collect();
        assert_eq!(&expected[start..], actual.as_slice());
    }

    #[test]
    fn a_partly_filled_band_pads_its_oldest_samples_with_silence() {
        let mut band = Band::new(8, 1, 0, 8.0, 8.0);
        band.window.fill(1.0);
        band.ingest(&[1.0, 2.0, 3.0]);
        let mut output = [9.0; 8];
        band.copy_windowed(&mut output);
        assert_eq!(output, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
        band.ingest(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        band.copy_windowed(&mut output);
        assert_eq!(output, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn fractional_display_rates_keep_ingesting_between_fft_updates() {
        for divisor in 1..=4 {
            let config = MultirateConfig {
                band_count: 1,
                fft_sizes: [128, 0, 0, 0, 0],
                decimations: [1, 0, 0, 0, 0],
                window_kind: 0,
                kaiser_beta: 8.0,
                flanagan_alpha: 8.0,
            };
            let mut fft = MultirateFft::default();
            fft.reset(config, &[0.0; 128]);
            for frame in 0..13 {
                fft.ingest(&[1.0; 128]);
                let result = fft.analyze_if_due(frame == 0, divisor);
                assert_eq!(
                    result.is_some(),
                    frame % divisor == 0,
                    "frame {frame}, divisor {divisor}"
                );
                if let Some(magnitudes) = result {
                    assert!((magnitudes[0] - 2.0).abs() < 0.001);
                }
            }
            // Changing cadence and reset requests take effect immediately.
            fft.ingest(&[0.0]);
            assert!(
                fft.analyze_if_due(false, if divisor == 4 { 1 } else { divisor + 1 })
                    .is_some()
            );
            assert!(fft.analyze_if_due(true, 4).is_some());
        }
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
        };
        let mut fft = MultirateFft::default();
        fft.reset(config, &[0.0; 128]);

        assert!(fft.analyze_if_due(true, 1).is_some());
        assert!(fft.analyze_if_due(false, 1).is_none());
        fft.ingest(&[0.25]);
        assert!(fft.analyze_if_due(false, 1).is_some());
        assert!(fft.analyze_if_due(false, 1).is_none());
    }
}
