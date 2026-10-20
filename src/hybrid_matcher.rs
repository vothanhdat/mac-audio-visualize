use std::f32::consts::PI;

use crate::multirate_fft::MultirateConfig;

const SILENCE_GATE_DB: f32 = -75.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridMatchConfig {
    pub frequencies_hz: [f32; 6],
    pub q_50_hz: f32,
    pub q_10_khz: f32,
    pub adapt_seconds: f32,
    pub max_correction_db: f32,
}

#[derive(Clone, Copy, Default)]
struct ReferenceResonator {
    oscillator: [f32; 2],
    value: [f32; 2],
}

impl ReferenceResonator {
    fn reset(&mut self) {
        self.oscillator = [1.0, 0.0];
        self.value = [0.0, 0.0];
    }

    fn process(&mut self, samples: &[f32], frequency: f32, q: f32, sample_rate: f32) {
        let phase = 2.0 * PI * frequency / sample_rate;
        let rotation = [phase.cos(), -phase.sin()];
        let damping = (-PI * frequency / (q.max(0.01) * sample_rate)).exp();
        for &sample in samples {
            let mixed = [self.oscillator[0] * sample, self.oscillator[1] * sample];
            self.value[0] = damping * self.value[0] + (1.0 - damping) * mixed[0];
            self.value[1] = damping * self.value[1] + (1.0 - damping) * mixed[1];
            self.oscillator = [
                self.oscillator[0] * rotation[0] - self.oscillator[1] * rotation[1],
                self.oscillator[0] * rotation[1] + self.oscillator[1] * rotation[0],
            ];
        }
        let length = (self.oscillator[0] * self.oscillator[0]
            + self.oscillator[1] * self.oscillator[1])
            .sqrt()
            .max(0.000_001);
        self.oscillator[0] /= length;
        self.oscillator[1] /= length;
    }

    fn magnitude(self) -> f32 {
        2.0 * (self.value[0] * self.value[0] + self.value[1] * self.value[1]).sqrt()
    }
}

#[derive(Default)]
pub struct HybridBandMatcher {
    config: Option<HybridMatchConfig>,
    references: [ReferenceResonator; 3],
    half_correction_db: f32,
    full_relative_db: f32,
    elapsed_since_update: f32,
}

impl HybridBandMatcher {
    pub fn reset(&mut self, config: HybridMatchConfig, history: &[f32], sample_rate: f32) {
        self.config = Some(config);
        self.half_correction_db = 0.0;
        self.full_relative_db = 0.0;
        self.elapsed_since_update = 0.0;
        for (index, reference) in self.references.iter_mut().enumerate() {
            reference.reset();
            reference.process(
                history,
                config.frequencies_hz[index],
                q_at(config, config.frequencies_hz[index]),
                sample_rate,
            );
        }
    }

    pub fn requires_reset(&self, config: HybridMatchConfig) -> bool {
        self.config != Some(config)
    }

    pub fn ingest(&mut self, samples: &[f32], sample_rate: f32) {
        let Some(config) = self.config else {
            return;
        };
        for (index, reference) in self.references.iter_mut().enumerate() {
            reference.process(
                samples,
                config.frequencies_hz[index],
                q_at(config, config.frequencies_hz[index]),
                sample_rate,
            );
        }
        self.elapsed_since_update += samples.len() as f32 / sample_rate;
    }

    pub fn update(
        &mut self,
        magnitudes: &[f32],
        fft_config: MultirateConfig,
        sample_rate: f32,
        elapsed_seconds: f32,
    ) {
        let Some(config) = self.config else {
            return;
        };
        let elapsed_seconds = elapsed_seconds.max(self.elapsed_since_update);
        self.elapsed_since_update = 0.0;
        let alpha = 1.0 - (-elapsed_seconds / config.adapt_seconds.max(0.01)).exp();
        let mut half_error_sum = 0.0;
        let mut half_weight_sum = 0.0;
        for (index, reference) in self.references.iter().enumerate() {
            if let Some(half_magnitude) = fft_magnitude_at(
                magnitudes,
                fft_config,
                0,
                config.frequencies_hz[index],
                sample_rate,
            ) {
                let resonator_db = magnitude_db(reference.magnitude());
                let half_db = magnitude_db(half_magnitude);
                let weight = comparison_weight(resonator_db.min(half_db));
                half_error_sum += (resonator_db - half_db) * weight;
                half_weight_sum += weight;
            }
        }
        if half_weight_sum > 0.0001 {
            let target = (half_error_sum / half_weight_sum)
                .clamp(-config.max_correction_db, config.max_correction_db);
            self.half_correction_db += (target - self.half_correction_db) * alpha;
            self.half_correction_db = self
                .half_correction_db
                .clamp(-config.max_correction_db, config.max_correction_db);
        }

        let mut crossover_error_sum = 0.0;
        let mut crossover_weight_sum = 0.0;
        for &frequency in &config.frequencies_hz[3..] {
            if let (Some(half_magnitude), Some(full_magnitude)) = (
                fft_magnitude_at(magnitudes, fft_config, 0, frequency, sample_rate),
                fft_magnitude_at(magnitudes, fft_config, 1, frequency, sample_rate),
            ) {
                let half_db = magnitude_db(half_magnitude);
                let full_db = magnitude_db(full_magnitude);
                let weight = comparison_weight(half_db.min(full_db));
                crossover_error_sum += (half_db - full_db) * weight;
                crossover_weight_sum += weight;
            }
        }
        if crossover_weight_sum > 0.0001 {
            let target_relative = (crossover_error_sum / crossover_weight_sum)
                .clamp(-config.max_correction_db, config.max_correction_db);
            self.full_relative_db += (target_relative - self.full_relative_db) * alpha;
            self.full_relative_db = self
                .full_relative_db
                .clamp(-config.max_correction_db, config.max_correction_db);
        }
    }

    pub fn corrections_db(&self) -> [f32; 4] {
        let full_correction = (self.half_correction_db + self.full_relative_db).clamp(
            self.config.map_or(0.0, |config| -config.max_correction_db),
            self.config.map_or(0.0, |config| config.max_correction_db),
        );
        [
            self.half_correction_db,
            self.half_correction_db,
            full_correction,
            full_correction,
        ]
    }

    pub fn clear(&mut self) {
        self.config = None;
        self.half_correction_db = 0.0;
        self.full_relative_db = 0.0;
        self.elapsed_since_update = 0.0;
    }
}

fn q_at(config: HybridMatchConfig, frequency: f32) -> f32 {
    let t = ((frequency.sqrt() - 50.0_f32.sqrt()) / (10_000.0_f32.sqrt() - 50.0_f32.sqrt()))
        .clamp(0.0, 1.0);
    config.q_50_hz + (config.q_10_khz - config.q_50_hz) * t
}

fn magnitude_db(magnitude: f32) -> f32 {
    20.0 * magnitude.max(0.000_001).log10()
}

fn comparison_weight(db: f32) -> f32 {
    let normalized = ((db - SILENCE_GATE_DB) / 20.0).clamp(0.0, 1.0);
    normalized * normalized * (3.0 - 2.0 * normalized)
}

fn fft_magnitude_at(
    magnitudes: &[f32],
    config: MultirateConfig,
    band: usize,
    frequency: f32,
    sample_rate: f32,
) -> Option<f32> {
    if band >= config.band_count {
        return None;
    }
    let size = config.fft_sizes[band];
    let effective_rate = sample_rate / config.decimations[band] as f32;
    if size < 2 || frequency >= effective_rate * 0.5 {
        return None;
    }
    let offset: usize = config.fft_sizes[..band]
        .iter()
        .map(|size| size / 2 + 1)
        .sum();
    let fractional_bin = frequency * size as f32 / effective_rate;
    let lower = (fractional_bin.floor() as usize).min(size / 2);
    let upper = (lower + 1).min(size / 2);
    let blend = fractional_bin.fract();
    let low = *magnitudes.get(offset + lower)?;
    let high = *magnitudes.get(offset + upper)?;
    Some(low + (high - low) * blend)
}

#[cfg(test)]
mod tests {
    use super::{HybridBandMatcher, HybridMatchConfig};
    use crate::multirate_fft::{MultirateConfig, MultirateFft};

    #[test]
    fn matcher_moves_corrections_toward_reference_levels() {
        let sample_rate = 44_100.0;
        let history: Vec<f32> = (0..16_384)
            .map(|index| {
                let phase_0 = 2.0 * std::f32::consts::PI * 4500.0 * index as f32 / sample_rate;
                let phase_1 = 2.0 * std::f32::consts::PI * 7800.0 * index as f32 / sample_rate;
                0.2 * phase_0.sin() + 0.2 * phase_1.sin()
            })
            .collect();
        let fft_config = MultirateConfig {
            band_count: 2,
            fft_sizes: [1024, 512, 0, 0, 0],
            decimations: [2, 1, 1, 1, 1],
            window_kind: 0,
            kaiser_beta: 8.0,
            flanagan_alpha: 8.0,
            analysis_rate_hz: 120.0,
        };
        let match_config = HybridMatchConfig {
            frequencies_hz: [4500.0, 4700.0, 4900.0, 7400.0, 7800.0, 8000.0],
            q_50_hz: 10.0,
            q_10_khz: 300.0,
            adapt_seconds: 1.0,
            max_correction_db: 12.0,
        };
        let mut fft = MultirateFft::default();
        fft.reset(fft_config, &history);
        let mut magnitudes = fft
            .analyze_if_due(sample_rate, true, true)
            .unwrap()
            .to_vec();
        for magnitude in &mut magnitudes[..=fft_config.fft_sizes[0] / 2] {
            *magnitude *= 10.0;
        }
        let mut matcher = HybridBandMatcher::default();
        matcher.reset(match_config, &history, sample_rate);
        matcher.update(&magnitudes, fft_config, sample_rate, 1.0);
        let corrections = matcher.corrections_db();
        assert!(corrections[0] < 0.0);
        assert_eq!(corrections[0], corrections[1]);
        assert_eq!(corrections[2], corrections[3]);
    }
}
