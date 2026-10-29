//! Short-window pitch-class energy, independent of the spectrum's resonator Q,
//! display bin spacing, envelope, and optional FFT frame divisor.
use crate::multirate_fft::{MultirateConfig, MultirateFft};

#[derive(Default)]
pub(super) struct FastNotes {
    fft: [MultirateFft; 2],
    power: [[f32; 12]; 2],
    sample_rate: f32,
    range: [f32; 2],
}

impl FastNotes {
    pub(super) fn update(
        &mut self,
        samples: [&[f32]; 2],
        new_count: usize,
        sample_rate: f32,
        range: [f32; 2],
        reset: bool,
    ) -> [f32; 12] {
        // 2048 samples at 48 kHz, scaling with the input rate. A Hann window
        // favors the middle of this ~43 ms span; there is no extra note hold.
        let size = ((2048.0 * sample_rate / 48000.0).round() as usize)
            .next_power_of_two()
            .clamp(256, 16384);
        let config = MultirateConfig {
            band_count: 1,
            fft_sizes: [size, 2, 2, 2, 2],
            decimations: [1; 5],
            window_kind: 0,
            kaiser_beta: 8.1,
            flanagan_alpha: 8.0,
        };
        let force = reset || self.sample_rate != sample_rate || self.range != range;
        for channel in 0..2 {
            let fft = &mut self.fft[channel];
            let input = samples[channel];
            let restart = force || fft.requires_reset(config);
            if restart {
                fft.reset(config, input);
            } else {
                fft.ingest(&input[input.len() - new_count.min(input.len())..]);
            }
            if let Some(magnitudes) = fft.analyze_if_due(restart, 1) {
                self.power[channel] = pitch_class_power(magnitudes, sample_rate, size, range);
            }
        }
        self.sample_rate = sample_rate;
        self.range = range;
        std::array::from_fn(|note| (self.power[0][note] + self.power[1][note]) * 0.5)
    }
}

fn pitch_class_power(
    magnitudes: &[f32],
    sample_rate: f32,
    size: usize,
    range: [f32; 2],
) -> [f32; 12] {
    let mut result = [0.0; 12];
    let lower = range[0].max(32.7032);
    let upper = range[1].min(3951.0664).min(sample_rate * 0.48);
    if lower > upper {
        return result;
    }
    let bin_hz = sample_rate / size as f32;
    for bin in 1..magnitudes.len() - 1 {
        let magnitude = magnitudes[bin];
        if magnitude < 1e-7 || magnitude <= magnitudes[bin - 1] || magnitude < magnitudes[bin + 1] {
            continue;
        }
        // Interpolate spectral peaks, rather than choosing the nearest FFT
        // bin: at the low end, one bin can span multiple musical semitones.
        let a = magnitudes[bin - 1].max(1e-12).ln();
        let b = magnitude.ln();
        let c = magnitudes[bin + 1].max(1e-12).ln();
        let curvature = a - 2.0 * b + c;
        let offset = if curvature.abs() > 1e-8 {
            (0.5 * (a - c) / curvature).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let hz = (bin as f32 + offset) * bin_hz;
        if hz < lower || hz > upper {
            continue;
        }
        let midi = 69.0 + 12.0 * (hz / 440.0).log2();
        let base = midi.floor() as i32;
        let fraction = midi - base as f32;
        let power = magnitude * magnitude;
        result[base.rem_euclid(12) as usize] += power * (1.0 - fraction);
        result[(base + 1).rem_euclid(12) as usize] += power * fraction;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strongest(power: [f32; 12]) -> usize {
        (0..12)
            .max_by(|&a, &b| power[a].total_cmp(&power[b]))
            .unwrap()
    }

    fn tone(rate: f32, midi: f32, amplitude: f32, count: usize) -> Vec<f32> {
        let hz = 440.0 * 2f64.powf((midi as f64 - 69.0) / 12.0);
        (0..count)
            .map(|n| (std::f64::consts::TAU * hz * n as f64 / rate as f64).sin() as f32 * amplitude)
            .collect()
    }

    #[test]
    fn real_audio_maps_all_pitch_classes_at_common_sample_rates() {
        for rate in [44100.0, 48000.0, 96000.0] {
            let silence = vec![0.0; 16384];
            let mut analyzer = FastNotes::default();
            for midi in 57..84 {
                let audio = tone(rate, midi as f32, 0.5, silence.len());
                let power =
                    analyzer.update([&silence, &audio], audio.len(), rate, [200.0, 2000.0], true);
                assert_eq!(strongest(power), midi % 12, "MIDI {midi} at {rate} Hz");
                assert!(power.iter().all(|p| p.is_finite()));
            }
        }
    }

    #[test]
    fn rapid_arpeggios_and_quieter_notes_take_over_within_fifty_ms() {
        for fps in [60, 120] {
            let rate = 48000.0;
            let chunk = 48000 / fps;
            let mut history = tone(rate, 69.0, 0.8, 16384);
            let silence = vec![0.0; history.len()];
            let mut analyzer = FastNotes::default();
            analyzer.update(
                [&silence, &history],
                history.len(),
                rate,
                [200.0, 2000.0],
                true,
            );
            let mut phase: f64 = 0.0;
            // 180 BPM sixteenth notes: 83 ms per note. Include a much quieter
            // second note to catch lingering energy from the previous winner.
            for (midi, amplitude) in [(73, 0.2), (76, 0.5), (67, 0.4), (64, 0.3), (71, 0.5)] {
                let hz = 440.0 * 2f64.powf((midi as f64 - 69.0) / 12.0);
                let mut first_frame = None;
                for frame in 1..=fps / 12 {
                    history.copy_within(chunk.., 0);
                    let end = history.len();
                    for sample in &mut history[end - chunk..] {
                        *sample = phase.sin() as f32 * amplitude;
                        phase = (phase + std::f64::consts::TAU * hz / rate as f64)
                            % std::f64::consts::TAU;
                    }
                    let power =
                        analyzer.update([&silence, &history], chunk, rate, [200.0, 2000.0], false);
                    if strongest(power) == midi % 12 && first_frame.is_none() {
                        first_frame = Some(frame);
                    }
                }
                let elapsed = first_frame.expect("new note should be detected") as f32 / fps as f32;
                assert!(elapsed <= 0.05, "MIDI {midi} took {elapsed}s at {fps} Hz");
            }
            history.fill(0.0);
            let silent = analyzer.update(
                [&silence, &history],
                history.len(),
                rate,
                [200.0, 2000.0],
                false,
            );
            assert!(
                silent.iter().all(|&p| p < 1e-12),
                "old notes should leave the window completely"
            );
        }
    }

    #[test]
    fn midrange_focus_excludes_loud_bass_and_treble_in_real_audio() {
        let rate = 48000.0;
        let bass = tone(rate, 36.0, 0.8, 16384);
        let treble = tone(rate, 96.0, 0.8, bass.len());
        let lead = tone(rate, 69.0, 0.1, bass.len());
        let audio: Vec<_> = bass
            .iter()
            .zip(&treble)
            .zip(&lead)
            .map(|((&a, &b), &c)| a + b + c)
            .collect();
        let mut analyzer = FastNotes::default();
        let full = analyzer.update([&audio, &audio], audio.len(), rate, [16.0, 22500.0], true);
        assert_eq!(strongest(full), 0);
        let mid = analyzer.update([&audio, &audio], 0, rate, [200.0, 2000.0], false);
        assert_eq!(strongest(mid), 9);
        assert!(mid[0] < mid[9] * 0.05);
        let empty = analyzer.update([&audio, &audio], 0, rate, [2000.0, 200.0], false);
        assert_eq!(empty, [0.0; 12]);
    }
}
