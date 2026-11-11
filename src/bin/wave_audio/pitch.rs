use std::f64::consts::TAU;

use visual_audio::waveform::Trace;

pub const NOTES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];
pub const DEFAULT_FILTER_DELAY_MS: u32 = 100;
const SILENCE_AMPLITUDE: f64 = 1e-5;

pub fn frequency(midi: u8) -> f64 {
    440.0 * 2.0_f64.powf((f64::from(midi) - 69.0) / 12.0)
}

#[derive(Default)]
struct FilterState {
    first: f64,
    second: f64,
}

struct NoteBand {
    midi: u8,
    pitch_class: usize,
    b: f64,
    a1: f64,
    a2: f64,
    stages: [FilterState; 2],
    nominal_period: f64,
    period: Option<f64>,
    previous_sample: f64,
    crossing: Option<f64>,
    block_power: f64,
    power: f64,
}

impl NoteBand {
    fn new(midi: u8, sample_rate: u32, max_delay_ms: u32) -> Self {
        let sample_rate = f64::from(sample_rate);
        let period = sample_rate / frequency(midi);
        let omega = TAU / period;
        // The two bandpasses have a combined group delay of 2 / alpha samples
        // at the center frequency. Widen only the bands that exceed the limit.
        let min_alpha = 2_000.0 / (sample_rate * f64::from(max_delay_ms));
        let alpha = (omega.sin() / (2.0 * 24.0)).max(min_alpha);
        let a0 = 1.0 + alpha;
        Self {
            midi,
            pitch_class: usize::from(midi) % NOTES.len(),
            b: alpha / a0,
            a1: -2.0 * omega.cos() / a0,
            a2: (1.0 - alpha) / a0,
            stages: Default::default(),
            nominal_period: period,
            period: None,
            previous_sample: 0.0,
            crossing: None,
            block_power: 0.0,
            power: 0.0,
        }
    }

    fn filter(&mut self, mut sample: f64, index: u64) -> f64 {
        for state in &mut self.stages {
            let scaled = self.b * sample;
            let output = scaled + state.first;
            state.first = state.second - self.a1 * output;
            state.second = -scaled - self.a2 * output;
            sample = output;
        }
        self.block_power += sample * sample;
        if self.previous_sample < 0.0 && sample >= 0.0 {
            let crossing = index as f64 - sample / (sample - self.previous_sample);
            self.period = self
                .crossing
                .map(|previous| crossing - previous)
                .filter(|period| {
                    // Allow detuning within the note, while rejecting a neighbor that leaked through.
                    (0.97 * self.nominal_period..=1.03 * self.nominal_period).contains(period)
                });
            self.crossing = Some(crossing);
        }
        self.previous_sample = sample;
        sample
    }

    fn reset(&mut self) {
        self.stages = Default::default();
        self.period = None;
        self.previous_sample = 0.0;
        self.crossing = None;
        self.block_power = 0.0;
        self.power = 0.0;
    }
}

#[derive(Clone, Copy)]
pub struct Trigger {
    pub midi: u8,
    pub end: f64,
    pub period: f64,
}

#[derive(Clone, Copy, Default)]
pub struct PitchFrame {
    pub triggers: [Option<Trigger>; NOTES.len()],
    pub silent: bool,
}

pub struct PitchHistory {
    bands: Vec<NoteBand>,
    traces: [Trace; NOTES.len()],
    frames: Vec<PitchFrame>,
    start: u64,
    next_sample: u64,
    block_size: u64,
    power_smoothing: f64,
    silence_samples: u64,
    quiet_samples: u64,
}

impl PitchHistory {
    pub fn new(sample_rate: u32, start_sample: u64, max_delay_ms: u32) -> Self {
        let block_size = u64::from(sample_rate / 200).max(1);
        let history_size = 4 * sample_rate as usize;
        let block_seconds = block_size as f64 / f64::from(sample_rate);
        Self {
            bands: (36..=95) // C2 through B6
                .take_while(|&midi| frequency(midi) < 0.45 * f64::from(sample_rate))
                .map(|midi| NoteBand::new(midi, sample_rate, max_delay_ms))
                .collect(),
            traces: std::array::from_fn(|_| Trace::new(history_size, start_sample)),
            frames: vec![PitchFrame::default(); history_size.div_ceil(block_size as usize)],
            start: start_sample,
            next_sample: start_sample,
            power_smoothing: 1.0 - (-block_seconds / 0.03).exp(),
            block_size,
            silence_samples: (u64::from(sample_rate) * 15).div_ceil(1000).max(1),
            quiet_samples: 0,
        }
    }

    pub fn set_delay_limit(&mut self, sample_rate: u32, max_delay_ms: u32) {
        for band in &mut self.bands {
            let tuned = NoteBand::new(band.midi, sample_rate, max_delay_ms);
            if tuned.b != band.b {
                *band = tuned;
            }
        }
    }

    pub fn ingest(&mut self, samples: &[f32]) {
        for &sample in samples {
            let sample = if sample.is_finite() {
                f64::from(sample)
            } else {
                0.0
            };
            self.quiet_samples = if sample.abs() > SILENCE_AMPLITUDE {
                0
            } else {
                self.quiet_samples.saturating_add(1)
            };
            let mut filtered = [0.0; NOTES.len()];
            if self.quiet_samples < self.silence_samples {
                for band in &mut self.bands {
                    filtered[band.pitch_class] += band.filter(sample, self.next_sample);
                }
            } else if self.quiet_samples == self.silence_samples {
                // Stop the filter tails and release phase lock when the input has gone quiet.
                for band in &mut self.bands {
                    band.reset();
                }
            }
            for (trace, value) in self.traces.iter_mut().zip(filtered) {
                trace.push(value as f32);
            }
            self.next_sample += 1;
            let elapsed = self.next_sample - self.start;
            if elapsed.is_multiple_of(self.block_size) {
                self.save_frame(elapsed / self.block_size - 1);
            }
        }
    }

    fn save_frame(&mut self, block: u64) {
        let mut strongest = [0.0; NOTES.len()];
        let mut triggers = [None; NOTES.len()];
        let end = (self.next_sample - 1) as f64;
        for band in &mut self.bands {
            let mean = band.block_power / self.block_size as f64;
            band.block_power = 0.0;
            band.power += (mean - band.power) * self.power_smoothing;
            if band.power > 1e-10
                && band.power > strongest[band.pitch_class]
                && let Some(crossing) = band.crossing
                && let Some(period) = band.period
                && end - crossing <= 2.0 * period
            {
                strongest[band.pitch_class] = band.power;
                triggers[band.pitch_class] = Some(Trigger {
                    midi: band.midi,
                    end: crossing,
                    period,
                });
            }
        }
        let slot = block as usize % self.frames.len();
        self.frames[slot] = PitchFrame {
            triggers,
            silent: self.quiet_samples >= self.silence_samples,
        };
    }

    pub fn frame_at(&self, end: f64) -> PitchFrame {
        let elapsed = end + 1.0 - self.start as f64;
        let blocks = (elapsed / self.block_size as f64).floor() as u64;
        let completed = (self.next_sample - self.start) / self.block_size;
        if blocks == 0 || blocks > completed || completed - blocks >= self.frames.len() as u64 {
            return PitchFrame::default();
        }
        self.frames[(blocks - 1) as usize % self.frames.len()]
    }

    pub fn trace(&self, pitch_class: usize) -> &Trace {
        &self.traces[pitch_class]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(notes: &[u8], sample_rate: u32) -> Vec<f32> {
        (0..sample_rate)
            .map(|i| {
                notes
                    .iter()
                    .map(|&midi| {
                        (TAU * frequency(midi) * f64::from(i) / f64::from(sample_rate)).sin() as f32
                            * 0.25
                    })
                    .sum()
            })
            .collect()
    }

    fn measure(notes: &[u8], sample_rate: u32) -> [f32; 12] {
        let mut history = PitchHistory::new(sample_rate, 0, DEFAULT_FILTER_DELAY_MS);
        let samples = input(notes, sample_rate);
        for chunk in samples.chunks(317) {
            history.ingest(chunk);
        }
        let count = i64::from(sample_rate / 10);
        let end = i64::from(sample_rate);
        std::array::from_fn(|note| {
            let trace = history.trace(note);
            let energy: f32 = (end - count..end)
                .map(|index| trace.get(index).unwrap().powi(2))
                .sum();
            (energy / count as f32).sqrt()
        })
    }

    #[test]
    fn separates_adjacent_semitones_and_folds_octaves() {
        for sample_rate in [44_100, 48_000] {
            for midi in [36, 37, 45, 69, 93] {
                let levels = measure(&[midi], sample_rate);
                let pitch_class = usize::from(midi % 12);
                for (note, level) in levels.into_iter().enumerate() {
                    if note == pitch_class {
                        assert!(level > 0.15, "{midi}: {levels:?}");
                    } else {
                        // Widened bass bands admit more of their nearest neighbors.
                        let adjacent = matches!((note + 12 - pitch_class) % 12, 1 | 11);
                        let max_leakage = if midi < 53 && adjacent { 0.085 } else { 0.04 };
                        assert!(level < max_leakage, "{midi}: {levels:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn shows_chord_notes_at_the_same_time() {
        let levels = measure(&[60, 64, 67], 48_000);
        for (note, level) in levels.into_iter().enumerate() {
            if [0, 4, 7].contains(&note) {
                assert!(level > 0.15, "{levels:?}");
            } else {
                assert!(level < 0.05, "{levels:?}");
            }
        }
    }

    #[test]
    fn locks_lanes_to_independent_periods_at_the_requested_time() {
        let sample_rate = 48_000;
        let start = 1000;
        let mut history = PitchHistory::new(sample_rate, start, DEFAULT_FILTER_DELAY_MS);
        let a_hz = 221.0;
        let c_hz = frequency(60);
        let samples: Vec<f32> = (0..sample_rate)
            .map(|sample| {
                let phase = TAU * f64::from(sample) / f64::from(sample_rate);
                (0.4 * (a_hz * phase).sin()
                    + 0.1 * (440.0 * phase).sin()
                    + 0.3 * (c_hz * phase).sin()) as f32
            })
            .collect();
        for chunk in samples.chunks(317) {
            history.ingest(chunk);
        }
        for end in [47_053.25, 48_901.75] {
            let triggers = history.frame_at(end).triggers;
            for (note, midi, hz) in [(9, 57, a_hz), (0, 60, c_hz)] {
                let trigger = triggers[note].unwrap();
                assert_eq!(trigger.midi, midi);
                let period = f64::from(sample_rate) / hz;
                assert!((trigger.period - period).abs() < 0.5);
                assert!(trigger.end <= end && end - trigger.end < period + 240.0);
            }
        }
        assert!(
            history
                .frame_at(start as f64 - 1.0)
                .triggers
                .iter()
                .all(Option::is_none)
        );
    }

    #[test]
    fn low_note_stops_ringing_and_releases_the_trigger_after_silence() {
        let sample_rate = 48_000;
        let tone = input(&[36], sample_rate);
        let mut history = PitchHistory::new(sample_rate, 0, DEFAULT_FILTER_DELAY_MS);
        history.ingest(&tone);
        let active_end = f64::from(sample_rate - 1);
        assert!(history.frame_at(active_end).triggers[0].is_some());

        history.ingest(&[0.0; 480]);
        assert!(!history.frame_at(active_end + 480.0).silent);
        history.ingest(&[0.0; 480]);
        let quiet_end = active_end + 960.0;
        let quiet = history.frame_at(quiet_end);
        assert!(quiet.silent && quiet.triggers.iter().all(Option::is_none));
        assert_eq!(history.trace(0).get(quiet_end as i64), Some(0.0));
        assert!(history.frame_at(active_end).triggers[0].is_some());

        history.ingest(&tone[..4800]);
        assert!(!history.frame_at(quiet_end + 4800.0).silent);
        assert!(history.frame_at(quiet_end + 4800.0).triggers[0].is_some());
    }

    #[test]
    fn preserves_the_signed_waveform_of_two_octaves_at_their_capture_times() {
        let sample_rate = 48_000;
        let start = 1000;
        let mut history = PitchHistory::new(sample_rate, start, DEFAULT_FILTER_DELAY_MS);
        let samples = input(&[69, 81], sample_rate);
        for chunk in samples.chunks(317) {
            history.ingest(chunk);
        }
        let trace = history.trace(9);
        for index in 47_000..48_000 {
            let filtered = trace.get(start as i64 + index).unwrap();
            let original = samples[index as usize];
            assert!(
                (filtered - original).abs() < 0.005,
                "{filtered} vs {original}"
            );
        }
        assert_eq!(trace.get(start as i64 - 1), None);
        assert_eq!(trace.get(start as i64 + i64::from(sample_rate)), None);
    }

    #[test]
    fn delay_limit_speeds_up_bass_onset() {
        let sample_rate = 48_000;
        let tone = input(&[36], sample_rate);
        let half_amplitude_ms = |limit| {
            let mut history = PitchHistory::new(sample_rate, 0, 250);
            history.set_delay_limit(sample_rate, limit);
            history.ingest(&tone);
            let trace = history.trace(0);
            let index = (0..i64::from(sample_rate))
                .find(|&index| trace.get(index).unwrap().abs() >= 0.125)
                .unwrap();
            index as f64 * 1000.0 / f64::from(sample_rate)
        };
        let fast = half_amplitude_ms(DEFAULT_FILTER_DELAY_MS);
        let original = half_amplitude_ms(250);
        assert!(half_amplitude_ms(60) < fast);
        assert!(
            fast < 100.0 && original > 170.0,
            "{fast:.1} ms vs {original:.1} ms"
        );
    }
}
