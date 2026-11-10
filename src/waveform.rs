//! Oscilloscope timebases: one raw audio buffer, sampled directly at each x.
use std::f64::consts::TAU;

pub struct Trace {
    points: Vec<f32>,
    start: u64,
    end: u64,
}

impl Trace {
    pub fn new(capacity: usize, start: u64) -> Self {
        Self {
            points: vec![0.0; capacity.max(1)],
            start,
            end: start,
        }
    }

    pub fn push(&mut self, value: f32) {
        let capacity = self.points.len() as u64;
        self.points[(self.end % capacity) as usize] = value;
        self.end += 1;
        self.start = self.start.max(self.end.saturating_sub(capacity));
    }

    pub fn get(&self, index: i64) -> Option<f32> {
        if index < 0 || (index as u64) < self.start || index as u64 >= self.end {
            None
        } else {
            Some(self.points[index as usize % self.points.len()])
        }
    }

    pub fn interpolate(&self, position: f64) -> Option<f32> {
        let index = position.floor() as i64;
        let a = self.get(index)?;
        let fraction = position - position.floor();
        if fraction < 1e-9 {
            return Some(a);
        }
        let b = self.get(index + 1)?;
        Some(a + (b - a) * fraction as f32)
    }

    /// Exact extrema of the piecewise-linear buffer within this interval.
    /// Clip at its boundaries: a sample after a phase-locked window must not
    /// leak into that window just because both land in the same screen pixel.
    pub fn segment_range(&self, from: f64, to: f64) -> Option<(f32, f32)> {
        if self.start == self.end {
            return None;
        }
        let from = from.max(self.start as f64);
        let to = to.min((self.end - 1) as f64);
        if from > to {
            return None;
        }
        let a = self.interpolate(from)?;
        let b = self.interpolate(to)?;
        let mut low = a.min(b);
        let mut high = a.max(b);
        for index in from.ceil() as i64..=to.floor() as i64 {
            let value = self.get(index).unwrap();
            low = low.min(value);
            high = high.max(value);
        }
        Some((low, high))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WaveWindow {
    pub end: f64,
    pub span: f64,
    pub reference_hz: f64,
}

impl WaveWindow {
    /// fraction=0 is the oldest point at screen left, fraction=1 is screen right.
    /// Equivalently, with x measured from right: sample(end - x/W * Fs/F).
    pub fn sample_position(self, fraction_from_left: f64) -> f64 {
        self.end - (1.0 - fraction_from_left) * self.span
    }
}

pub struct WaveformHistory {
    pub raw: Trace,
    pub sample_count: u64,
    sample_rate: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ReferenceSweep {
    pub end: f64,
    pub min_hz: f64,
    pub max_hz: f64,
    pub logarithmic: bool,
    sample_rate: f64,
}

impl ReferenceSweep {
    pub fn frequency_at(self, fraction_from_left: f64) -> f64 {
        let u = fraction_from_left.clamp(0.0, 1.0);
        if self.logarithmic {
            self.min_hz * (self.max_hz / self.min_hz).powf(u)
        } else {
            self.min_hz + (self.max_hz - self.min_hz) * u
        }
    }

    /// One raw-buffer lookup with a spatially varying reference frequency:
    /// sample(end - (1-u) * Fs/F(u)), u increasing from screen left to right.
    pub fn sample_position(self, fraction_from_left: f64) -> f64 {
        let u = fraction_from_left.clamp(0.0, 1.0);
        self.end - (1.0 - u) * (self.sample_rate / self.frequency_at(u))
    }
}

impl WaveformHistory {
    pub fn new(sample_rate: u32) -> Self {
        assert!(sample_rate > 0);
        // At F >= 1 Hz: 1 s window + <1 s phase hold + <=1.5 s output delay.
        Self {
            raw: Trace::new(sample_rate as usize * 4, 0),
            sample_count: 0,
            sample_rate,
        }
    }

    pub fn ingest(&mut self, samples: &[f32]) {
        for &sample in samples {
            self.raw.push(if sample.is_finite() { sample } else { 0.0 });
            self.sample_count += 1;
        }
    }

    pub fn window(&self, available_end: f64, reference_hz: f64, phase_lock: bool) -> WaveWindow {
        let maximum = (self.sample_rate as f64 * 0.45).max(1.0);
        let reference_hz = if reference_hz.is_finite() {
            reference_hz.clamp(1.0, maximum)
        } else {
            60.0_f64.min(maximum)
        };
        let span = self.sample_rate as f64 / reference_hz;
        let end = if phase_lock {
            (available_end / span).floor() * span
        } else {
            available_end
        };
        WaveWindow {
            end,
            span,
            reference_hz,
        }
    }

    pub fn value_at(&self, window: WaveWindow, fraction_from_left: f64) -> Option<f32> {
        self.raw
            .interpolate(window.sample_position(fraction_from_left))
    }

    pub fn sweep(
        &self,
        available_end: f64,
        min_hz: f64,
        max_hz: f64,
        phase_lock: bool,
        logarithmic: bool,
    ) -> ReferenceSweep {
        let anchor = self.window(available_end, min_hz, phase_lock);
        let maximum = (self.sample_rate as f64 * 0.45).max(1.0);
        let max_hz = if max_hz.is_finite() {
            max_hz.clamp(anchor.reference_hz, maximum)
        } else {
            anchor.reference_hz
        };
        ReferenceSweep {
            end: anchor.end,
            min_hz: anchor.reference_hz,
            max_hz,
            logarithmic,
            sample_rate: self.sample_rate as f64,
        }
    }
}

pub fn tone_sample(index: u64, sample_rate: u32, frequency_hz: f64) -> f32 {
    (0.7 * (TAU * frequency_hz * index as f64 / sample_rate as f64).sin()) as f32
}

pub fn beating_sample(index: u64, sample_rate: u32) -> f32 {
    let t = index as f64 / sample_rate as f64;
    (0.35 * ((TAU * 440.0 * t).sin() + (TAU * 442.0 * t).sin())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_sweep_matches_the_buffer_equation_and_is_monotonic() {
        let history = WaveformHistory::new(48_000);
        for logarithmic in [false, true] {
            let sweep = history.sweep(48_123.375, 60.0, 6000.0, false, logarithmic);
            assert_eq!(sweep.frequency_at(0.0), 60.0);
            assert_eq!(sweep.frequency_at(1.0), 6000.0);
            assert_eq!(sweep.sample_position(0.0), 48_123.375 - 800.0);
            assert_eq!(sweep.sample_position(1.0), 48_123.375);
            let mid = if logarithmic { 600.0 } else { 3030.0 };
            assert!((sweep.frequency_at(0.5) - mid).abs() < 1e-9);
            assert!(
                (sweep.sample_position(0.5) - (48_123.375 - 0.5 * 48_000.0 / mid)).abs() < 1e-9
            );
            for i in 1..=2000 {
                assert!(
                    sweep.sample_position(i as f64 / 2000.0)
                        >= sweep.sample_position((i - 1) as f64 / 2000.0)
                );
            }
        }
    }

    #[test]
    fn equal_sweep_limits_are_the_original_single_frequency_window() {
        let history = WaveformHistory::new(48_000);
        for locked in [false, true] {
            let original = history.window(48_123.375, 60.0, locked);
            let sweep = history.sweep(48_123.375, 60.0, 60.0, locked, true);
            for point in 0..=100 {
                let u = point as f64 / 100.0;
                assert_eq!(sweep.sample_position(u), original.sample_position(u));
            }
        }
    }

    #[test]
    fn sweep_uses_one_shared_phase_anchor_and_retains_the_raw_signal() {
        let mut history = WaveformHistory::new(48_000);
        history.ingest(
            &(0..96_000)
                .map(|i| tone_sample(i, 48_000, 60.0))
                .collect::<Vec<_>>(),
        );
        let a = history.sweep(48_000.0, 60.0, 6000.0, true, true);
        let b = history.sweep(48_400.0, 60.0, 6000.0, true, true);
        let c = history.sweep(48_800.0, 60.0, 6000.0, true, true);
        assert_eq!(a.end, b.end);
        for point in 0..=100 {
            let u = point as f64 / 100.0;
            let x = history.raw.interpolate(a.sample_position(u)).unwrap();
            let y = history.raw.interpolate(c.sample_position(u)).unwrap();
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn window_is_one_period_and_phase_hold_never_reads_ahead() {
        let history = WaveformHistory::new(48_000);
        for frequency in [30.0, 59.94, 60.0, 120.0, 440.0] {
            for end in [10_001.0, 20_123.375, 40_999.0] {
                let free = history.window(end, frequency, false);
                let locked = history.window(end, frequency, true);
                assert_eq!(free.end, end);
                assert!((free.span - 48_000.0 / frequency).abs() < 1e-10);
                assert!(locked.end <= end + 1e-10 && end - locked.end < locked.span + 1e-10);
                assert_eq!(free.sample_position(1.0), end);
                assert_eq!(free.sample_position(0.0), end - free.span);
            }
        }
    }

    #[test]
    fn sixty_hz_is_stationary_with_phase_lock_at_120_fps() {
        let mut history = WaveformHistory::new(48_000);
        history.ingest(
            &(0..96_000)
                .map(|i| tone_sample(i, 48_000, 60.0))
                .collect::<Vec<_>>(),
        );
        let reference = history.window(48_000.0, 60.0, true);
        for frame in 1..120 {
            let end = 48_000.0 + frame as f64 * 400.0;
            let locked = history.window(end, 60.0, true);
            for point in 0..=100 {
                let x = point as f64 / 100.0;
                assert!(
                    (history.value_at(reference, x).unwrap()
                        - history.value_at(locked, x).unwrap())
                    .abs()
                        < 1e-6
                );
            }
        }
        let a = history.window(48_000.0, 60.0, false);
        let b = history.window(48_400.0, 60.0, false);
        assert!(
            (history.value_at(a, 0.25).unwrap() + history.value_at(b, 0.25).unwrap()).abs() < 1e-6
        );
        assert!(history.value_at(a, 0.25).unwrap().abs() > 0.69);
    }

    #[test]
    fn fractional_periods_stay_locked_independently_of_render_rate() {
        let mut history = WaveformHistory::new(48_000);
        history.ingest(
            &(0..96_000)
                .map(|i| tone_sample(i, 48_000, 59.94))
                .collect::<Vec<_>>(),
        );
        let reference = history.window(48_000.0, 59.94, true);
        for fps in [30, 60, 120, 144] {
            for frame in 1..fps {
                let end = 48_000.0 + (frame as f64 / fps as f64 * 47_000.0).floor();
                let window = history.window(end, 59.94, true);
                for point in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    assert!(
                        (history.value_at(reference, point).unwrap()
                            - history.value_at(window, point).unwrap())
                        .abs()
                            < 2e-5
                    );
                }
            }
        }
    }

    #[test]
    fn different_timebases_read_the_same_unfiltered_signal() {
        let mut history = WaveformHistory::new(48_000);
        let samples: Vec<_> = (0..48_000).map(|i| beating_sample(i, 48_000)).collect();
        for chunk in samples.chunks(137) {
            history.ingest(chunk);
        }
        for f in [30.0, 60.0, 120.0, 440.0] {
            let window = history.window(47_999.0, f, true);
            for x in [0.0, 0.3, 0.6, 1.0] {
                assert_eq!(
                    history.value_at(window, x),
                    history
                        .raw
                        .interpolate(window.end - (1.0 - x) * 48_000.0 / f)
                );
            }
        }
    }

    #[test]
    fn extrema_preserve_peaks_but_exclude_samples_after_window_end() {
        let mut trace = Trace::new(4, 0);
        for value in [0.0, 0.9, -0.8, 0.2, 1.0] {
            trace.push(value);
        }
        assert!(trace.get(0).is_none());
        assert_eq!(trace.segment_range(-10.0, 10.0), Some((-0.8, 1.0)));
        let (low, high) = trace.segment_range(2.5, 3.5).unwrap();
        assert!((low + 0.3).abs() < 1e-6 && (high - 0.6).abs() < 1e-6);
        assert_eq!(trace.segment_range(3.0, 3.0), Some((0.2, 0.2)));
        assert_eq!(trace.segment_range(-2.0, 0.0), None);
    }

    #[test]
    fn wrapped_history_keeps_the_absolute_phase_origin() {
        let mut history = WaveformHistory::new(1000);
        history.ingest(
            &(0..7000)
                .map(|i| tone_sample(i, 1000, 10.0))
                .collect::<Vec<_>>(),
        );
        assert!(history.raw.get(2999).is_none());
        let a = history.window(5000.0, 10.0, true);
        let b = history.window(6999.0, 10.0, true);
        for x in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert!(
                (history.value_at(a, x).unwrap() - history.value_at(b, x).unwrap()).abs() < 1e-6
            );
        }
        let empty = WaveformHistory::new(48_000);
        assert!(
            empty
                .value_at(empty.window(-1.0, 60.0, true), 0.5)
                .is_none()
        );
    }
}
