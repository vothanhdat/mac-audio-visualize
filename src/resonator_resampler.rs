// A small streaming anti-aliasing cascade for the resonator bank. Each stage
// halves the rate, giving the GPU /2, /4, /8, and optionally /16 versions of
// the same new audio block without repeating filtering work for every
// frequency bar.

const HALF_RATE_TAPS: [f32; 15] = [
    -0.001_652_631_7,
    0.005_392_317,
    0.011_405_306,
    -0.020_483_937,
    -0.060_768_04,
    0.040_690_497,
    0.300_267_64,
    0.450_297_7,
    0.300_267_64,
    0.040_690_497,
    -0.060_768_04,
    -0.020_483_937,
    0.011_405_306,
    0.005_392_317,
    -0.001_652_631_7,
];

#[derive(Default)]
struct HalfRateStage {
    history: [f32; HALF_RATE_TAPS.len()],
    write: usize,
    phase: bool,
}

impl HalfRateStage {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn push(&mut self, sample: f32) -> Option<f32> {
        self.history[self.write] = sample;
        self.write = (self.write + 1) % self.history.len();
        self.phase = !self.phase;
        if self.phase {
            return None;
        }

        let mut filtered = 0.0;
        for (tap, coefficient) in HALF_RATE_TAPS.iter().enumerate() {
            let index = (self.write + self.history.len() - 1 - tap) % self.history.len();
            filtered += self.history[index] * coefficient;
        }
        Some(filtered)
    }
}

#[derive(Default)]
struct SampleDelay {
    samples: Vec<f32>,
    write: usize,
}

impl SampleDelay {
    fn reset(&mut self, delay_samples: usize) {
        self.samples.clear();
        self.samples.resize(delay_samples, 0.0);
        self.write = 0;
    }

    fn push(&mut self, sample: f32) -> f32 {
        if self.samples.is_empty() {
            return sample;
        }
        let delayed = self.samples[self.write];
        self.samples[self.write] = sample;
        self.write = (self.write + 1) % self.samples.len();
        delayed
    }
}

#[derive(Default)]
pub struct ResonatorResampler {
    half: HalfRateStage,
    quarter: HalfRateStage,
    eighth: HalfRateStage,
    sixteenth: HalfRateStage,
    half_delay: SampleDelay,
    quarter_delay: SampleDelay,
    eighth_delay: SampleDelay,
    half_samples: Vec<f32>,
    quarter_samples: Vec<f32>,
    eighth_samples: Vec<f32>,
    sixteenth_samples: Vec<f32>,
}

impl ResonatorResampler {
    pub fn process(
        &mut self,
        samples: &[f32],
        reset: bool,
        include_sixteenth: bool,
        align_timestamps: bool,
    ) {
        if reset {
            self.half.reset();
            self.quarter.reset();
            self.eighth.reset();
            self.sixteenth.reset();
            let (half_delay, quarter_delay, eighth_delay) =
                match (align_timestamps, include_sixteenth) {
                    (true, true) => (49, 21, 7),
                    (true, false) => (21, 7, 0),
                    (false, _) => (0, 0, 0),
                };
            self.half_delay.reset(half_delay);
            self.quarter_delay.reset(quarter_delay);
            self.eighth_delay.reset(eighth_delay);
        }
        self.half_samples.clear();
        self.quarter_samples.clear();
        self.eighth_samples.clear();
        self.sixteenth_samples.clear();

        for &sample in samples {
            let Some(half) = self.half.push(sample) else {
                continue;
            };
            self.half_samples.push(self.half_delay.push(half));

            let Some(quarter) = self.quarter.push(half) else {
                continue;
            };
            self.quarter_samples.push(self.quarter_delay.push(quarter));

            let Some(eighth) = self.eighth.push(quarter) else {
                continue;
            };
            self.eighth_samples.push(self.eighth_delay.push(eighth));

            if include_sixteenth && let Some(sixteenth) = self.sixteenth.push(eighth) {
                self.sixteenth_samples.push(sixteenth);
            }
        }
    }

    pub fn streams(&self) -> [&[f32]; 4] {
        [
            &self.half_samples,
            &self.quarter_samples,
            &self.eighth_samples,
            &self.sixteenth_samples,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::ResonatorResampler;

    #[test]
    fn cascade_produces_expected_sample_counts() {
        let mut resampler = ResonatorResampler::default();
        resampler.process(&vec![1.0; 16_384], true, true, true);
        let streams = resampler.streams();
        assert_eq!(streams[0].len(), 8192);
        assert_eq!(streams[1].len(), 4096);
        assert_eq!(streams[2].len(), 2048);
        assert_eq!(streams[3].len(), 1024);
        assert!(
            streams
                .iter()
                .flat_map(|stream| stream.iter())
                .all(|sample| sample.is_finite())
        );
    }

    #[test]
    fn five_stage_alignment_matches_slowest_stream_timestamp() {
        let input: Vec<f32> = (0..16_384)
            .map(|index| ((index as f32) * 0.017).sin())
            .collect();
        let mut natural = ResonatorResampler::default();
        natural.process(&input, true, true, false);
        let natural_streams = natural.streams();
        let mut aligned = ResonatorResampler::default();
        aligned.process(&input, true, true, true);
        let aligned_streams = aligned.streams();

        for (stream, delay) in [49, 21, 7, 0].into_iter().enumerate() {
            for index in delay..aligned_streams[stream].len() {
                assert!(
                    (aligned_streams[stream][index] - natural_streams[stream][index - delay]).abs()
                        < 1.0e-6
                );
            }
        }
    }
}
