struct Params {
    sample_rate: f32,
    elapsed: f32,
    gain: f32,
    bin_count: u32,
    resolution: vec2<f32>,
    axis_focus: vec2<f32>,
    perception: vec4<f32>,
    visual: vec4<f32>,
    capture: vec4<u32>,
    db_scale: vec4<f32>,
    frequency_range: vec4<f32>,
    auditory_0: vec4<f32>,
    auditory_1: vec4<f32>,
    fft_sizes_0: vec4<u32>,
    fft_sizes_1: vec4<u32>,
    fft_decimations_0: vec4<u32>,
    fft_decimations_1: vec4<u32>,
    fft_crossovers: vec4<f32>,
    resonator: vec4<f32>,
    resonator_decay: vec4<f32>,
    resonator_stream: vec4<u32>,
    hybrid: vec4<f32>,
    resonator_stream_extra: vec4<u32>,
}

struct ResonatorState {
    oscillator: vec2<f32>,
    value: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<storage, read_write> spectrum: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> quarter_rate_samples: array<f32>;
@group(0) @binding(4) var<storage, read_write> eighth_rate_samples: array<f32>;
@group(0) @binding(5) var<storage, read_write> half_rate_samples: array<f32>;
@group(0) @binding(6) var<storage, read_write> sixteenth_rate_samples: array<f32>;
@group(0) @binding(8) var<storage, read_write> states: array<ResonatorState>;

const PI: f32 = 3.141592653589793;
const MAX_SAMPLE_COUNT: u32 = 16384u;

fn local_db_floor(t: f32) -> f32 {
    return mix(params.db_scale.x, params.db_scale.y, smoothstep(0.0, 1.0, t));
}

fn focused_log_frequency_t(display_t: f32) -> f32 {
    if (display_t <= 0.0) { return 0.0; }
    if (display_t >= 1.0) { return 1.0; }
    let origin = clamp(params.axis_focus.y, 0.001, 0.999);
    let exponent = pow(4.0, params.axis_focus.x * 2.0 - 1.0);
    let origin_logit = log(origin / (1.0 - origin));
    let display_logit = log(display_t / (1.0 - display_t));
    let source_logit = origin_logit + (display_logit - origin_logit) / exponent;
    return 1.0 / (1.0 + exp(-source_logit));
}

fn complex_multiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_conjugate(value: vec2<f32>) -> vec2<f32> {
    return vec2(value.x, -value.y);
}

fn complex_divide(numerator: vec2<f32>, denominator: vec2<f32>) -> vec2<f32> {
    let scale = max(dot(denominator, denominator), 0.000001);
    return complex_multiply(numerator, complex_conjugate(denominator)) / scale;
}

fn measured_phasor(
    state: vec2<f32>,
    oscillator: vec2<f32>,
    rotation: vec2<f32>,
    damping: f32,
) -> vec2<f32> {
    if (params.resonator.z < 0.5) {
        return state;
    }

    // A real sinusoid mixed down by the complex oscillator contains the wanted
    // stationary phasor plus a counter-rotating image at twice the tuned
    // frequency. Reconstruct that image through the resonator's exact one-pole
    // response, then solve z = wanted + image * conjugate(wanted). This removes
    // phase-dependent level ripple without another time window.
    let positive_double_rotation = complex_multiply(
        complex_conjugate(rotation),
        complex_conjugate(rotation),
    );
    var image_response = complex_divide(
        vec2(1.0 - damping, 0.0),
        vec2(1.0, 0.0) - damping * positive_double_rotation,
    );
    let previous_oscillator =
        complex_multiply(oscillator, complex_conjugate(rotation));
    let image_phase = complex_multiply(previous_oscillator, previous_oscillator);
    let image = complex_multiply(image_response, image_phase);
    let denominator = max(1.0 - dot(image, image), 0.05);
    let wanted =
        (state - complex_multiply(image, complex_conjugate(state))) / denominator;
    return wanted;
}

fn level_from_amplitude(amplitude: f32, floor_db: f32) -> f32 {
    let magnitude = amplitude;
    let db = 20.0 * log2(max(magnitude * params.gain, 0.000001)) / log2(10.0);
    return clamp((db - floor_db) / max(params.db_scale.z - floor_db, 0.001), 0.0, 1.0);
}

@compute @workgroup_size(64)
fn resonate(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    if (bin >= params.bin_count) { return; }

    let display_t = f32(bin) / f32(params.bin_count - 1u);
    let frequency_t = focused_log_frequency_t(display_t);
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    let frequency = lower * pow(upper / lower, frequency_t);
    let floor_db = local_db_floor(frequency_t);
    let reset = params.capture.z != 0u;
    // Hybrid mode follows the multirate stages directly: /8 and /4 remain
    // resonators, while /2 and /1 are supplied by the FFT mapper.
    if (params.hybrid.z > 0.5 && frequency > params.sample_rate * 0.091) { return; }

    var state = states[bin];
    if (reset) {
        state.oscillator = vec2(1.0, 0.0);
        state.value = vec2(0.0);
    }

    var rate_divisor = 1u;
    var new_count = params.resonator_stream.x;
    if (params.capture.w != 0u) {
        if (
            params.resonator_stream_extra.x > 0u
            && frequency <= params.sample_rate * 0.0205
        ) {
            rate_divisor = 16u;
            new_count = params.resonator_stream_extra.x;
        } else if (frequency <= params.sample_rate * 0.041) {
            rate_divisor = 8u;
            new_count = params.resonator_stream.w;
        } else if (frequency <= params.sample_rate * 0.091) {
            rate_divisor = 4u;
            new_count = params.resonator_stream.z;
        } else if (frequency <= params.sample_rate * 0.181) {
            rate_divisor = 2u;
            new_count = params.resonator_stream.y;
        }
    }
    let effective_rate = params.sample_rate / f32(rate_divisor);
    let phase = 2.0 * PI * frequency / effective_rate;
    let rotation = vec2(cos(phase), -sin(phase));
    let q_t = clamp(
        (sqrt(frequency) - sqrt(50.0)) / (sqrt(10000.0) - sqrt(50.0)),
        0.0,
        1.0,
    );
    let q = mix(params.resonator.x, params.resonator.y, q_t);
    let damping = exp(-PI * frequency / (max(q, 0.01) * effective_rate));
    let full_rate_start = MAX_SAMPLE_COUNT - min(new_count, MAX_SAMPLE_COUNT);

    for (var index = 0u; index < new_count; index++) {
        var sample = samples[full_rate_start + index];
        if (rate_divisor == 2u) {
            sample = half_rate_samples[index];
        } else if (rate_divisor == 4u) {
            sample = quarter_rate_samples[index];
        } else if (rate_divisor == 8u) {
            sample = eighth_rate_samples[index];
        } else if (rate_divisor == 16u) {
            sample = sixteenth_rate_samples[index];
        }
        let mixed = state.oscillator * sample;
        state.value = damping * state.value + (1.0 - damping) * mixed;
        state.oscillator = complex_multiply(state.oscillator, rotation);
    }
    state.oscillator /= max(length(state.oscillator), 0.000001);

    let previous = spectrum[bin];
    let phasor = measured_phasor(state.value, state.oscillator, rotation, damping);
    let amplitude = 2.0 * length(phasor);
    var measured = level_from_amplitude(amplitude, floor_db);
    var envelope_state = measured;
    if (!reset) {
        let elapsed_seconds = f32(params.resonator_stream.x) / params.sample_rate;
        let decay_db_per_second =
            params.resonator_decay.x / max(params.resonator_decay.y, 0.001);
        let db_span = max(params.db_scale.z - floor_db, 0.001);
        let decay = decay_db_per_second * elapsed_seconds / db_span;
        if (params.resonator_decay.z < 0.5) {
            measured = max(measured, previous.x - decay);
            envelope_state = measured;
        } else {
            // Show a newly captured peak at full level. The decayed value is
            // retained separately as the starting envelope for the next update.
            measured = max(measured, previous.z);
            envelope_state = max(measured - decay, 0.0);
        }
    }

    let hybrid_fft_history = select(0.0, previous.w, params.hybrid.z > 0.5 && !reset);
    spectrum[bin] = vec4(measured, previous.y, envelope_state, hybrid_fft_history);
    states[bin] = state;
}
