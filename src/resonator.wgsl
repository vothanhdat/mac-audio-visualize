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
    processed_samples: f32,
    last_sample: f32,
    energy_cursor: u32,
    energy_filled: u32,
    paired_energy: f32,
}

// Keep energy history after the shared 64-byte-per-bin auxiliary reservation.
// Display highlights occupy the unused tail of this reservation.
struct ResonatorAuxiliary {
    states: array<ResonatorState, 2048>, // 40-byte stride
    reserved: array<vec4<f32>, 3072>,
    history: array<f32>,
}

@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<storage, read_write> spectrum: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> quarter_rate_samples: array<f32>;
@group(0) @binding(4) var<storage, read_write> eighth_rate_samples: array<f32>;
@group(0) @binding(5) var<storage, read_write> half_rate_samples: array<f32>;
@group(0) @binding(6) var<storage, read_write> sixteenth_rate_samples: array<f32>;
@group(0) @binding(8) var<storage, read_write> auxiliary: ResonatorAuxiliary;

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

// exp(-x) rounds close to one for low frequencies / high Q. Preserve the
// small input weight instead of subtracting two nearly equal f32 values.
fn one_minus_exp_negative(x: f32) -> f32 {
    if (x < 0.1) {
        return x * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0))));
    }
    return 1.0 - exp(-x);
}

fn measured_phasor(
    state: vec2<f32>,
    oscillator: vec2<f32>,
    rotation: vec2<f32>,
    phase: f32,
    decay_exponent: f32,
    input_weight: f32,
    processed_samples: f32,
) -> vec2<f32> {
    if (params.resonator.z < 0.5) {
        return state;
    }

    // A real tone mixed down to DC also leaves an image at twice the tuned
    // frequency. This is the exact one-pole response at that image frequency.
    // The half-angle form avoids cancellation near DC and needs no epsilon
    // that would incorrectly attenuate the image at high sample rates.
    let damping = 1.0 - input_weight;
    let sine = sin(phase);
    let denominator_real = input_weight + 2.0 * damping * sine * sine;
    let denominator_imag = -damping * sin(2.0 * phase);
    let response = input_weight * vec2(denominator_real, -denominator_imag)
        / (input_weight * input_weight + 4.0 * damping * sine * sine);
    var normalized_response = response;

    // Since reset the filter has a finite exponential window, not an infinite
    // one. Normalize only the image ratio; leave the wanted signal's natural
    // attack gain intact. Once settled, use the cheaper infinite response.
    let age = decay_exponent * processed_samples;
    if (processed_samples > 0.0 && age < 16.0) {
        let remaining = exp(-age);
        let gain = one_minus_exp_negative(age);
        let angle = phase * processed_samples;
        let sine_age = sin(angle);
        let finite_window = vec2(
            gain + 2.0 * remaining * sine_age * sine_age,
            -remaining * sin(2.0 * angle),
        );
        let finite_response = complex_multiply(response, finite_window) / gain;
        // With very few samples phase and amplitude cannot be distinguished
        // reliably. Fade to the finite solution only when it is conditioned.
        let determinant = 1.0 - dot(finite_response, finite_response);
        normalized_response = mix(
            response, finite_response, smoothstep(0.05, 0.2, determinant),
        );
    }
    let previous_oscillator =
        complex_multiply(oscillator, complex_conjugate(rotation));
    let image_phase = complex_multiply(previous_oscillator, previous_oscillator);
    let image = complex_multiply(normalized_response, image_phase);
    let denominator = max(1.0 - dot(image, image), 0.05);
    return (state - complex_multiply(image, complex_conjugate(state))) / denominator;
}

// Coefficients for A² = calibrated (position² + quadrature_velocity²).
// Use the recurrence instead of subtracting adjacent positions near DC.
fn position_velocity_coefficients(rotation: vec2<f32>, input_weight: f32) -> vec3<f32> {
    let damping = 1.0 - input_weight;
    let sine = -rotation.y;
    let cosine = rotation.x;
    let denominator = vec2(input_weight + 2.0 * damping * sine * sine,
        2.0 * damping * sine * cosine);
    let image_response = input_weight * complex_conjugate(denominator)
        / max(dot(denominator, denominator), 0.00000000000000000001);
    let calibration = vec2(1.0, 0.0) + image_response;
    return vec3(input_weight * cosine / max(damping * sine, 0.00000001),
        1.0 / damping, 4.0 / dot(calibration, calibration));
}

fn position_velocity_energy(
    state: vec2<f32>, oscillator: vec2<f32>, rotation: vec2<f32>,
    coefficients: vec3<f32>, sample: f32,
) -> f32 {
    let previous_oscillator = complex_multiply(oscillator, complex_conjugate(rotation));
    let physical = complex_multiply(state, complex_conjugate(previous_oscillator));
    let position = physical.x;
    let velocity_quadrature = coefficients.x * (sample - position) - coefficients.y * physical.y;
    return coefficients.z * dot(vec2(position, velocity_quadrature), vec2(position, velocity_quadrature));
}

fn position_velocity_magnitude(
    state: vec2<f32>, oscillator: vec2<f32>, rotation: vec2<f32>,
    input_weight: f32, sample: f32,
) -> f32 {
    return sqrt(position_velocity_energy(state, oscillator, rotation,
        position_velocity_coefficients(rotation, input_weight), sample));
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

    var state = auxiliary.states[bin];
    if (reset) {
        state.oscillator = vec2(1.0, 0.0);
        state.value = vec2(0.0);
        state.processed_samples = 0.0;
        state.last_sample = 0.0;
        state.energy_cursor = 0u;
        state.energy_filled = 0u;
        state.paired_energy = 0.0;
    }

    var rate_divisor = 1u;
    var new_count = params.resonator_stream.x;
    if (params.capture.w != 0u) {
        if (
            // Select the stage from configuration, even when this frame has
            // no /16 samples. Changing the rate would reinterpret stored state.
            params.resonator_stream_extra.y != 0u
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
    let decay_exponent = PI * frequency / (max(q, 0.01) * effective_rate);
    let input_weight = one_minus_exp_negative(decay_exponent);
    let full_rate_start = MAX_SAMPLE_COUNT - min(new_count, MAX_SAMPLE_COUNT);

    let pair_measurement = params.resonator.z > 2.5;
    let position_only = params.resonator.z > 3.5;
    let energy_stride = params.resonator_stream_extra.z;
    let energy_offset = bin * energy_stride;
    let quarter_delay = effective_rate / (4.0 * frequency);
    let delay_low = u32(floor(quarter_delay));
    let delay_fraction = fract(quarter_delay);
    var energy_coefficients = vec3(0.0);
    if (pair_measurement) { energy_coefficients = position_velocity_coefficients(rotation, input_weight); }

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
        state.last_sample = sample;
        let mixed = state.oscillator * sample;
        state.value += input_weight * (mixed - state.value);
        state.oscillator = complex_multiply(state.oscillator, rotation);
        if (pair_measurement) {
            var current = 0.0;
            if (position_only) {
                let previous_oscillator = complex_multiply(state.oscillator, complex_conjugate(rotation));
                current = complex_multiply(state.value, complex_conjugate(previous_oscillator)).x;
            } else {
                current = position_velocity_energy(state.value, state.oscillator,
                    rotation, energy_coefficients, sample);
            }
            auxiliary.history[energy_offset + state.energy_cursor] = current;
            state.energy_filled = min(state.energy_filled + 1u, energy_stride);
            var delayed = current;
            if (state.energy_filled > delay_low + 1u) {
                let low = (state.energy_cursor + energy_stride - delay_low) % energy_stride;
                let high = (low + energy_stride - 1u) % energy_stride;
                delayed = mix(auxiliary.history[energy_offset + low],
                    auxiliary.history[energy_offset + high], delay_fraction);
            }
            if (position_only) {
                // Interpolate displacement first, then square the two positions.
                // For a settled on-center sinusoid they are 90 degrees apart.
                state.paired_energy = energy_coefficients.z * (current * current + delayed * delayed);
            } else {
                state.paired_energy = 0.5 * (current + delayed);
            }
            state.energy_cursor = (state.energy_cursor + 1u) % energy_stride;
        }
    }
    state.oscillator /= max(length(state.oscillator), 0.000001);

    if (decay_exponent * state.processed_samples < 16.0) {
        state.processed_samples += f32(new_count);
    }
    let previous = spectrum[bin];
    var amplitude = 0.0;
    if (pair_measurement) {
        amplitude = sqrt(state.paired_energy);
    } else if (params.resonator.z > 1.5) {
        amplitude = position_velocity_magnitude(
            state.value, state.oscillator, rotation, input_weight, state.last_sample,
        );
    } else {
        let phasor = measured_phasor(
            state.value, state.oscillator, rotation, phase, decay_exponent,
            input_weight, state.processed_samples,
        );
        amplitude = 2.0 * length(phasor);
    }
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
    auxiliary.states[bin] = state;
}
