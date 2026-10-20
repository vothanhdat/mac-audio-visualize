struct Params {
    sample_rate: f32,
    elapsed: f32,
    gain: f32,
    bin_count: u32,
    resolution: vec2<f32>,
    axis_focus: vec2<f32>,
    attack_0: vec4<f32>,
    attack_1: vec4<f32>,
    decay_0: vec4<f32>,
    decay_1: vec4<f32>,
    perception: vec4<f32>,
    visual: vec4<f32>,
    sample_counts_0: vec4<u32>,
    sample_counts_1: vec4<u32>,
    db_scale: vec4<f32>,
    frequency_range: vec4<f32>,
    frequency_smoothing: vec4<f32>,
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
}

@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<storage, read_write> spectrum: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> mid_samples: array<f32>;
@group(0) @binding(4) var<storage, read_write> bass_samples: array<f32>;
@group(0) @binding(5) var<storage, read_write> upper_mid_samples: array<f32>;
@group(0) @binding(6) var<storage, read_write> sub_bass_samples: array<f32>;
@group(0) @binding(7) var<storage, read_write> display_spectrum: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> auxiliary: array<vec4<f32>>;

const PI: f32 = 3.141592653589793;
const MAX_SAMPLE_COUNT: u32 = 16384u;
const MAX_BIN_COUNT: u32 = 2048u;
const SMOOTHING_OFFSET: u32 = MAX_BIN_COUNT * 3u;
fn local_db_floor(t: f32) -> f32 {
    return mix(params.db_scale.x, params.db_scale.y, smoothstep(0.0, 1.0, t));
}

// Inverse of the display-axis lens. Strength 0.5 makes the exponent exactly
// one and therefore preserves the ordinary logarithmic frequency axis.
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

fn frequency_at_bin(bin: u32) -> vec2<f32> {
    if (params.perception.x > 0.5) {
        let lower = params.frequency_range.x;
        let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
        let lower_midi = 69.0 + 12.0 * log2(lower / 440.0);
        let first_midi = ceil(lower_midi - 0.0001);
        let midi_note = first_midi + f32(bin);
        let frequency = 440.0 * pow(2.0, (midi_note - 69.0) / 12.0);
        let frequency_t = clamp(log2(frequency / lower) / log2(upper / lower), 0.0, 1.0);
        return vec2(frequency, frequency_t);
    }
    let display_t = f32(bin) / f32(max(params.bin_count - 1u, 1u));
    let frequency_t = focused_log_frequency_t(display_t);
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    return vec2(lower * pow(upper / lower, frequency_t), frequency_t);
}

fn measured_db_at_bin(bin: u32) -> f32 {
    let point = frequency_at_bin(bin);
    let floor_db = local_db_floor(point.y);
    return mix(floor_db, params.db_scale.z, spectrum[bin].x);
}

fn smoothing_width_at(frequency: f32) -> f32 {
    let frequency_mix = smoothstep(
        0.0,
        1.0,
        clamp(
            log2(frequency / 50.0) / log2(10000.0 / 50.0),
            0.0,
            1.0,
        ),
    );
    return mix(
        params.frequency_smoothing.y,
        params.frequency_smoothing.z,
        frequency_mix,
    );
}

fn sinc_pi(x: f32) -> f32 {
    if (abs(x) < 0.000001) { return 1.0; }
    return sin(PI * x) / (PI * x);
}

fn bessel_i0(x: f32) -> f32 {
    let y = x * x * 0.25;
    var sum = 1.0;
    var term = 1.0;
    for (var k = 1u; k <= 12u; k++) {
        term *= y / (f32(k) * f32(k));
        sum += term;
    }
    return sum;
}

// Causal, normalized Hamming-windowed sinc. Filtering happens once per output
// sample, not once per frequency bin, so the longer bass filter is inexpensive.
fn filtered_sample(sample_index: u32, cutoff_hz: f32, taps: u32) -> f32 {
    let delay = f32(taps - 1u) * 0.5;
    let normalized_cutoff = cutoff_hz / params.sample_rate;
    var value = 0.0;
    var weight_sum = 0.0;
    for (var tap = 0u; tap < taps; tap++) {
        let k = f32(tap) - delay;
        let window = 0.54 - 0.46 * cos(2.0 * PI * f32(tap) / f32(taps - 1u));
        let weight = 2.0 * normalized_cutoff * sinc_pi(2.0 * normalized_cutoff * k) * window;
        let index = u32(max(i32(sample_index) - i32(tap), 0));
        value += samples[index] * weight;
        weight_sum += weight;
    }
    return value / max(weight_sum, 0.000001);
}

fn selected_sample_count(band: u32) -> u32 {
    if (band == 4u) { return params.sample_counts_0.x; }
    if (band == 3u) { return params.sample_counts_0.y; }
    if (band == 2u) { return params.sample_counts_0.z; }
    if (band == 1u) { return params.sample_counts_0.w; }
    return params.sample_counts_1.x;
}

fn prepared_sample_count(band: u32) -> u32 {
    return selected_sample_count(band) * select(1u, 2u, params.attack_1.w > 0.5);
}

@compute @workgroup_size(64)
fn decimate_mid(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = prepared_sample_count(2u);
    if (id.x >= count) { return; }
    let start = MAX_SAMPLE_COUNT - count * 8u;
    mid_samples[id.x] = filtered_sample(start + id.x * 8u, 1800.0, 63u);
}

@compute @workgroup_size(64)
fn decimate_bass(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = prepared_sample_count(3u);
    if (id.x >= count) { return; }
    let start = MAX_SAMPLE_COUNT - count * 32u;
    bass_samples[id.x] = filtered_sample(start + id.x * 32u, 450.0, 255u);
}

@compute @workgroup_size(64)
fn decimate_upper_mid(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = prepared_sample_count(1u);
    if (id.x >= count) { return; }
    let start = MAX_SAMPLE_COUNT - count * 2u;
    upper_mid_samples[id.x] = filtered_sample(start + id.x * 2u, 7500.0, 31u);
}

@compute @workgroup_size(64)
fn decimate_sub_bass(@builtin(global_invocation_id) id: vec3<u32>) {
    let count = prepared_sample_count(4u);
    if (id.x >= count) { return; }
    let start = MAX_SAMPLE_COUNT - count * 64u;
    sub_bass_samples[id.x] = filtered_sample(start + id.x * 64u, 120.0, 511u);
}

fn source_sample(band: u32, index: u32, count: u32) -> f32 {
    let prepared = prepared_sample_count(band);
    let offset = prepared - count;
    if (band == 4u) { return sub_bass_samples[offset + index]; }
    if (band == 3u) { return bass_samples[offset + index]; }
    if (band == 2u) { return mid_samples[offset + index]; }
    if (band == 1u) { return upper_mid_samples[offset + index]; }
    return samples[MAX_SAMPLE_COUNT - count + index];
}

fn measure_magnitude(frequency: f32, band: u32, count: u32, effective_rate: f32) -> f32 {
    let phase_step = 2.0 * PI * frequency / effective_rate;
    let rotation = vec2(cos(phase_step), sin(phase_step));
    let window_step = 2.0 * PI / f32(count - 1u);
    let window_rotation = vec2(cos(window_step), sin(window_step));
    var oscillator = vec2(1.0, 0.0);
    var window_oscillator = vec2(1.0, 0.0);
    var real = 0.0;
    var imag = 0.0;
    var window_sum = 0.0;
    for (var i = 0u; i < count; i++) {
        let position = f32(i) / f32(count - 1u);
        var window = 0.5 - 0.5 * window_oscillator.x;
        if (params.attack_1.z > 0.5 && params.attack_1.z < 1.5) {
            // Rising half-Hann: zero at the oldest sample, one at newest.
            window = 0.5 - 0.5 * cos(PI * position);
        } else if (params.attack_1.z >= 1.5 && params.attack_1.z < 2.5) {
            // Rising first half of a Blackman window, oriented toward "now".
            window = 0.42 - 0.5 * cos(PI * position) + 0.08 * cos(2.0 * PI * position);
        } else if (params.attack_1.z >= 2.5 && params.attack_1.z < 3.5) {
            // Rising half of a Kaiser window. Old samples receive
            // the tapered edge; the newest sample receives the center peak.
            let beta = params.decay_1.z;
            let x = position - 1.0;
            window = bessel_i0(beta * sqrt(max(1.0 - x * x, 0.0))) / bessel_i0(beta);
        } else if (params.attack_1.z >= 3.5) {
            // Flanagan auditory-model impulse: t² exp(-alpha*t), with t as
            // normalized age from now. Its peak is at age 2 / alpha.
            let alpha = params.decay_1.w;
            let age = 1.0 - position;
            let peak_age = 2.0 / alpha;
            let peak = peak_age * peak_age * exp(-alpha * peak_age);
            window = age * age * exp(-alpha * age) / peak;
        }
        let value = source_sample(band, i, count) * window;
        window_sum += window;
        real += value * oscillator.x;
        imag -= value * oscillator.y;
        oscillator = vec2(
            oscillator.x * rotation.x - oscillator.y * rotation.y,
            oscillator.x * rotation.y + oscillator.y * rotation.x,
        );
        window_oscillator = vec2(
            window_oscillator.x * window_rotation.x - window_oscillator.y * window_rotation.y,
            window_oscillator.x * window_rotation.y + window_oscillator.y * window_rotation.x,
        );
    }
    // Normalize by coherent gain so window changes do not alter dB amplitude.
    return sqrt(real * real + imag * imag) * (2.0 / max(window_sum, 0.000001));
}

fn analysis_window_count(band: u32, count_mode: u32) -> u32 {
    let base = selected_sample_count(band);
    if (count_mode == 0u) { return max(base / 2u, 32u); }
    if (count_mode == 2u) { return base * 2u; }
    return base;
}

fn band_magnitude(frequency: f32, band: u32, count_mode: u32) -> f32 {
    let count = analysis_window_count(band, count_mode);
    if (band == 4u) { return measure_magnitude(frequency, 4u, count, params.sample_rate / 64.0); }
    if (band == 3u) { return measure_magnitude(frequency, 3u, count, params.sample_rate / 32.0); }
    if (band == 2u) { return measure_magnitude(frequency, 2u, count, params.sample_rate / 8.0); }
    if (band == 1u) { return measure_magnitude(frequency, 1u, count, params.sample_rate / 2.0); }
    return measure_magnitude(frequency, 0u, count, params.sample_rate);
}

fn crossfade_magnitude(frequency: f32, low_band: u32, high_band: u32, low_hz: f32, high_hz: f32, count_mode: u32) -> f32 {
    let blend = smoothstep(low_hz, high_hz, frequency);
    // Blend in log amplitude so crossover energy moves perceptually smoothly.
    let low_db = log2(max(band_magnitude(frequency, low_band, count_mode), 0.000001));
    let high_db = log2(max(band_magnitude(frequency, high_band, count_mode), 0.000001));
    return exp2(mix(low_db, high_db, blend));
}

fn detailed_at(frequency: f32, count_mode: u32) -> f32 {
    if (frequency < 65.0) {
        return band_magnitude(frequency, 4u, count_mode);
    } else if (frequency < 95.0) {
        return crossfade_magnitude(frequency, 4u, 3u, 65.0, 95.0, count_mode);
    } else if (frequency < 260.0) {
        return band_magnitude(frequency, 3u, count_mode);
    } else if (frequency < 380.0) {
        return crossfade_magnitude(frequency, 3u, 2u, 260.0, 380.0, count_mode);
    } else if (frequency < 1050.0) {
        return band_magnitude(frequency, 2u, count_mode);
    } else if (frequency < 1500.0) {
        return crossfade_magnitude(frequency, 2u, 1u, 1050.0, 1500.0, count_mode);
    } else if (frequency < 4200.0) {
        return band_magnitude(frequency, 1u, count_mode);
    } else if (frequency < 6000.0) {
        return crossfade_magnitude(frequency, 1u, 0u, 4200.0, 6000.0, count_mode);
    }
    return band_magnitude(frequency, 0u, count_mode);
}

@compute @workgroup_size(64)
fn analyze(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    if (bin >= params.bin_count) { return; }
    let t = f32(bin) / f32(params.bin_count - 1u);
    let frequency_t = focused_log_frequency_t(t);
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    let frequency = lower * pow(
        upper / lower,
        frequency_t,
    );

    var detailed_magnitude = detailed_at(frequency, select(1u, 2u, params.attack_1.w > 0.5));
    let previous = spectrum[bin];
    let previous_fast = select(previous.z, 0.0, params.perception.y > 0.5);
    let previous_fastest = select(previous.w, 0.0, params.perception.y > 0.5);
    var fast_history = previous_fast;
    var fastest_history = previous_fastest;
    if (params.attack_1.w > 0.5) {
        let fast_magnitude = detailed_at(frequency, 1u);
        let fast_db = 20.0 * log2(max(fast_magnitude * params.gain, 0.000001)) / log2(10.0);
        let fast_level = clamp(
            (fast_db - local_db_floor(frequency_t)) /
                max(params.db_scale.z - local_db_floor(frequency_t), 0.001),
            0.0,
            1.0,
        );
        let detail_db = 20.0 * log2(max(detailed_magnitude * params.gain, 0.000001)) / log2(10.0);
        let detail_level = clamp(
            (detail_db - local_db_floor(frequency_t)) /
                max(params.db_scale.z - local_db_floor(frequency_t), 0.001),
            0.0,
            1.0,
        );
        let fresh_attack = smoothstep(0.0, 0.12, max(fast_level - previous_fast, 0.0));
        let adaptive_mix = params.decay_1.y * fresh_attack;
        let adaptive_level = mix(detail_level, fast_level, adaptive_mix);
        let adaptive_db = mix(local_db_floor(frequency_t), params.db_scale.z, adaptive_level);
        detailed_magnitude = pow(10.0, adaptive_db / 20.0) / params.gain;
        fast_history = fast_level;
        if (params.attack_1.w > 1.5) {
            let fastest_magnitude = detailed_at(frequency, 0u);
            let fastest_db = 20.0 * log2(max(fastest_magnitude * params.gain, 0.000001)) / log2(10.0);
            let fastest_level = clamp(
                (fastest_db - local_db_floor(frequency_t)) /
                    max(params.db_scale.z - local_db_floor(frequency_t), 0.001),
                0.0,
                1.0,
            );
            let leading_edge = smoothstep(0.0, 0.16, max(fastest_level - previous_fastest, 0.0));
            let fastest_mix = params.decay_1.y * leading_edge;
            let current_db = 20.0 * log2(max(detailed_magnitude * params.gain, 0.000001)) / log2(10.0);
            let current_level = clamp(
                (current_db - local_db_floor(frequency_t)) /
                    max(params.db_scale.z - local_db_floor(frequency_t), 0.001),
                0.0,
                1.0,
            );
            let triple_level = mix(current_level, fastest_level, fastest_mix);
            let triple_db = mix(local_db_floor(frequency_t), params.db_scale.z, triple_level);
            detailed_magnitude = pow(10.0, triple_db / 20.0) / params.gain;
            fastest_history = fastest_level;
        }
    }
    let detailed_db = 20.0 * log2(max(detailed_magnitude * params.gain, 0.000001)) / log2(10.0);
    let db_floor = local_db_floor(frequency_t);
    let detailed = clamp(
        (detailed_db - db_floor) / max(params.db_scale.z - db_floor, 0.001),
        0.0,
        1.0,
    );
    var detail_attack = params.attack_1.x;
    var detail_decay = params.decay_1.x;
    if (frequency < 80.0) {
        detail_attack = params.attack_0.x;
        detail_decay = params.decay_0.x;
    } else if (frequency < 320.0) {
        detail_attack = params.attack_0.y;
        detail_decay = params.decay_0.y;
    } else if (frequency < 1280.0) {
        detail_attack = params.attack_0.z;
        detail_decay = params.decay_0.z;
    } else if (frequency < 5120.0) {
        detail_attack = params.attack_0.w;
        detail_decay = params.decay_0.w;
    }
    var smoothed_detail = detailed;
    if (params.attack_1.y > 0.5) {
        smoothed_detail = mix(detailed, previous.x, select(detail_decay, detail_attack, detailed > previous.x));
    }
    // Preserve y: compose_display uses it as the bar's fall velocity in dB/s.
    spectrum[bin] = vec4(smoothed_detail, previous.y, fast_history, fastest_history);
}

// One invocation intentionally performs two O(bin_count) directional scans.
// At the visualizer's maximum 2048 bins this is cheaper than having every bin
// rescan its neighbors, and it exactly implements the directional envelope.
@compute @workgroup_size(1)
fn smooth_frequency(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u || params.bin_count == 0u) { return; }

    let max_lift_db = params.frequency_smoothing.w;
    var previous_frequency = frequency_at_bin(0u).x;
    var forward_db = measured_db_at_bin(0u);
    auxiliary[SMOOTHING_OFFSET].x = forward_db;

    for (var bin = 1u; bin < params.bin_count; bin++) {
        let frequency = frequency_at_bin(bin).x;
        let distance_octaves = abs(log2(frequency / previous_frequency));
        let width_octaves = smoothing_width_at(frequency);
        let raw_db = measured_db_at_bin(bin);
        let traveled_db = max_lift_db * distance_octaves / max(width_octaves, 0.000001);
        forward_db = max(raw_db, forward_db - traveled_db);
        forward_db = min(forward_db, raw_db + max_lift_db);
        auxiliary[SMOOTHING_OFFSET + bin].x = forward_db;
        previous_frequency = frequency;
    }

    var bin = params.bin_count - 1u;
    previous_frequency = frequency_at_bin(bin).x;
    var backward_db = measured_db_at_bin(bin);
    auxiliary[SMOOTHING_OFFSET + bin].y = max(auxiliary[SMOOTHING_OFFSET + bin].x, backward_db);

    while (bin > 0u) {
        bin -= 1u;
        let frequency = frequency_at_bin(bin).x;
        let distance_octaves = abs(log2(previous_frequency / frequency));
        let width_octaves = smoothing_width_at(frequency);
        let raw_db = measured_db_at_bin(bin);
        let traveled_db = max_lift_db * distance_octaves / max(width_octaves, 0.000001);
        backward_db = max(raw_db, backward_db - traveled_db);
        backward_db = min(backward_db, raw_db + max_lift_db);
        auxiliary[SMOOTHING_OFFSET + bin].y = max(auxiliary[SMOOTHING_OFFSET + bin].x, backward_db);
        previous_frequency = frequency;
    }
}

@compute @workgroup_size(64)
fn compose_display(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.bin_count) { return; }
    let raw = spectrum[id.x];
    let point = frequency_at_bin(id.x);
    let frequency_t = point.y;
    let db_span = max(params.db_scale.z - local_db_floor(frequency_t), 0.001);
    let previous_display = display_spectrum[id.x];
    var measured = raw.x;
    let smoothing_strength = params.frequency_smoothing.x;
    if (smoothing_strength > 0.0001) {
        let raw_db = mix(local_db_floor(frequency_t), params.db_scale.z, raw.x);
        let envelope_db = auxiliary[SMOOTHING_OFFSET + id.x].y;
        let smoothed_db = mix(raw_db, envelope_db, smoothing_strength);
        measured = clamp(
            (smoothed_db - local_db_floor(frequency_t)) / db_span,
            0.0,
            1.0,
        );
    }
    var displayed = previous_display.x;
    var bar_velocity = raw.y;
    if (params.perception.y > 0.5) {
        displayed = measured;
        bar_velocity = 0.0;
    } else if (measured >= displayed) {
        displayed = measured;
        bar_velocity = 0.0;
    } else {
        // Bars have no hold. They begin falling immediately and accelerate
        // faster than the peak markers under their independent gravity.
        let dt = params.perception.w;
        let traveled_db = bar_velocity * dt + 0.5 * params.perception.z * dt * dt;
        bar_velocity += params.perception.z * dt;
        displayed = max(measured, displayed - traveled_db / db_span);
    }
    spectrum[id.x].y = bar_velocity;

    var fall_velocity = previous_display.y;
    var peak = previous_display.z;
    var hold = previous_display.w;
    let peak_momentum = params.visual.z < 0.0;
    let peak_gravity = abs(params.visual.z);
    if (params.perception.y > 0.5) {
        peak = displayed;
        fall_velocity = 0.0;
        hold = params.visual.w;
    } else if (displayed >= peak) {
        let dt = max(params.perception.w, 0.000001);
        let rise_velocity = (displayed - peak) * db_span / dt;
        peak = displayed;
        if (peak_momentum) {
            // Transfer only a small part of the measured upward velocity. The
            // peak remains responsive, but coasts visibly instead of flying
            // far beyond a sharp one-frame transient.
            fall_velocity = min(fall_velocity, -rise_velocity * 0.02);
            hold = 0.0;
        } else {
            fall_velocity = 0.0;
            hold = params.visual.w;
        }
    } else if (hold > 0.0) {
        hold = max(hold - params.perception.w, 0.0);
    } else {
        // Momentum uses negative velocity while traveling upward. Gravity
        // brings it to the apex, then the ordinary positive fall continues.
        let dt = params.perception.w;
        let next_velocity = fall_velocity + peak_gravity * dt;
        if (peak_momentum && fall_velocity < 0.0 && next_velocity >= 0.0) {
            let apex_time = -fall_velocity / max(peak_gravity, 0.000001);
            let traveled_db =
                fall_velocity * apex_time + 0.5 * peak_gravity * apex_time * apex_time;
            peak = min(1.0, peak - traveled_db / db_span);
            fall_velocity = 0.0;
            hold = params.visual.w;
        } else {
            let traveled_db = fall_velocity * dt + 0.5 * peak_gravity * dt * dt;
            fall_velocity = next_velocity;
            let next_peak = peak - traveled_db / db_span;
            if (next_peak >= 1.0 && fall_velocity < 0.0) {
                peak = 1.0;
                fall_velocity = 0.0;
                hold = params.visual.w;
            } else {
                peak = max(displayed, next_peak);
            }
        }
    }

    // Per-bin onset envelopes used only for light. Both trigger only when the
    // peak indicator itself moves upward, not for motion hidden below it. The
    // bar envelope fades faster so a transient does not leave the spectrum
    // body looking smeared while the peak marker keeps its longer glow.
    var rise_highlight = auxiliary[SMOOTHING_OFFSET + id.x].z;
    var bar_rise_highlight = auxiliary[SMOOTHING_OFFSET + id.x].w;
    if (params.perception.y > 0.5 || params.frequency_range.w <= 0.0001) {
        rise_highlight = 0.0;
        bar_rise_highlight = 0.0;
    } else {
        let dt = max(params.perception.w, 0.000001);
        let fade_seconds = max(params.auditory_1.w, 0.001);
        rise_highlight *= exp(-dt / fade_seconds);
        let bar_fade_seconds = max(fade_seconds * 0.33, 0.001);
        bar_rise_highlight *= exp(-dt / bar_fade_seconds);
        let rise_db = max(peak - previous_display.z, 0.0) * db_span;
        let full_scale_db = max(params.resonator_decay.w, 0.001);
        let onset = rise_db / full_scale_db;
        rise_highlight = min(rise_highlight + onset, 1.0);
        bar_rise_highlight = min(bar_rise_highlight + onset, 1.0);
    }
    auxiliary[SMOOTHING_OFFSET + id.x].z = rise_highlight;
    auxiliary[SMOOTHING_OFFSET + id.x].w = bar_rise_highlight;

    display_spectrum[id.x] = vec4(displayed, fall_velocity, peak, hold);
}
