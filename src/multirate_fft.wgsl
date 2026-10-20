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
    resonator_stream_extra: vec4<u32>,
    hybrid_match_corrections: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> magnitudes: array<f32>;
@group(0) @binding(1) var<storage, read_write> spectrum: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

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

fn fft_size(band: u32) -> u32 {
    if (band < 4u) { return params.fft_sizes_0[band]; }
    return params.fft_sizes_1.x;
}

fn decimation(band: u32) -> u32 {
    if (band < 4u) { return params.fft_decimations_0[band]; }
    return params.fft_decimations_1.x;
}

fn magnitude_offset(band: u32) -> u32 {
    var offset = 0u;
    for (var index = 0u; index < band; index += 1u) {
        offset += fft_size(index) / 2u + 1u;
    }
    return offset;
}

fn magnitude_at(band: u32, frequency: f32) -> f32 {
    let size = fft_size(band);
    let effective_rate = params.sample_rate / f32(decimation(band));
    if (frequency >= effective_rate * 0.5) { return 0.0; }
    let fractional_bin = frequency * f32(size) / effective_rate;
    let lower_bin = min(u32(floor(fractional_bin)), size / 2u);
    let upper_bin = min(lower_bin + 1u, size / 2u);
    let blend = fract(fractional_bin);
    let offset = magnitude_offset(band);
    return mix(magnitudes[offset + lower_bin], magnitudes[offset + upper_bin], blend);
}

fn multirate_magnitude(frequency: f32) -> f32 {
    let band_count = params.fft_sizes_1.y;
    for (var band = 0u; band + 1u < band_count; band += 1u) {
        let crossover = params.fft_crossovers[band];
        let blend_start = crossover * 0.85;
        let blend_end = crossover * 1.15;
        if (frequency < blend_start) {
            return magnitude_at(band, frequency);
        }
        if (frequency < blend_end) {
            return mix(
                magnitude_at(band, frequency),
                magnitude_at(band + 1u, frequency),
                smoothstep(blend_start, blend_end, frequency),
            );
        }
    }
    return magnitude_at(band_count - 1u, frequency);
}

fn level_from_magnitude(magnitude: f32, frequency_t: f32) -> f32 {
    let db = 20.0 * log2(max(magnitude * params.gain, 0.000001)) / log2(10.0);
    let floor_db = local_db_floor(frequency_t);
    return clamp(
        (db - floor_db) / max(params.db_scale.z - floor_db, 0.001),
        0.0,
        1.0,
    );
}

fn temporal_level(measured: f32, previous: f32, frequency: f32) -> f32 {
    if (params.attack_1.y <= 0.5) { return measured; }
    var attack = params.attack_1.x;
    var decay = params.decay_1.x;
    if (frequency < 80.0) {
        attack = params.attack_0.x;
        decay = params.decay_0.x;
    } else if (frequency < 320.0) {
        attack = params.attack_0.y;
        decay = params.decay_0.y;
    } else if (frequency < 1280.0) {
        attack = params.attack_0.z;
        decay = params.decay_0.z;
    } else if (frequency < 5120.0) {
        attack = params.attack_0.w;
        decay = params.decay_0.w;
    }
    return mix(measured, previous, select(decay, attack, measured > previous));
}

@compute @workgroup_size(64)
fn map_fft(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    if (bin >= params.bin_count) { return; }
    let display_t = f32(bin) / f32(params.bin_count - 1u);
    let frequency_t = focused_log_frequency_t(display_t);
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    let frequency = lower * pow(upper / lower, frequency_t);
    let measured = level_from_magnitude(multirate_magnitude(frequency), frequency_t);
    let previous = spectrum[bin];
    let output = temporal_level(measured, previous.x, frequency);
    spectrum[bin] = vec4(output, previous.y, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn map_hybrid(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    if (bin >= params.bin_count) { return; }
    let display_t = f32(bin) / f32(params.bin_count - 1u);
    let frequency_t = focused_log_frequency_t(display_t);
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    let frequency = lower * pow(upper / lower, frequency_t);
    let resonator_limit = params.sample_rate * 0.091;
    if (frequency <= resonator_limit) { return; }

    let previous = spectrum[bin];
    let fft_band = select(1u, 0u, frequency <= params.sample_rate * 0.181);
    let adaptive_correction_db = select(
        params.hybrid_match_corrections.z,
        params.hybrid_match_corrections.x,
        fft_band == 0u,
    );
    let gain_db =
        select(params.hybrid.y, params.hybrid.x, fft_band == 0u)
        + adaptive_correction_db;
    let band_gain = exp2(gain_db / 6.020599913);
    let measured = level_from_magnitude(
        magnitude_at(fft_band, frequency) * band_gain,
        frequency_t,
    );
    let fft_history = select(previous.w, measured, params.perception.y > 0.5);
    let fft_level = temporal_level(measured, fft_history, frequency);
    spectrum[bin] = vec4(fft_level, previous.y, 0.0, fft_level);
}
