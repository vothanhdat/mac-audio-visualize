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
}

@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<storage, read_write> spectrum: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(8) var<storage, read_write> auxiliary: array<vec4<f32>>;

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

fn selected_stage(
    order: u32,
    stage_0: vec2<f32>,
    stage_1: vec2<f32>,
    stage_2: vec2<f32>,
    stage_3: vec2<f32>,
) -> vec2<f32> {
    if (order == 1u) { return stage_0; }
    if (order == 2u) { return stage_1; }
    if (order == 3u) { return stage_2; }
    return stage_3;
}

@compute @workgroup_size(64)
fn analyze_erb(@builtin(global_invocation_id) id: vec3<u32>) {
    let bin = id.x;
    if (bin >= params.bin_count) { return; }

    let display_t = f32(bin) / f32(params.bin_count - 1u);
    let frequency_t = focused_log_frequency_t(display_t);
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    let frequency = lower * pow(upper / lower, frequency_t);
    let floor_db = local_db_floor(frequency_t);
    let reset = params.sample_counts_1.z != 0u;

    // Three vec4 values per bin: oscillator + stage 1, stages 2/3,
    // and stage 4. Cascading four complex one-pole sections produces the
    // fourth-order gammatone envelope used by practical auditory models.
    let base = bin * 3u;
    var packed_0 = auxiliary[base];
    var packed_1 = auxiliary[base + 1u];
    var packed_2 = auxiliary[base + 2u];
    var oscillator = packed_0.xy;
    var stage_0 = packed_0.zw;
    var stage_1 = packed_1.xy;
    var stage_2 = packed_1.zw;
    var stage_3 = packed_2.xy;
    var estimated_frequency = packed_2.z;
    var phase_coherence = packed_2.w;
    if (reset) {
        oscillator = vec2(1.0, 0.0);
        stage_0 = vec2(0.0);
        stage_1 = vec2(0.0);
        stage_2 = vec2(0.0);
        stage_3 = vec2(0.0);
        estimated_frequency = frequency;
        phase_coherence = 0.0;
    }

    let phase = 2.0 * PI * frequency / params.sample_rate;
    let rotation = vec2(cos(phase), -sin(phase));
    let erb_hz = 24.7 * (1.0 + 4.37 * frequency / 1000.0);
    let frequency_adaptation = smoothstep(
        0.0,
        1.0,
        clamp(
            log2(frequency / 50.0) / log2(10000.0 / 50.0),
            0.0,
            1.0,
        ),
    );
    let bandwidth_hz = 1.019 * erb_hz * params.auditory_0.x;
    let damping = exp(-2.0 * PI * bandwidth_hz / params.sample_rate);
    let feed = 1.0 - damping;
    let order = u32(clamp(round(params.auditory_0.z), 1.0, 4.0));
    let new_count = min(params.sample_counts_1.y, MAX_SAMPLE_COUNT);
    let start = MAX_SAMPLE_COUNT - new_count;
    var previous_output = selected_stage(order, stage_0, stage_1, stage_2, stage_3);
    var phase_sum = vec2(0.0);
    var phase_weight = 0.0;

    for (var index = start; index < MAX_SAMPLE_COUNT; index++) {
        let mixed = oscillator * samples[index];
        stage_0 = damping * stage_0 + feed * mixed;
        if (order >= 2u) { stage_1 = damping * stage_1 + feed * stage_0; }
        if (order >= 3u) { stage_2 = damping * stage_2 + feed * stage_1; }
        if (order >= 4u) { stage_3 = damping * stage_3 + feed * stage_2; }
        let current_output = selected_stage(order, stage_0, stage_1, stage_2, stage_3);
        phase_sum += vec2(
            previous_output.x * current_output.x + previous_output.y * current_output.y,
            previous_output.x * current_output.y - previous_output.y * current_output.x,
        );
        phase_weight += length(previous_output) * length(current_output);
        previous_output = current_output;
        oscillator = complex_multiply(oscillator, rotation);
    }
    oscillator /= max(length(oscillator), 0.000001);

    let output = selected_stage(order, stage_0, stage_1, stage_2, stage_3);
    if (phase_weight > 0.000000000001) {
        let residual_phase_step = atan2(phase_sum.y, phase_sum.x);
        let frequency_offset = clamp(
            residual_phase_step * params.sample_rate / (2.0 * PI),
            -2.0 * bandwidth_hz,
            2.0 * bandwidth_hz,
        );
        estimated_frequency = frequency + frequency_offset;
        phase_coherence = clamp(length(phase_sum) / phase_weight, 0.0, 1.0);
    }

    let magnitude = 2.0 * length(output);
    let db = 20.0 * log2(max(magnitude * params.gain, 0.000001)) / log2(10.0);
    var level = clamp(
        (db - floor_db) / max(params.db_scale.z - floor_db, 0.001),
        0.0,
        1.0,
    );
    // The exponent moves from linear displayed dB toward a compressive
    // cochlear-like response while retaining exact zero and full-scale levels.
    let compression_exponent = mix(1.0, 0.3, params.auditory_0.y);
    level = pow(level, compression_exponent);

    // A coherent complex phase trajectory identifies a stable tone inside the
    // broad auditory filter. Suppress that tone's filter skirts without
    // narrowing incoherent/noise energy or increasing the filter's latency.
    let tonal_width_ratio = mix(
        params.auditory_1.x,
        params.auditory_1.z,
        frequency_adaptation,
    );
    let tonal_width_hz = max(erb_hz * params.auditory_0.x * tonal_width_ratio, 0.01);
    let tonal_distance = abs(estimated_frequency - frequency) / tonal_width_hz;
    let tonal_weight = exp(-0.5 * tonal_distance * tonal_distance);
    let phase_sharpening_amount = mix(
        params.auditory_0.w,
        params.auditory_1.y,
        frequency_adaptation,
    );
    let phase_sharpening = phase_sharpening_amount * phase_coherence;
    level *= mix(1.0, tonal_weight, phase_sharpening);

    let previous = spectrum[bin];
    spectrum[bin] = vec4(level, previous.y, 0.0, 0.0);
    auxiliary[base] = vec4(oscillator, stage_0);
    auxiliary[base + 1u] = vec4(stage_1, stage_2);
    auxiliary[base + 2u] = vec4(stage_3, estimated_frequency, phase_coherence);
}
