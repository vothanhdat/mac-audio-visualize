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
}

@group(0) @binding(1) var<storage, read_write> spectrum: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(7) var<storage, read_write> display_spectrum: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> auxiliary: array<vec4<f32>>;

const MAX_BIN_COUNT: u32 = 2048u;
const HIGHLIGHT_OFFSET: u32 = MAX_BIN_COUNT * 3u;
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

@compute @workgroup_size(64)
fn compose_display(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.bin_count) { return; }
    let raw = spectrum[id.x];
    let point = frequency_at_bin(id.x);
    let frequency_t = point.y;
    let db_span = max(params.db_scale.z - local_db_floor(frequency_t), 0.001);
    let previous_display = display_spectrum[id.x];
    var measured = raw.x;
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
    var rise_highlight = auxiliary[HIGHLIGHT_OFFSET + id.x].z;
    var bar_rise_highlight = auxiliary[HIGHLIGHT_OFFSET + id.x].w;
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
    auxiliary[HIGHLIGHT_OFFSET + id.x].z = rise_highlight;
    auxiliary[HIGHLIGHT_OFFSET + id.x].w = bar_rise_highlight;

    display_spectrum[id.x] = vec4(displayed, fall_velocity, peak, hold);
}
