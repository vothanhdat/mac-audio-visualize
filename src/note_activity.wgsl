// Fast mode reads short-window audio power; spectrum mode reads measured bins
// before bar gravity or peak hold. All analyzers share this parameter prefix.
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
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> left: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> right: array<vec4<f32>>;
// 0..11: note power, visible activity, dominant-note fade, reserved.
// 12: dominant pitch-class index. A single workgroup owns this tiny buffer.
@group(0) @binding(3) var<storage, read_write> notes: array<vec4<f32>, 13>;
@group(0) @binding(4) var<uniform> appearance: vec4<f32>;
@group(0) @binding(5) var<storage, read> fast_notes: array<f32, 12>;
@group(0) @binding(6) var<uniform> smoothing: vec4<f32>;
var<workgroup> powers: array<f32, 12>;
var<workgroup> total_power: f32;
var<workgroup> strongest_power: f32;
var<workgroup> dominant: u32;

fn focused_position(t: f32) -> f32 {
    if (t <= 0.0) { return 0.0; }
    if (t >= 1.0) { return 1.0; }
    let origin = clamp(params.axis_focus.y, 0.001, 0.999);
    let exponent = pow(4.0, params.axis_focus.x * 2.0 - 1.0);
    let logit = log(origin / (1.0 - origin));
    return 1.0 / (1.0 + exp(-(logit + exponent * (log(t / (1.0 - t)) - logit))));
}

fn frequency_t_at_bin(bin: u32) -> f32 {
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    if (params.perception.x > 0.5) {
        let first = ceil(69.0 + 12.0 * log2(lower / 440.0) - 0.0001);
        let hz = 440.0 * pow(2.0, (first + f32(bin) - 69.0) / 12.0);
        return clamp(log2(hz / lower) / log2(upper / lower), 0.0, 1.0);
    }
    let t = f32(bin) / f32(max(params.bin_count - 1u, 1u));
    if (t <= 0.0 || t >= 1.0) { return t; }
    let origin = clamp(params.axis_focus.y, 0.001, 0.999);
    let exponent = pow(4.0, params.axis_focus.x * 2.0 - 1.0);
    let logit = log(origin / (1.0 - origin));
    return 1.0 / (1.0 + exp(-(logit + (log(t / (1.0 - t)) - logit) / exponent)));
}

fn bin_power(bin: u32) -> f32 {
    let floor_db = mix(params.db_scale.x, params.db_scale.y,
        smoothstep(0.0, 1.0, frequency_t_at_bin(bin)));
    let levels = vec2(left[bin].x, right[bin].x);
    let db = vec2(floor_db) + levels * (params.db_scale.z - floor_db);
    // A cropped/empty bin has zero energy, regardless of its local dB floor.
    let power = select(vec2(0.0), pow(vec2(10.0), db * 0.1), levels > vec2(0.00001));
    return (power.x + power.y) * 0.5;
}

fn power_at_midi(midi: f32) -> f32 {
    let hz = 440.0 * pow(2.0, (midi - 69.0) / 12.0);
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    if (hz < max(lower, appearance.y) || hz > min(upper, appearance.z)) { return 0.0; }
    var bin = focused_position(log2(hz / lower) / log2(upper / lower))
        * f32(params.bin_count - 1u);
    if (params.perception.x > 0.5) {
        bin = midi - ceil(69.0 + 12.0 * log2(lower / 440.0) - 0.0001);
    }
    if (bin < 0.0 || bin > f32(params.bin_count - 1u)) { return 0.0; }
    let low = u32(floor(bin));
    return mix(bin_power(low), bin_power(min(low + 1u, params.bin_count - 1u)), fract(bin));
}

@compute @workgroup_size(12)
fn analyze_notes(@builtin(local_invocation_index) note: u32) {
    var energy = 0.0;
    // C1–B7: fold matching notes across octaves. Sampling at semitone centers
    // avoids giving densely spaced display bins more votes than other notes.
    let fast = appearance.w > 0.5;
    if (fast) {
        energy = fast_notes[note];
    } else {
        for (var octave = 1u; octave <= 7u; octave++) {
            energy += power_at_midi(f32((octave + 1u) * 12u + note));
        }
    }
    let previous = notes[note];
    let dt = clamp(params.perception.w, 0.0, 0.1);
    let tau = select(0.08, 0.008, energy > previous.x);
    let reset = params.perception.y > 0.5;
    // Fast audio windows already limit memory: select the winner from current
    // energy, so an old, louder note cannot linger in a second release filter.
    let power = select(mix(previous.x, energy, 1.0 - exp(-dt / tau)), energy, reset || fast);
    powers[note] = power;
    workgroupBarrier();
    if (note == 0u) {
        var sum = 0.0;
        var best = 0u;
        for (var i = 0u; i < 12u; i++) {
            sum += powers[i];
            if (powers[i] > powers[best]) { best = i; }
        }
        let old = min(u32(notes[12].x), 11u);
        // Keep close contenders from swapping the primary highlight every frame.
        let lead = select(1.05, 1.01, fast);
        if (!reset && powers[old] * lead >= powers[best]) { best = old; }
        dominant = best;
        strongest_power = powers[best];
        total_power = sum;
        notes[12] = vec4(f32(best), 0.0, 0.0, 0.0);
    }
    workgroupBarrier();
    // Absolute gating lets the entire ribbon settle during silence. Relative
    // contrast keeps a few active notes visible without lighting every note.
    let audible = smoothstep(-72.0, -48.0,
        10.0 * log2(max(total_power, 1e-12)) / log2(10.0));
    let relative = powers[note] / max(strongest_power, 1e-12);
    var activity = pow(clamp(relative, 0.0, 1.0), 0.7) * audible;
    let winner = select(0.0, audible, note == dominant);
    let highlight_tau = select(0.06, 0.02, winner > previous.z);
    var fade = select(mix(previous.z, winner, 1.0 - exp(-dt / highlight_tau)), winner, reset);
    if (fast) {
        fade = max(winner, previous.z * exp(-dt / 0.012));
    }
    // Smooth only the visible glow and label size. Winner selection above
    // still reads current energy, preserving fast audio detection.
    if (smoothing.x > 0.0 && !reset) {
        let blend = 1.0 - exp(-dt / max(smoothing.x, 0.001));
        activity = mix(previous.y, activity, blend);
        fade = mix(previous.z, winner, blend);
    }
    notes[note] = vec4(power, activity, fade, 0.0);
}
