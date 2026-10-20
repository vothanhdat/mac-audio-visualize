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
}

struct RenderChannelParams {
    channel_index: f32,
    channel_count: f32,
    bottom_mirror: f32,
    stereo_treble_center: f32,
    mirror_baseline_percent: f32,
    mirror_initial_fade_percent: f32,
    mirror_fade_percent: f32,
    axis_labels_overlay: f32,
    axis_label_mode: f32,
    show_frequency_grid: f32,
    bar_depth_pixels: f32,
    peak_depth_brightness_percent: f32,
    volume_stage_count: f32,
    _padding_0: f32,
    _padding_1: f32,
    _padding_2: f32,
}

@group(0) @binding(7) var<storage, read> spectrum: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> auxiliary: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(9) var<uniform> render_channel: RenderChannelParams;

const LEFT: f32 = 0.0;
const RIGHT: f32 = 0.0;
const TOP: f32 = 20.0;
const BOTTOM: f32 = 32.0;
const MAX_BIN_COUNT: u32 = 2048u;
const SMOOTHING_OFFSET: u32 = MAX_BIN_COUNT * 3u;

struct GridOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn grid_vs(@builtin(vertex_index) vertex: u32) -> GridOut {
    let positions = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0),
    );
    var out: GridOut;
    out.position = vec4(positions[vertex], 0.0, 1.0);
    out.uv = positions[vertex] * vec2(0.5, -0.5) + vec2(0.5);
    return out;
}

fn plot_size() -> vec2<f32> {
    let bottom_padding = select(BOTTOM, 0.0, render_channel.axis_labels_overlay >= 0.5);
    return max(params.resolution - vec2(LEFT + RIGHT, TOP + bottom_padding), vec2(1.0));
}

fn axis_label_origin_y() -> f32 {
    if (render_channel.axis_labels_overlay >= 0.5) {
        return params.resolution.y - 14.0;
    }
    return TOP + plot_size().y + 10.0;
}

fn focused_position(log_frequency_t: f32) -> f32 {
    if (log_frequency_t <= 0.0) { return 0.0; }
    if (log_frequency_t >= 1.0) { return 1.0; }
    let origin = clamp(params.axis_focus.y, 0.001, 0.999);
    let exponent = pow(4.0, params.axis_focus.x * 2.0 - 1.0);
    let origin_logit = log(origin / (1.0 - origin));
    let source_logit = log(log_frequency_t / (1.0 - log_frequency_t));
    let display_logit = origin_logit + exponent * (source_logit - origin_logit);
    return 1.0 / (1.0 + exp(-display_logit));
}

fn inverse_focused_position(display_t: f32) -> f32 {
    if (display_t <= 0.0) { return 0.0; }
    if (display_t >= 1.0) { return 1.0; }
    let origin = clamp(params.axis_focus.y, 0.001, 0.999);
    let exponent = pow(4.0, params.axis_focus.x * 2.0 - 1.0);
    let origin_logit = log(origin / (1.0 - origin));
    let display_logit = log(display_t / (1.0 - display_t));
    let source_logit = origin_logit + (display_logit - origin_logit) / exponent;
    return 1.0 / (1.0 + exp(-source_logit));
}

fn frequency_x(hz: f32) -> f32 {
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    if (params.perception.x > 0.5) {
        let first_midi = ceil(69.0 + 12.0 * log2(lower / 440.0) - 0.0001);
        let midi_note = 69.0 + 12.0 * log2(hz / 440.0);
        let normalized = (midi_note - first_midi + 0.5) / f32(params.bin_count);
        if (render_channel.channel_count < 1.5) {
            return LEFT + normalized * plot_size().x;
        }
        let mirrored_position = select(
            0.5 + normalized * 0.5,
            1.0 - normalized * 0.5,
            render_channel.stereo_treble_center >= 0.5,
        );
        return LEFT + plot_size().x * mirrored_position;
    }
    let log_frequency_t = log2(hz / lower) / log2(upper / lower);
    let normalized = focused_position(log_frequency_t);
    if (render_channel.channel_count < 1.5) {
        return LEFT + normalized * plot_size().x;
    }
    let mirrored_position = select(
        0.5 + normalized * 0.5,
        1.0 - normalized * 0.5,
        render_channel.stereo_treble_center >= 0.5,
    );
    return LEFT + plot_size().x * mirrored_position;
}

fn mirrored_frequency_x(hz: f32) -> f32 {
    return LEFT + plot_size().x - frequency_x(hz);
}

fn frequency_line_alpha(pixel_x: f32, hz: f32, width: f32) -> f32 {
    let primary = line_alpha(pixel_x - frequency_x(hz), width);
    if (render_channel.channel_count < 1.5) {
        return primary;
    }
    return max(primary, line_alpha(pixel_x - mirrored_frequency_x(hz), width));
}

fn natural_note_grid_alpha(pixel_x: f32, lower: f32, upper: f32) -> f32 {
    var display_t = clamp((pixel_x - LEFT) / plot_size().x, 0.0, 1.0);
    if (render_channel.channel_count >= 1.5) {
        display_t = abs(display_t * 2.0 - 1.0);
        if (render_channel.stereo_treble_center >= 0.5) {
            display_t = 1.0 - display_t;
        }
    }
    let frequency_t = inverse_focused_position(display_t);
    let frequency = lower * pow(upper / lower, frequency_t);
    let midi_note = 69.0 + 12.0 * log2(frequency / 440.0);
    let octave_c_midi = floor(midi_note / 12.0) * 12.0;
    let octave_c = 440.0 * pow(2.0, (octave_c_midi - 69.0) / 12.0);

    // Only reveal the natural-note subdivisions when an octave has enough
    // screen space for seven distinct notes.
    let octave_width = abs(frequency_x(octave_c * 2.0) - frequency_x(octave_c));
    if (octave_width < 84.0) { return 0.0; }

    // D, E, F, G, A, B. C is already represented by the labelled octave grid.
    let semitones = array<f32, 6>(2.0, 4.0, 5.0, 7.0, 9.0, 11.0);
    var alpha = 0.0;
    for (var note = 0u; note < 6u; note++) {
        let note_frequency = octave_c * pow(2.0, semitones[note] / 12.0);
        if (note_frequency >= lower && note_frequency <= upper) {
            alpha = max(alpha, frequency_line_alpha(pixel_x, note_frequency, 0.15));
        }
    }
    return alpha * 0.18;
}

fn line_alpha(distance: f32, width: f32) -> f32 {
    return 1.0 - smoothstep(width, width + 1.0, abs(distance));
}

// Tiny 3x5 GPU bitmap font. Codes 0..9 are digits, then K, minus, H, Z, D, B, C.
fn glyph_row(code: u32, row: u32) -> u32 {
    let empty = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
    var rows = empty;
    switch code {
        case 0u: { rows = array<u32, 5>(7u, 5u, 5u, 5u, 7u); }
        case 1u: { rows = array<u32, 5>(2u, 6u, 2u, 2u, 7u); }
        case 2u: { rows = array<u32, 5>(7u, 1u, 7u, 4u, 7u); }
        case 3u: { rows = array<u32, 5>(7u, 1u, 7u, 1u, 7u); }
        case 4u: { rows = array<u32, 5>(5u, 5u, 7u, 1u, 1u); }
        case 5u: { rows = array<u32, 5>(7u, 4u, 7u, 1u, 7u); }
        case 6u: { rows = array<u32, 5>(7u, 4u, 7u, 5u, 7u); }
        case 7u: { rows = array<u32, 5>(7u, 1u, 1u, 1u, 1u); }
        case 8u: { rows = array<u32, 5>(7u, 5u, 7u, 5u, 7u); }
        case 9u: { rows = array<u32, 5>(7u, 5u, 7u, 1u, 7u); }
        case 10u: { rows = array<u32, 5>(5u, 5u, 6u, 5u, 5u); } // K
        case 11u: { rows = array<u32, 5>(0u, 0u, 7u, 0u, 0u); } // minus
        case 12u: { rows = array<u32, 5>(5u, 5u, 7u, 5u, 5u); } // H
        case 13u: { rows = array<u32, 5>(7u, 1u, 2u, 4u, 7u); } // Z
        case 14u: { rows = array<u32, 5>(6u, 5u, 5u, 5u, 6u); } // D
        case 15u: { rows = array<u32, 5>(6u, 5u, 6u, 5u, 6u); } // B
        case 16u: { rows = array<u32, 5>(7u, 4u, 4u, 4u, 7u); } // C
        default: {}
    }
    return rows[min(row, 4u)];
}

fn glyph(pixel: vec2<f32>, origin: vec2<f32>, code: u32, scale: f32) -> f32 {
    let local = floor((pixel - origin) / scale);
    if (local.x < 0.0 || local.x >= 3.0 || local.y < 0.0 || local.y >= 5.0) { return 0.0; }
    let column = u32(local.x);
    let bits = glyph_row(code, u32(local.y));
    return f32((bits >> (2u - column)) & 1u);
}

fn x_label_code(tick: u32, character: u32) -> i32 {
    // Scientific pitch notation: C0 through C11.
    if (character == 0u) { return 16; }
    if (tick < 10u) {
        return select(-1, i32(tick), character == 1u);
    }
    if (character == 1u) { return 1; }
    return select(-1, i32(tick - 10u), character == 2u);
}

fn hz_label_length(tick: u32) -> u32 {
    if (tick <= 1u || (tick >= 5u && tick <= 7u)) {
        return 2u;
    }
    return 3u;
}

fn hz_label_code(tick: u32, character: u32) -> i32 {
    var codes = vec4<i32>(-1, -1, -1, -1);
    switch tick {
        case 0u: { codes = vec4<i32>(2, 0, -1, -1); }   // 20
        case 1u: { codes = vec4<i32>(5, 0, -1, -1); }   // 50
        case 2u: { codes = vec4<i32>(1, 0, 0, -1); }    // 100
        case 3u: { codes = vec4<i32>(2, 0, 0, -1); }    // 200
        case 4u: { codes = vec4<i32>(5, 0, 0, -1); }    // 500
        case 5u: { codes = vec4<i32>(1, 10, -1, -1); }  // 1K
        case 6u: { codes = vec4<i32>(2, 10, -1, -1); }  // 2K
        case 7u: { codes = vec4<i32>(5, 10, -1, -1); }  // 5K
        case 8u: { codes = vec4<i32>(1, 0, 10, -1); }   // 10K
        case 9u: { codes = vec4<i32>(2, 0, 10, -1); }   // 20K
        case 10u: { codes = vec4<i32>(4, 0, 10, -1); }  // 40K
        default: {}
    }
    return codes[min(character, 3u)];
}

fn axis_text_alpha(pixel: vec2<f32>, lower: f32, upper: f32) -> f32 {
    let note_frequencies = array<f32, 12>(
        16.3516, 32.7032, 65.4064, 130.8128, 261.6256, 523.2511,
        1046.5023, 2093.0045, 4186.0090, 8372.0181, 16744.0362, 33488.0724,
    );
    let hz_frequencies = array<f32, 11>(
        20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
        2000.0, 5000.0, 10000.0, 20000.0, 40000.0,
    );
    let show_hz = render_channel.axis_label_mode >= 0.5;
    let scale = 2.0;
    let label_y = axis_label_origin_y();
    let copies = select(1u, 2u, render_channel.channel_count >= 1.5);
    var text_alpha = 0.0;

    if (show_hz) {
        for (var tick = 0u; tick < 11u; tick++) {
            if (hz_frequencies[tick] >= lower && hz_frequencies[tick] <= upper) {
                let length = hz_label_length(tick);
                let label_width = (f32(length) * 4.0 - 1.0) * scale;
                for (var copy = 0u; copy < copies; copy++) {
                    let tick_x = select(
                        frequency_x(hz_frequencies[tick]),
                        mirrored_frequency_x(hz_frequencies[tick]),
                        copy == 1u,
                    );
                    let origin_x = clamp(
                        tick_x - label_width * 0.5,
                        0.0,
                        params.resolution.x - label_width,
                    );
                    for (var character = 0u; character < 3u; character++) {
                        let code = hz_label_code(tick, character);
                        if (code >= 0) {
                            let origin =
                                vec2(origin_x + f32(character) * 4.0 * scale, label_y);
                            text_alpha = max(text_alpha, glyph(pixel, origin, u32(code), scale));
                        }
                    }
                }
            }
        }
    } else {
        for (var tick = 0u; tick < 12u; tick++) {
            if (note_frequencies[tick] >= lower && note_frequencies[tick] <= upper) {
                let length = select(2.0, 3.0, tick >= 10u);
                let label_width = (length * 4.0 - 1.0) * scale;
                for (var copy = 0u; copy < copies; copy++) {
                    let tick_x = select(
                        frequency_x(note_frequencies[tick]),
                        mirrored_frequency_x(note_frequencies[tick]),
                        copy == 1u,
                    );
                    let origin_x = clamp(
                        tick_x - label_width * 0.5,
                        0.0,
                        params.resolution.x - label_width,
                    );
                    for (var character = 0u; character < 3u; character++) {
                        let code = x_label_code(tick, character);
                        if (code >= 0) {
                            let origin =
                                vec2(origin_x + f32(character) * 4.0 * scale, label_y);
                            text_alpha = max(text_alpha, glyph(pixel, origin, u32(code), scale));
                        }
                    }
                }
            }
        }
    }
    return text_alpha;
}

@fragment
fn grid_fs(in: GridOut) -> @location(0) vec4<f32> {
    let pixel = in.uv * params.resolution;
    let size = plot_size();
    let inside = pixel.x >= LEFT && pixel.x <= LEFT + size.x && pixel.y >= TOP && pixel.y <= TOP + size.y;
    let grid_color = vec3(0.20, 0.44, 0.70);
    var grid_alpha = 0.0;

    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    let note_frequencies = array<f32, 12>(
        16.3516, 32.7032, 65.4064, 130.8128, 261.6256, 523.2511,
        1046.5023, 2093.0045, 4186.0090, 8372.0181, 16744.0362, 33488.0724,
    );
    let hz_frequencies = array<f32, 11>(
        20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
        2000.0, 5000.0, 10000.0, 20000.0, 40000.0,
    );
    let show_hz = render_channel.axis_label_mode >= 0.5;
    if (inside && render_channel.show_frequency_grid >= 0.5) {
        let edge_alpha = max(line_alpha(pixel.x - LEFT, 0.7), line_alpha(pixel.x - LEFT - size.x, 0.7));
        grid_alpha = max(grid_alpha, edge_alpha * 0.75);
        if (show_hz) {
            for (var i = 0u; i < 11u; i++) {
                if (hz_frequencies[i] >= lower && hz_frequencies[i] <= upper) {
                    let major = i == 2u || i == 5u || i == 8u;
                    let alpha =
                        frequency_line_alpha(pixel.x, hz_frequencies[i], select(0.35, 0.7, major));
                    grid_alpha = max(grid_alpha, alpha * select(0.35, 0.75, major));
                }
            }
        } else {
            for (var i = 0u; i < 12u; i++) {
                if (note_frequencies[i] >= lower && note_frequencies[i] <= upper) {
                    let major = i == 1u || i == 4u || i == 7u || i == 10u;
                    let alpha = frequency_line_alpha(
                        pixel.x,
                        note_frequencies[i],
                        select(0.35, 0.7, major),
                    );
                    grid_alpha = max(grid_alpha, alpha * select(0.35, 0.75, major));
                }
            }
            grid_alpha = max(grid_alpha, natural_note_grid_alpha(pixel.x, lower, upper));
        }
    }

    let text_color = vec3(0.52, 0.68, 0.82);
    let text_alpha = select(
        axis_text_alpha(pixel, lower, upper) * 0.55,
        0.0,
        render_channel.axis_labels_overlay >= 0.5,
    );
    grid_alpha *= 0.20;
    let alpha = max(grid_alpha, text_alpha);
    let color = mix(grid_color, text_color, text_alpha);
    return vec4(color, alpha);
}

@fragment
fn axis_overlay_fs(in: GridOut) -> @location(0) vec4<f32> {
    let pixel = in.uv * params.resolution;
    let lower = params.frequency_range.x;
    let upper = min(params.frequency_range.y, params.sample_rate * 0.48);
    if (axis_text_alpha(pixel, lower, upper) < 0.5) {
        discard;
    }
    // Overlay labels are intentionally solid white for maximum consistency.
    return vec4(1.0);
}

struct BarOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) energy: f32,
    @location(2) bin_t: f32,
    @location(3) response: f32,
    @location(4) reflection_alpha: f32,
    @location(5) transient: f32,
}

fn channel_pitch() -> f32 {
    return plot_size().x / render_channel.channel_count / f32(params.bin_count);
}

fn bar_center_x(bin: u32) -> f32 {
    let pitch = channel_pitch();
    if (render_channel.channel_count < 1.5) {
        return LEFT + (f32(bin) + 0.5) * pitch;
    }
    let center = LEFT + plot_size().x * 0.5;
    let direction = select(-1.0, 1.0, render_channel.channel_index >= 0.5);
    let display_bin = select(
        bin,
        params.bin_count - 1u - bin,
        render_channel.stereo_treble_center >= 0.5,
    );
    return center + direction * (f32(display_bin) + 0.5) * pitch;
}

fn prism_shift(bin: u32) -> vec2<f32> {
    let depth_pixels = clamp(render_channel.bar_depth_pixels, 0.0, 300.0);
    let radial = clamp(
        (bar_center_x(bin) / params.resolution.x - 0.5) * 2.0,
        -1.0,
        1.0,
    );
    return vec2(radial * depth_pixels * 0.75, depth_pixels * 0.65);
}

fn spectrum_bin(instance: u32) -> u32 {
    return instance % params.bin_count;
}

fn is_bottom_copy(instance: u32) -> bool {
    return render_channel.bottom_mirror >= 0.5 && instance >= params.bin_count;
}

fn bar_y(corner_y: f32, value: f32, instance: u32) -> f32 {
    let size = plot_size();
    if (render_channel.bottom_mirror < 0.5) {
        return TOP + size.y - corner_y * value * size.y;
    }
    let baseline = clamp(render_channel.mirror_baseline_percent * 0.01, 0.0, 1.0);
    let center = TOP + size.y * baseline;
    let bottom_copy = is_bottom_copy(instance);
    let direction = select(-1.0, 1.0, bottom_copy);
    let mirrored_height = size.y * baseline;
    return center + direction * corner_y * value * mirrored_height;
}

fn peak_center_y(value: f32, instance: u32) -> f32 {
    let size = plot_size();
    if (render_channel.bottom_mirror < 0.5) {
        return TOP + size.y * (1.0 - value);
    }
    let baseline = clamp(render_channel.mirror_baseline_percent * 0.01, 0.0, 1.0);
    let center = TOP + size.y * baseline;
    let bottom_copy = is_bottom_copy(instance);
    let direction = select(-1.0, 1.0, bottom_copy);
    let mirrored_height = size.y * baseline;
    return center + direction * value * mirrored_height;
}

fn reflection_alpha(progress: f32, instance: u32) -> f32 {
    if (!is_bottom_copy(instance)) {
        return 1.0;
    }
    let initial_fade = clamp(render_channel.mirror_initial_fade_percent * 0.01, 0.0, 1.0);
    let fade_strength = clamp(render_channel.mirror_fade_percent * 0.01, 0.0, 20.0);
    let distance = clamp(progress, 0.0, 1.0);
    // One unit of strength leaves 1% brightness at the reflected tip.
    let attenuation = exp2(-6.643856 * fade_strength * distance);
    return (1.0 - initial_fade) * attenuation;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> BarOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let corner = corners[vertex];
    let bin = spectrum_bin(instance);
    let pitch = channel_pitch();
    let width = max(pitch - params.frequency_range.z, 0.5);
    let x_px = bar_center_x(bin) - width * 0.5 + corner.x * width;
    let detailed = spectrum[bin].x;
    let response = auxiliary[SMOOTHING_OFFSET + bin].w;
    let displayed = detailed;
    let y_px = bar_y(corner.y, displayed, instance);

    var out: BarOut;
    out.position = vec4(x_px / params.resolution.x * 2.0 - 1.0, 1.0 - y_px / params.resolution.y * 2.0, 0.0, 1.0);
    out.uv = corner;
    out.energy = detailed;
    out.bin_t = f32(bin) / f32(params.bin_count - 1u);
    out.response = response;
    out.reflection_alpha = reflection_alpha(corner.y, instance);
    out.transient = response;
    return out;
}

fn shaded_bar(in: BarOut, face_kind: f32) -> vec4<f32> {
    let low = vec3(0.08, 0.40, 1.0);
    let middle = vec3(0.02, 0.95, 0.78);
    let high = vec3(1.0, 0.20, 0.58);
    let frequency_color = mix(mix(low, middle, min(in.bin_t * 2.0, 1.0)), high, max(in.bin_t * 2.0 - 1.0, 0.0));
    let white_hot = smoothstep(0.70, 1.0, in.energy) * 0.88;
    let onset = in.transient * params.frequency_range.w;
    let transient_white = clamp(onset * 0.10, 0.0, 0.12);
    let luminous_color = mix(
        frequency_color,
        vec3(1.0),
        max(white_hot, transient_white),
    );
    let vertical_glow = 0.58 + 0.42 * in.uv.y;
    // Energy is linear in displayed dB (-80..0): louder bars emit more light
    // without changing their measured height.
    let energy_glow = mix(params.visual.x, params.visual.y, in.energy * in.energy);
    let transient_glow = 1.0 + onset * 1.35;
    var top_fade = 1.0;
    if (params.db_scale.w > 0.0001 && abs(face_kind - 1.0) > 0.25) {
        top_fade = 1.0 - smoothstep(1.0 - params.db_scale.w, 1.0, in.uv.y);
    }
    var face_light = 1.0;
    if (face_kind > 1.5) {
        face_light = 0.58;
    } else if (face_kind > 0.5) {
        face_light = 1.32;
    }
    return vec4(
        luminous_color * vertical_glow * energy_glow * transient_glow *
            face_light * in.reflection_alpha,
        0.92 * top_fade * in.reflection_alpha,
    );
}

@fragment
fn fs_main(in: BarOut) -> @location(0) vec4<f32> {
    return shaded_bar(in, 0.0);
}

@vertex
fn prism_vs(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
) -> BarOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let stage = vertex / 6u;
    let corner = corners[vertex % 6u];
    let bin = spectrum_bin(instance);
    let pitch = channel_pitch();
    let width = max(pitch - params.frequency_range.z, 0.5);
    let energy = spectrum[bin].x;
    let stage_count = u32(clamp(
        round(render_channel.volume_stage_count),
        3.0,
        8.0,
    ));
    let layer_t = exp2(f32(stage) - f32(stage_count - 1u));
    let shift = prism_shift(bin) * layer_t;
    let x_px =
        bar_center_x(bin) - width * 0.5 + corner.x * width + shift.x;
    let y_px = bar_y(corner.y, energy, instance) + shift.y;
    let stage_active = stage < stage_count;
    let final_layer = stage + 1u >= stage_count;

    var out: BarOut;
    if (stage_active) {
        out.position = vec4(
            x_px / params.resolution.x * 2.0 - 1.0,
            1.0 - y_px / params.resolution.y * 2.0,
            0.0,
            1.0,
        );
    } else {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
    }
    out.uv = corner;
    out.energy = energy;
    out.bin_t = f32(bin) / f32(params.bin_count - 1u);
    out.response = layer_t;
    out.reflection_alpha = select(0.0, 1.0, final_layer);
    out.transient = auxiliary[SMOOTHING_OFFSET + bin].w;
    return out;
}

@fragment
fn prism_fs(in: BarOut) -> @location(0) vec4<f32> {
    if (in.energy <= 0.0001) {
        discard;
    }
    let low = vec3(0.08, 0.40, 1.0);
    let middle = vec3(0.02, 0.95, 0.78);
    let high = vec3(1.0, 0.20, 0.58);
    let frequency_color = mix(
        mix(low, middle, min(in.bin_t * 2.0, 1.0)),
        high,
        max(in.bin_t * 2.0 - 1.0, 0.0),
    );
    let white_hot = smoothstep(0.70, 1.0, in.energy) * 0.88;
    let onset = in.transient * params.frequency_range.w;
    let transient_white = clamp(onset * 0.10, 0.0, 0.12);
    let luminous_color = mix(
        frequency_color,
        vec3(1.0),
        max(white_hot, transient_white),
    );
    let vertical_glow = 0.58 + 0.42 * in.uv.y;
    let energy_glow =
        mix(params.visual.x, params.visual.y, in.energy * in.energy);
    var top_fade = 1.0;
    if (params.db_scale.w > 0.0001) {
        top_fade =
            1.0 - smoothstep(1.0 - params.db_scale.w, 1.0, in.uv.y);
    }
    let final_layer = in.reflection_alpha >= 0.5;
    let depth_light = mix(0.22, 0.72, pow(in.response, 1.4));
    let layer_light = select(depth_light, 1.0, final_layer);
    let layer_alpha = select(
        0.12 + 0.16 * in.response,
        0.92,
        final_layer,
    );
    let transient_glow = 1.0 + onset * 1.35;
    return vec4(
        luminous_color * vertical_glow * energy_glow * layer_light * transient_glow,
        layer_alpha * top_fade,
    );
}

@vertex
fn glow_vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> BarOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let corner = corners[vertex];
    let bin = spectrum_bin(instance);
    let pitch = channel_pitch();
    let core_width = max(pitch - params.frequency_range.z, 0.5);
    let width = core_width + 7.0;
    let shift = prism_shift(bin);
    let x_px = bar_center_x(bin) - width * 0.5 + corner.x * width + shift.x;
    let energy = spectrum[bin].x;
    let y_px = bar_y(corner.y, energy, instance) + shift.y;

    var out: BarOut;
    out.position = vec4(x_px / params.resolution.x * 2.0 - 1.0, 1.0 - y_px / params.resolution.y * 2.0, 0.0, 1.0);
    out.uv = corner;
    out.energy = energy;
    out.bin_t = f32(bin) / f32(params.bin_count - 1u);
    out.response = auxiliary[SMOOTHING_OFFSET + bin].w;
    out.reflection_alpha = reflection_alpha(corner.y, instance);
    out.transient = out.response;
    return out;
}

@fragment
fn glow_fs(in: BarOut) -> @location(0) vec4<f32> {
    let low = vec3(0.08, 0.40, 1.0);
    let middle = vec3(0.02, 0.95, 0.78);
    let high = vec3(1.0, 0.20, 0.58);
    let frequency_color = mix(mix(low, middle, min(in.bin_t * 2.0, 1.0)), high, max(in.bin_t * 2.0 - 1.0, 0.0));
    let horizontal = pow(max(1.0 - abs(in.uv.x * 2.0 - 1.0), 0.0), 3.0);
    let onset = in.transient * params.frequency_range.w;
    let strength = horizontal * smoothstep(0.45, 1.0, in.energy) *
        (0.07 + 0.09 * in.uv.y) * (1.0 + onset * 1.8);
    return vec4(
        frequency_color * strength * in.reflection_alpha,
        strength * in.reflection_alpha,
    );
}

@vertex
fn peak_vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> BarOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let corner = corners[vertex];
    let bin = spectrum_bin(instance);
    let pitch = channel_pitch();
    let width = max(pitch - params.frequency_range.z, 0.5);
    let shift = prism_shift(bin);
    let x_px = bar_center_x(bin) - width * 0.5 + corner.x * width + shift.x;
    let peak = spectrum[bin].z;
    let center_y = peak_center_y(peak, instance);
    let marker_height = clamp(width * 0.38, 1.4, 3.0);
    let y_px = center_y + (corner.y - 0.5) * marker_height + shift.y;

    var out: BarOut;
    out.position = vec4(x_px / params.resolution.x * 2.0 - 1.0, 1.0 - y_px / params.resolution.y * 2.0, 0.0, 1.0);
    out.uv = corner;
    out.energy = peak;
    out.bin_t = f32(bin) / f32(params.bin_count - 1u);
    out.response = auxiliary[SMOOTHING_OFFSET + bin].z;
    out.reflection_alpha = reflection_alpha(peak, instance);
    out.transient = 0.0;
    return out;
}

@vertex
fn peak_volume_vs(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
) -> BarOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let stage = vertex / 6u;
    let corner = corners[vertex % 6u];
    let bin = spectrum_bin(instance);
    let pitch = channel_pitch();
    let width = max(pitch - params.frequency_range.z, 0.5);
    let stage_count = u32(clamp(
        round(render_channel.volume_stage_count),
        3.0,
        8.0,
    ));
    let layer_t = exp2(f32(stage) - f32(stage_count - 1u));
    let shift = prism_shift(bin) * layer_t;
    let peak = spectrum[bin].z;
    let center_y = peak_center_y(peak, instance);
    let marker_height = clamp(width * 0.38, 1.4, 3.0);
    let x_px =
        bar_center_x(bin) - width * 0.5 + corner.x * width + shift.x;
    let y_px =
        center_y + (corner.y - 0.5) * marker_height + shift.y;
    let stage_active = stage < stage_count;
    let final_layer = stage + 1u >= stage_count;

    var out: BarOut;
    if (stage_active) {
        out.position = vec4(
            x_px / params.resolution.x * 2.0 - 1.0,
            1.0 - y_px / params.resolution.y * 2.0,
            0.0,
            1.0,
        );
    } else {
        out.position = vec4(2.0, 2.0, 0.0, 1.0);
    }
    out.uv = corner;
    out.energy = peak;
    out.bin_t = f32(bin) / f32(params.bin_count - 1u);
    out.response = auxiliary[SMOOTHING_OFFSET + bin].z;
    let depth_brightness =
        clamp(render_channel.peak_depth_brightness_percent * 0.01, 0.0, 1.0);
    let rear_layer_alpha =
        depth_brightness * (0.33 + 0.67 * layer_t);
    out.reflection_alpha = select(rear_layer_alpha, 1.0, final_layer);
    out.transient = 0.0;
    return out;
}

@fragment
fn peak_fs(in: BarOut) -> @location(0) vec4<f32> {
    let low = vec3(0.08, 0.40, 1.0);
    let middle = vec3(0.02, 0.95, 0.78);
    let high = vec3(1.0, 0.20, 0.58);
    let color = mix(mix(low, middle, min(in.bin_t * 2.0, 1.0)), high, max(in.bin_t * 2.0 - 1.0, 0.0));
    let onset = in.response * params.frequency_range.w;
    let hot_color = mix(color, vec3(1.0), clamp(onset * 0.18, 0.0, 0.16));
    return vec4(
        hot_color * (1.05 + 0.55 * in.energy + onset * 2.0) * in.reflection_alpha,
        0.98 * in.reflection_alpha,
    );
}

@vertex
fn peak_glow_vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> BarOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let corner = corners[vertex];
    let bin = spectrum_bin(instance);
    let pitch = channel_pitch();
    let core_width = max(pitch - params.frequency_range.z, 0.5);
    let response = auxiliary[SMOOTHING_OFFSET + bin].z;
    let onset = response * params.frequency_range.w;
    let width = core_width + 8.0 + onset * 16.0;
    let height = 6.0 + onset * 10.0;
    let shift = prism_shift(bin);
    let x_px = bar_center_x(bin) - width * 0.5 + corner.x * width + shift.x;
    let peak = spectrum[bin].z;
    let center_y = peak_center_y(peak, instance);
    let y_px = center_y + (corner.y - 0.5) * height + shift.y;

    var out: BarOut;
    out.position = vec4(x_px / params.resolution.x * 2.0 - 1.0, 1.0 - y_px / params.resolution.y * 2.0, 0.0, 1.0);
    out.uv = corner;
    out.energy = peak;
    out.bin_t = f32(bin) / f32(params.bin_count - 1u);
    out.response = response;
    out.reflection_alpha = reflection_alpha(peak, instance);
    out.transient = 0.0;
    return out;
}

@fragment
fn peak_glow_fs(in: BarOut) -> @location(0) vec4<f32> {
    let low = vec3(0.08, 0.40, 1.0);
    let middle = vec3(0.02, 0.95, 0.78);
    let high = vec3(1.0, 0.20, 0.58);
    let frequency_color = mix(mix(low, middle, min(in.bin_t * 2.0, 1.0)), high, max(in.bin_t * 2.0 - 1.0, 0.0));
    let centered = abs(in.uv * 2.0 - vec2(1.0));
    let halo = pow(max(1.0 - length(centered), 0.0), 2.0);
    let onset = in.response * params.frequency_range.w;
    let strength = halo * onset * 0.75 * in.reflection_alpha;
    let color = mix(frequency_color, vec3(1.0), clamp(onset * 0.12, 0.0, 0.10));
    return vec4(color * strength, strength);
}
