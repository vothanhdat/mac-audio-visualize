struct Params {
    audio: vec4<f32>,
    resolution: vec2<f32>,
    axis_focus: vec2<f32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> notes: array<vec4<f32>, 13>;
@group(0) @binding(2) var labels: texture_2d<f32>;
@group(0) @binding(3) var label_sampler: sampler;
@group(0) @binding(4) var<uniform> appearance: vec4<f32>;

struct RibbonOut {
    @builtin(position) position: vec4<f32>,
    @location(0) pixel: vec2<f32>,
}

fn ribbon_width() -> f32 { return params.resolution.x * clamp(appearance.x, 0.2, 1.0); }
fn label_y() -> f32 { return min(clamp(params.resolution.y * 0.12, 38.0, 85.0), params.resolution.y * 0.35); }
fn font_size() -> f32 { return min(clamp(params.resolution.x * 0.016, 12.0, 28.0), ribbon_width() / 12.0 * 0.6); }

@vertex
fn ribbon_vs(@builtin(vertex_index) vertex: u32) -> RibbonOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let half_height = font_size() * 1.4 + 16.0;
    let x = params.resolution.x * 0.5 + (corners[vertex].x - 0.5) * (ribbon_width() + 40.0);
    let y = label_y() + (corners[vertex].y * 2.0 - 1.0) * half_height;
    var out: RibbonOut;
    out.position = vec4(x / params.resolution.x * 2.0 - 1.0,
        1.0 - y / params.resolution.y * 2.0, 0.0, 1.0);
    out.pixel = vec2(x, y);
    return out;
}

// A continuous warm-to-cool gradient, independent of the spectrum's palette.
fn note_color(t: f32) -> vec3<f32> {
    let colors = array<vec3<f32>, 7>(
        vec3(1.0, 0.24, 0.12), vec3(1.0, 0.66, 0.08),
        vec3(0.72, 0.91, 0.16), vec3(0.06, 0.80, 0.52),
        vec3(0.05, 0.63, 1.0), vec3(0.42, 0.22, 1.0), vec3(0.95, 0.12, 0.68),
    );
    let p = clamp(t, 0.0, 1.0) * 6.0;
    let i = min(u32(p), 5u);
    return mix(colors[i], colors[i + 1u], p - f32(i));
}

@fragment
fn ribbon_fs(in: RibbonOut) -> @location(0) vec4<f32> {
    let width = ribbon_width();
    let pitch = width / 12.0;
    let start = (params.resolution.x - width) * 0.5;
    let t = (in.pixel.x - start) / width;
    let size = font_size();
    let line_y = label_y() + size * 0.95;
    let dy = in.pixel.y - line_y;
    let edge = smoothstep(0.0, 0.025, t) * (1.0 - smoothstep(0.975, 1.0, t));
    var energy = 0.0;
    var halo = vec3(0.0);
    var ink = vec3(0.0);
    var ink_alpha = 0.0;
    for (var note = 0u; note < 12u; note++) {
        let state = notes[note];
        let x = start + (f32(note) + 0.5) * pitch;
        let dx = (in.pixel.x - x) / max(pitch, 1.0);
        let mound = exp(-dx * dx * 12.0) * state.y;
        energy += mound;
        let color = note_color(f32(note) / 11.0);
        // Local glows belong to notes with energy; the quiet gradient remains thin.
        halo += color * mound * exp(-dy * dy / 42.0) * (0.20 + state.z * 0.45);
        let label_size = size * (1.0 + state.z * 0.35);
        // The atlas has twelve 96px cells, rasterized at a 64px font size.
        let cell = label_size * 1.5;
        let uv = (in.pixel - vec2(x, label_y())) / cell + vec2(0.5);
        let inside = all(uv >= vec2(0.0)) && all(uv <= vec2(1.0));
        let atlas_uv = vec2((f32(note) + clamp(uv.x, 0.0, 1.0)) / 12.0,
            clamp(uv.y, 0.0, 1.0));
        let text = textureSampleLevel(labels, label_sampler, atlas_uv, 0.0).r
            * select(0.0, 1.0, inside);
        let dim = mix(vec3(0.29, 0.37, 0.48), color * 0.85, state.y * 0.55);
        let text_color = mix(dim, vec3(0.97, 0.96, 1.0), state.z);
        ink += text_color * text;
        ink_alpha = max(ink_alpha, text);
        let label_halo = exp(-dx * dx * 18.0
            - pow((in.pixel.y - label_y()) / max(size, 1.0), 2.0) * 2.0);
        halo += color * label_halo * state.z * 0.12;
    }
    let line = exp(-dy * dy / 0.75) * edge;
    let light = note_color(t) * line * (0.13 + min(energy, 1.0) * 0.85) + halo * edge;
    let light_alpha = clamp(max(max(light.r, light.g), light.b), 0.0, 0.95);
    // Straight-alpha output; ALPHA_BLENDING produces premultiplied pixels on
    // transparent surfaces and works identically over the opaque spectrum.
    let alpha = ink_alpha + light_alpha * (1.0 - ink_alpha);
    let premultiplied = ink + light * (1.0 - ink_alpha);
    return vec4(premultiplied / max(alpha, 0.00001), alpha);
}
