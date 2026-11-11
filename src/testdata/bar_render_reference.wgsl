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
    let response = auxiliary[HIGHLIGHT_OFFSET + bin].w;
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

fn fs_main(in: BarOut) -> @location(0) vec4<f32> {
    return shaded_bar(in, 0.0);
}

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
    out.response = auxiliary[HIGHLIGHT_OFFSET + bin].w;
    out.reflection_alpha = reflection_alpha(corner.y, instance);
    out.transient = out.response;
    return out;
}

fn glow_fs(in: BarOut) -> @location(0) vec4<f32> {
    // Both renderers share the current palette; this compares bar geometry
    // and passes, not palette arithmetic.
    let frequency_color = palette_color(in.bin_t);
    let horizontal = pow(max(1.0 - abs(in.uv.x * 2.0 - 1.0), 0.0), 3.0);
    let onset = in.transient * params.frequency_range.w;
    let strength = horizontal * smoothstep(0.45, 1.0, in.energy) *
        (0.07 + 0.09 * in.uv.y) * (1.0 + onset * 1.8);
    return vec4(
        frequency_color * strength * in.reflection_alpha,
        strength * in.reflection_alpha,
    );
}

fn peak_glow_vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> BarOut {
    let corners = array<vec2<f32>, 6>(
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    );
    let corner = corners[vertex];
    let bin = spectrum_bin(instance);
    let pitch = channel_pitch();
    let core_width = max(pitch - params.frequency_range.z, 0.5);
    let response = auxiliary[HIGHLIGHT_OFFSET + bin].z;
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
