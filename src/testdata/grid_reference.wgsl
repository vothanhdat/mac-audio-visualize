// Pre-optimization fragment entry point, for pixel-equivalence tests.
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
