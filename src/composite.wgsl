@group(0) @binding(0) var opaque_visual: texture_2d<f32>;

struct CompositeParams {
    transparent_background: f32,
    surface_mode: f32,
    surface_camera_height: f32,
    surface_camera_distance: f32,
    surface_camera_fov_degrees: f32,
    surface_camera_target_y: f32,
    surface_reflection_strength_percent: f32,
    surface_reflection_fade_percent: f32,
    surface_history_enabled: f32,
    surface_history_write_row: f32,
    surface_history_valid_rows: f32,
    surface_history_fade_percent: f32,
    surface_history_opacity_percent: f32,
    surface_history_grayscale: f32,
    surface_history_brightness_percent: f32,
    surface_history_contrast_percent: f32,
    peak_trail_enabled: f32,
    peak_trail_delta_seconds: f32,
    peak_trail_fade_seconds: f32,
    peak_trail_movement_percent: f32,
    peak_trail_opacity_percent: f32,
    peak_trail_history_rows: f32,
    peak_trail_capture_current: f32,
    _peak_trail_padding_2: f32,
}

@group(0) @binding(1) var<uniform> params: CompositeParams;
@group(0) @binding(2) var spectrum_history: texture_2d<f32>;
@group(0) @binding(3) var previous_peak_trail: texture_2d<f32>;
@group(0) @binding(4) var peak_trail_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> VertexOut {
    let positions = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0),
    );
    var out: VertexOut;
    out.position = vec4(positions[vertex], 0.0, 1.0);
    return out;
}

fn background() -> vec4<f32> {
    if (params.transparent_background >= 0.5) {
        return vec4(0.0);
    }
    return vec4(0.0, 0.0, 0.0, 1.0);
}

fn load_visual(uv: vec2<f32>) -> vec3<f32> {
    let dimensions = textureDimensions(opaque_visual);
    let pixel = min(vec2<u32>(uv * vec2<f32>(dimensions)), dimensions - vec2(1u));
    return max(textureLoad(opaque_visual, pixel, 0).rgb, vec3(0.0));
}

fn load_history(uv: vec2<f32>) -> vec3<f32> {
    if (params.surface_history_enabled < 0.5 || params.surface_history_valid_rows < 0.5) {
        return vec3(0.0);
    }

    let dimensions = textureDimensions(spectrum_history);
    let row_count = dimensions.y;
    let age_progress = clamp(1.0 - uv.y, 0.0, 1.0);
    let row_position = age_progress * f32(row_count - 1u);
    let row_age = min(u32(floor(row_position)), row_count - 1u);
    let valid_rows = min(u32(params.surface_history_valid_rows + 0.5), row_count);
    if (row_age >= valid_rows) {
        return vec3(0.0);
    }

    let next_write_row = min(u32(params.surface_history_write_row + 0.5), row_count - 1u);
    let physical_row = (next_write_row + row_count - 1u - row_age) % row_count;
    let pixel_x = min(u32(clamp(uv.x, 0.0, 1.0) * f32(dimensions.x)), dimensions.x - 1u);
    var stored = max(
        textureLoad(spectrum_history, vec2(pixel_x, physical_row), 0).rgb,
        vec3(0.0),
    );
    let next_row_age = min(row_age + 1u, row_count - 1u);
    if (next_row_age < valid_rows && next_row_age != row_age) {
        let next_physical_row =
            (next_write_row + row_count - 1u - next_row_age) % row_count;
        let next_stored = max(
            textureLoad(spectrum_history, vec2(pixel_x, next_physical_row), 0).rgb,
            vec3(0.0),
        );
        stored = mix(stored, next_stored, fract(row_position));
    }
    if (params.surface_history_grayscale >= 0.5) {
        let luminance = dot(stored, vec3(0.2126, 0.7152, 0.0722));
        stored = vec3(luminance);
    }
    let contrast = clamp(params.surface_history_contrast_percent * 0.01, 0.25, 3.0);
    let brightness = clamp(params.surface_history_brightness_percent * 0.01, 0.0, 3.0);
    stored = max((stored - vec3(0.5)) * contrast + vec3(0.5), vec3(0.0));
    stored *= brightness;
    let fade_strength =
        clamp(params.surface_history_fade_percent * 0.01, 0.0, 2.0);
    let attenuation = exp2(-6.643856 * fade_strength * age_progress);
    let opacity = clamp(params.surface_history_opacity_percent * 0.01, 0.0, 1.0);
    return stored * attenuation * opacity;
}

fn camera_trail_source_uv(screen_uv: vec2<f32>) -> vec2<f32> {
    let dimensions = textureDimensions(previous_peak_trail);
    let camera = vec3(
        0.0,
        clamp(params.surface_camera_height, 0.25, 3.0),
        clamp(params.surface_camera_distance, 1.25, 8.0),
    );
    let camera_target = vec3(
        0.0,
        clamp(params.surface_camera_target_y, 0.0, 1.0),
        0.0,
    );
    let forward = normalize(camera_target - camera);
    let right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    let camera_up = normalize(cross(right, forward));
    let aspect = f32(dimensions.x) / max(f32(dimensions.y), 1.0);
    let fov_radians =
        clamp(params.surface_camera_fov_degrees, 20.0, 100.0) *
        0.017453292519943295;
    let tan_half_fov = tan(0.5 * fov_radians);
    let ndc = vec2(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0);
    let ray = normalize(
        forward +
        right * (ndc.x * aspect * tan_half_fov) +
        camera_up * (ndc.y * tan_half_fov),
    );
    let bottom_ray = normalize(forward - camera_up * tan_half_fov);
    let bottom_t = -camera.y / min(bottom_ray.y, -0.0001);
    let history_length =
        max(camera.z + bottom_ray.z * max(bottom_t, 0.0), 0.0001);
    let movement = clamp(params.peak_trail_movement_percent * 0.01, 0.0, 3.0);
    let history_rows = max(params.peak_trail_history_rows - 1.0, 1.0);
    let destination_z = min(history_length * movement / history_rows, camera.z - 0.001);
    let destination_t = (destination_z - camera.z) / min(ray.z, -0.0001);
    let destination = camera + ray * destination_t;
    let previous_world = vec3(destination.x, destination.y, 0.0);
    let relative = previous_world - camera;
    let camera_depth = dot(relative, forward);
    if (camera_depth <= 0.0001) {
        return vec2(-1.0);
    }
    let source_ndc = vec2(
        dot(relative, right) / (camera_depth * aspect * tan_half_fov),
        dot(relative, camera_up) / (camera_depth * tan_half_fov),
    );
    return vec2(source_ndc.x * 0.5 + 0.5, 0.5 - source_ndc.y * 0.5);
}

fn load_moving_peak_trail(screen_uv: vec2<f32>) -> vec4<f32> {
    if (params.peak_trail_enabled < 0.5 || params.surface_mode < 0.5) {
        return vec4(0.0);
    }
    let dt = clamp(params.peak_trail_delta_seconds, 0.0, 0.1);
    let source_uv = camera_trail_source_uv(screen_uv);
    if (any(source_uv < vec2(0.0)) || any(source_uv > vec2(1.0))) {
        return vec4(0.0);
    }
    let retained = exp(-dt / max(params.peak_trail_fade_seconds, 0.001));
    let opacity = clamp(params.peak_trail_opacity_percent * 0.01, 0.0, 2.0);
    // The feedback texture is premultiplied; opacity scales its emitted light
    // and coverage together. Re-clamp after a >100% visibility boost.
    let scaled =
        max(
            textureSampleLevel(previous_peak_trail, peak_trail_sampler, source_uv, 0.0),
            vec4(0.0),
        ) * retained * opacity;
    let alpha = clamp(scaled.a, 0.0, 1.0);
    if (alpha < (1.0 / 255.0)) {
        return vec4(0.0);
    }
    return vec4(min(scaled.rgb, vec3(alpha)), alpha);
}

fn compose(color: vec3<f32>, opacity: f32) -> vec4<f32> {
    let clamped_opacity = clamp(opacity, 0.0, 1.0);
    if (params.transparent_background < 0.5) {
        return vec4(color * clamped_opacity, 1.0);
    }

    // The visual target contains emitted light over black. Convert that light
    // into premultiplied window color and opacity for the macOS compositor.
    let light_alpha = clamp(max(color.r, max(color.g, color.b)), 0.0, 1.0);
    let alpha = light_alpha * clamped_opacity;
    return vec4(min(color, vec3(light_alpha)) * clamped_opacity, alpha);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(opaque_visual);
    let screen_uv = position.xy / vec2<f32>(dimensions);

    if (params.surface_mode < 0.5) {
        return compose(load_visual(screen_uv), 1.0);
    }

    let moving_peak_trail = load_moving_peak_trail(screen_uv);

    // One real pinhole camera is shared by the upright spectrum at Z=0 and
    // the history floor at Y=0. The camera looks at the spectrum's lower
    // center so their intersection is a stable X-axis hinge.
    let camera = vec3(
        0.0,
        clamp(params.surface_camera_height, 0.25, 3.0),
        clamp(params.surface_camera_distance, 1.25, 8.0),
    );
    let camera_target = vec3(
        0.0,
        clamp(params.surface_camera_target_y, 0.0, 1.0),
        0.0,
    );
    let forward = normalize(camera_target - camera);
    let right = normalize(cross(forward, vec3(0.0, 1.0, 0.0)));
    let camera_up = normalize(cross(right, forward));
    let aspect = f32(dimensions.x) / max(f32(dimensions.y), 1.0);
    let fov_radians =
        clamp(params.surface_camera_fov_degrees, 20.0, 100.0) *
        0.017453292519943295;
    let tan_half_fov = tan(0.5 * fov_radians);
    let ndc = vec2(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0);
    let ray = normalize(
        forward +
        right * (ndc.x * aspect * tan_half_fov) +
        camera_up * (ndc.y * tan_half_fov),
    );
    let target_distance = length(camera_target - camera);
    let surface_half_width =
        max(target_distance * tan_half_fov * aspect, 0.0001);

    // Intersect the live X-Y plane.
    let live_t = -camera.z / min(ray.z, -0.0001);
    let live_point = camera + ray * live_t;
    let live_source_x = live_point.x / (2.0 * surface_half_width) + 0.5;
    if (
        live_t > 0.0 &&
        live_point.y >= 0.0 && live_point.y <= 1.0 &&
        live_source_x >= 0.0 && live_source_x <= 1.0
    ) {
        return compose(
            load_visual(vec2(live_source_x, 1.0 - live_point.y)) + moving_peak_trail.rgb,
            1.0,
        );
    }

    var foreground = moving_peak_trail.rgb;

    // Intersect the X-Z history floor. Its far end is the point reached by the
    // bottom-center camera ray, so the complete circular history occupies the
    // visible floor without an independent screen-space skew.
    let bottom_ray = normalize(forward - camera_up * tan_half_fov);
    let bottom_t = -camera.y / min(bottom_ray.y, -0.0001);
    let history_length =
        max(camera.z + bottom_ray.z * max(bottom_t, 0.0), 0.0001);
    if (ray.y < -0.0001) {
        let floor_t = -camera.y / ray.y;
        let floor_point = camera + ray * floor_t;
        let floor_source_x =
            floor_point.x / (2.0 * surface_half_width) + 0.5;
        if (
            params.surface_history_enabled >= 0.5 &&
            floor_t > 0.0 &&
            floor_point.z >= 0.0 && floor_point.z <= history_length &&
            floor_source_x >= 0.0 && floor_source_x <= 1.0
        ) {
            let history_depth =
                clamp(floor_point.z / history_length, 0.0, 1.0);
            foreground += load_history(
                vec2(floor_source_x, 1.0 - history_depth),
            );
        }
    }

    // Mirror the live plane below Y=0. This is geometrically aligned with the
    // same hinge and camera, then attenuated independently before addition.
    if (
        live_t > 0.0 &&
        live_point.y < 0.0 && live_point.y >= -1.0 &&
        live_source_x >= 0.0 && live_source_x <= 1.0
    ) {
        let reflection_progress = -live_point.y;
        let strength =
            clamp(params.surface_reflection_strength_percent * 0.01, 0.0, 1.0);
        let fade_strength =
            clamp(params.surface_reflection_fade_percent * 0.01, 0.0, 20.0);
        let attenuation =
            strength * exp2(-6.643856 * fade_strength * reflection_progress);
        foreground += load_visual(
            vec2(live_source_x, 1.0 + live_point.y),
        ) * attenuation;
    }

    if (max(foreground.r, max(foreground.g, foreground.b)) > 0.0) {
        return compose(foreground, 1.0);
    }
    return background();
}
