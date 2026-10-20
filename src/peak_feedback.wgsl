@group(0) @binding(0) var previous_trail: texture_2d<f32>;
@group(0) @binding(1) var current_peak: texture_2d<f32>;

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

@group(0) @binding(2) var<uniform> params: CompositeParams;
@group(0) @binding(3) var peak_trail_sampler: sampler;

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

fn load_texture(texture: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    if (any(uv < vec2(0.0)) || any(uv > vec2(1.0))) {
        return vec4(0.0);
    }
    return max(textureSampleLevel(texture, peak_trail_sampler, uv, 0.0), vec4(0.0));
}

fn transformed_previous(screen_uv: vec2<f32>) -> vec4<f32> {
    let dt = clamp(params.peak_trail_delta_seconds, 0.0, 0.1);
    let dimensions = textureDimensions(previous_trail);
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
        return vec4(0.0);
    }
    let source_ndc = vec2(
        dot(relative, right) / (camera_depth * aspect * tan_half_fov),
        dot(relative, camera_up) / (camera_depth * tan_half_fov),
    );
    let source_uv = vec2(source_ndc.x * 0.5 + 0.5, 0.5 - source_ndc.y * 0.5);
    let retained = exp(-dt / max(params.peak_trail_fade_seconds, 0.001));
    return load_texture(previous_trail, source_uv) * retained;
}

fn projected_current_peak(screen_uv: vec2<f32>) -> vec4<f32> {
    let dimensions = textureDimensions(current_peak);
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
    let live_t = -camera.z / min(ray.z, -0.0001);
    let live_point = camera + ray * live_t;
    let target_distance = length(camera_target - camera);
    let surface_half_width = max(target_distance * tan_half_fov * aspect, 0.0001);
    let source_x = live_point.x / (2.0 * surface_half_width) + 0.5;
    if (
        live_t <= 0.0 || live_point.y < 0.0 || live_point.y > 1.0 ||
        source_x < 0.0 || source_x > 1.0
    ) {
        return vec4(0.0);
    }
    return load_texture(current_peak, vec2(source_x, 1.0 - live_point.y));
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(previous_trail);
    let screen_uv = position.xy / vec2<f32>(dimensions);
    let old = transformed_previous(screen_uv);
    var result = old;
    if (params.peak_trail_capture_current >= 0.5) {
        let current = projected_current_peak(screen_uv);
        // A temporal light trail should not accumulate opaque coverage like
        // stacked translucent sheets. Keep the strongest emitted light and
        // coverage at each pixel instead.
        result = max(current, old);
    }
    let alpha = clamp(result.a, 0.0, 1.0);
    if (alpha < (1.0 / 255.0)) {
        return vec4(0.0);
    }
    // Keep a strict premultiplied representation so dim alpha cannot carry a
    // bright RGB residue that later becomes full-screen fog.
    return vec4(min(max(result.rgb, vec3(0.0)), vec3(alpha)), alpha);
}
