use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use egui_wgpu::{Renderer as EguiRenderer, RendererOptions, ScreenDescriptor};
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{
    hybrid_matcher::{HybridBandMatcher, HybridMatchConfig},
    multirate_fft::{MultirateConfig, MultirateFft},
    resonator_resampler::ResonatorResampler,
};
use visual_audio::audio::{CaptureStall, SampleRing};

#[path = "note_ribbon.rs"]
mod note_ribbon;
use note_ribbon::NoteRibbon;

#[path = "static_grid.rs"]
mod static_grid;
use static_grid::StaticGrid;

// Audio block used for streaming analyzer warm-up and output-delay alignment.
pub const SAMPLE_COUNT: usize = 16_384;
const HISTORY_SAMPLE_COUNT: usize = 65_536;
const MAX_BIN_COUNT: u32 = 2048;
const GAMMATONE_WARMUP_SAMPLES: usize = 4096;
const PLOT_HORIZONTAL_MARGIN: f32 = 0.0;
const PLOT_BOTTOM_PADDING: u32 = 32;
const MIN_SURFACE_HISTORY_ROWS: u32 = 64;
const MAX_SURFACE_HISTORY_ROWS: u32 = 2048;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AnalysisParams {
    sample_rate: f32,
    elapsed: f32,
    gain: f32,
    bin_count: u32,
    resolution: [f32; 2],
    axis_focus: [f32; 2],
    perception: [f32; 4],
    visual: [f32; 4],
    capture: [u32; 4],
    db_scale: [f32; 4],
    frequency_range: [f32; 4],
    auditory_0: [f32; 4],
    auditory_1: [f32; 4],
    fft_sizes_0: [u32; 4],
    fft_sizes_1: [u32; 4],
    fft_decimations_0: [u32; 4],
    fft_decimations_1: [u32; 4],
    fft_crossovers: [f32; 4],
    resonator: [f32; 4],
    resonator_decay: [f32; 4],
    resonator_stream: [u32; 4],
    hybrid: [f32; 4],
    resonator_stream_extra: [u32; 4],
    hybrid_match_corrections: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
struct VisualizerSettings {
    gain: f32,
    channel_mode: ChannelMode,
    stereo_treble_center: bool,
    visualization_delay_ms: f32,
    auto_output_sync: bool,
    min_brightness: f32,
    max_brightness: f32,
    rise_highlight: bool,
    rise_highlight_strength: f32,
    rise_highlight_fade_ms: f32,
    peak_glow_full_scale_db: f32,
    bar_gravity_db_per_second2: f32,
    peak_hold_ms: f32,
    peak_gravity_db_per_second2: f32,
    peak_momentum: bool,
    analysis_method: AnalysisMethod,
    asymmetric_kaiser_beta: f32,
    flanagan_alpha: f32,
    multirate_fft_window: AnalysisWindow,
    multirate_fft_sizes: [u32; 3],
    multirate_fft_decimations: [u32; 3],
    multirate_fft_crossovers_hz: [f32; 2],
    multirate_fft5_window: AnalysisWindow,
    multirate_fft5_sizes: [u32; 5],
    multirate_fft5_decimations: [u32; 5],
    multirate_fft5_crossovers_hz: [f32; 4],
    auditory_erb_bandwidth_scale: f32,
    auditory_compression_percent: f32,
    auditory_filter_order: u32,
    auditory_phase_sharpening_percent: f32,
    auditory_phase_sharpening_10_khz_percent: Option<f32>,
    auditory_tonal_width_percent: f32,
    auditory_tonal_width_10_khz_percent: Option<f32>,
    resonator_q_50_hz: f32,
    resonator_q_10_khz: f32,
    resonator_measurement: ResonatorMeasurement,
    resonator_envelope: ResonatorEnvelope,
    resonator_decay_db: f32,
    resonator_decay_seconds: f32,
    resonator_multirate: bool,
    resonator_five_stage: bool,
    resonator_timestamp_alignment: bool,
    hybrid_fft_window: AnalysisWindow,
    #[serde(alias = "hybrid_fft_size")]
    hybrid_half_fft_size: u32,
    hybrid_full_fft_size: u32,
    fft_display_divisor: u32,
    hybrid_adaptive_matching: bool,
    hybrid_match_adapt_seconds: f32,
    hybrid_match_max_correction_db: f32,
    render_mode: RenderMode,
    surface_camera_height: f32,
    surface_camera_distance: f32,
    surface_camera_fov_degrees: f32,
    surface_camera_target_y: f32,
    #[serde(default)]
    surface_bar_depth_px: Option<f32>,
    // Retained only to migrate settings saved before fixed-pixel depth.
    surface_bar_depth_percent: f32,
    surface_peak_depth_brightness_percent: f32,
    surface_volume_stage_count: u32,
    surface_reflection_strength_percent: f32,
    surface_reflection_fade_percent: f32,
    surface_history_enabled: bool,
    surface_history_rows: u32,
    surface_history_fade_percent: f32,
    surface_history_opacity_percent: f32,
    surface_history_grayscale: bool,
    surface_history_brightness_percent: f32,
    surface_history_contrast_percent: f32,
    surface_peak_trail_enabled: bool,
    surface_peak_trail_fade_seconds: f32,
    surface_peak_trail_movement_percent: f32,
    surface_peak_trail_opacity_percent: f32,
    surface_peak_trail_capture_interval_frames: u32,
    axis_labels_overlay: bool,
    axis_label_mode: AxisLabelMode,
    show_frequency_grid: bool,
    show_note_ribbon: bool,
    note_ribbon_width_percent: f32,
    note_midrange_only: bool,
    note_fast_response: bool,
    note_smoothing_ms: f32,
    bottom_mirror: bool,
    mirror_baseline_percent: f32,
    mirror_initial_fade_percent: f32,
    mirror_fade_percent: f32,
    bar_layout: BarLayout,
    bar_width_px: f32,
    bar_spacing_px: f32,
    low_db_floor: f32,
    high_db_floor: f32,
    bar_top_fade_percent: f32,
    min_frequency_hz: f32,
    max_frequency_hz: f32,
    frequency_focus_strength_percent: f32,
    frequency_focus_origin_percent: f32,
    transparent_background: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum AnalysisMethod {
    MultirateFft,
    MultirateFft5,
    #[serde(alias = "WindowedDft")]
    ResonatorBank,
    HybridResonatorFft,
    AuditoryErb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ChannelMode {
    Mono,
    Left,
    Right,
    StereoMirror,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum RenderMode {
    Flat,
    Surface3d,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum BarLayout {
    FixedPixels,
    MusicalNotes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum AxisLabelMode {
    Notes,
    Hertz,
}

impl AxisLabelMode {
    fn label(self) -> &'static str {
        match self {
            Self::Notes => "Musical notes",
            Self::Hertz => "Hz",
        }
    }

    fn shader_value(self) -> f32 {
        match self {
            Self::Notes => 0.0,
            Self::Hertz => 1.0,
        }
    }
}

impl RenderMode {
    fn label(self) -> &'static str {
        match self {
            Self::Flat => "Flat",
            Self::Surface3d => "3D surface",
        }
    }
}

impl BarLayout {
    fn label(self) -> &'static str {
        match self {
            Self::FixedPixels => "Fixed pixels",
            Self::MusicalNotes => "One bar per note",
        }
    }
}

impl ChannelMode {
    fn label(self) -> &'static str {
        match self {
            Self::Mono => "Mono (L + R)",
            Self::Left => "Left",
            Self::Right => "Right",
            Self::StereoMirror => "Stereo mirror (L | R)",
        }
    }

    fn channel_count(self) -> usize {
        match self {
            Self::StereoMirror => 2,
            _ => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ResonatorEnvelope {
    DecayThenPeak,
    PeakThenDecay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ResonatorMeasurement {
    RawMagnitude,
    #[serde(alias = "PhaseTracked")]
    ImageRejected,
    PositionVelocity,
    PositionVelocityPi2,
    PositionPi2,
}

impl ResonatorMeasurement {
    fn label(self) -> &'static str {
        match self {
            Self::RawMagnitude => "Raw magnitude",
            Self::ImageRejected => "Phase-corrected magnitude",
            Self::PositionVelocity => "Position + velocity",
            Self::PositionVelocityPi2 => "Position + velocity, π/2 pair",
            Self::PositionPi2 => "Position only, π/2 pair",
        }
    }

    fn shader_value(self) -> f32 {
        match self {
            Self::RawMagnitude => 0.0,
            Self::ImageRejected => 1.0,
            Self::PositionVelocity => 2.0,
            Self::PositionVelocityPi2 => 3.0,
            Self::PositionPi2 => 4.0,
        }
    }
}

impl ResonatorEnvelope {
    fn label(self) -> &'static str {
        match self {
            Self::DecayThenPeak => "Decay then peak",
            Self::PeakThenDecay => "Peak then decay",
        }
    }

    fn shader_value(self) -> f32 {
        match self {
            Self::DecayThenPeak => 0.0,
            Self::PeakThenDecay => 1.0,
        }
    }
}

impl AnalysisMethod {
    fn label(self) -> &'static str {
        match self {
            Self::MultirateFft => "Multirate FFT (3-band)",
            Self::MultirateFft5 => "Multirate FFT (5-band)",
            Self::ResonatorBank => "Resonator bank",
            Self::HybridResonatorFft => "Hybrid resonator + FFT",
            Self::AuditoryErb => "Auditory ERB",
        }
    }

    fn is_multirate_fft(self) -> bool {
        matches!(
            self,
            Self::MultirateFft | Self::MultirateFft5 | Self::HybridResonatorFft
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum AnalysisWindow {
    Hann,
    HalfHann,
    HalfBlackman,
    AsymmetricKaiser8,
    Flanagan,
}

impl AnalysisWindow {
    fn label(self) -> &'static str {
        match self {
            Self::Hann => "Hann",
            Self::HalfHann => "Half-Hann",
            Self::HalfBlackman => "Half-Blackman",
            Self::AsymmetricKaiser8 => "Asymmetric Kaiser",
            Self::Flanagan => "Flanagan",
        }
    }

    fn shader_value(self) -> f32 {
        match self {
            Self::Hann => 0.0,
            Self::HalfHann => 1.0,
            Self::HalfBlackman => 2.0,
            Self::AsymmetricKaiser8 => 3.0,
            Self::Flanagan => 4.0,
        }
    }
}

struct UiLayer {
    context: egui::Context,
    state: egui_winit::State,
    renderer: EguiRenderer,
    settings_tab: SettingsTab,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsTab {
    Analyzer,
    Visual,
}

struct UiFrame {
    paint_jobs: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
    screen: ScreenDescriptor,
}

impl Default for VisualizerSettings {
    fn default() -> Self {
        Self {
            gain: 1.7,
            channel_mode: ChannelMode::Mono,
            stereo_treble_center: false,
            visualization_delay_ms: 0.0,
            auto_output_sync: true,
            min_brightness: 0.02,
            max_brightness: 5.3,
            rise_highlight: true,
            rise_highlight_strength: 0.35,
            rise_highlight_fade_ms: 160.0,
            peak_glow_full_scale_db: 6.0,
            bar_gravity_db_per_second2: 5000.0,
            peak_hold_ms: 120.0,
            peak_gravity_db_per_second2: 45.0,
            peak_momentum: true,
            analysis_method: AnalysisMethod::HybridResonatorFft,
            asymmetric_kaiser_beta: 8.1,
            flanagan_alpha: 8.0,
            multirate_fft_window: AnalysisWindow::HalfBlackman,
            multirate_fft_sizes: [128, 512, 2048],
            multirate_fft_decimations: [64, 8, 1],
            multirate_fft_crossovers_hz: [230.0, 2200.0],
            multirate_fft5_window: AnalysisWindow::AsymmetricKaiser8,
            multirate_fft5_sizes: [128, 256, 512, 1024, 2048],
            multirate_fft5_decimations: [128, 64, 16, 4, 1],
            multirate_fft5_crossovers_hz: [80.0, 320.0, 1280.0, 5120.0],
            auditory_erb_bandwidth_scale: 0.5,
            auditory_compression_percent: 0.0,
            auditory_filter_order: 4,
            auditory_phase_sharpening_percent: 100.0,
            auditory_phase_sharpening_10_khz_percent: Some(100.0),
            auditory_tonal_width_percent: 85.0,
            auditory_tonal_width_10_khz_percent: Some(8.0),
            resonator_q_50_hz: 10.0,
            resonator_q_10_khz: 200.0,
            resonator_measurement: ResonatorMeasurement::ImageRejected,
            resonator_envelope: ResonatorEnvelope::PeakThenDecay,
            resonator_decay_db: 60.0,
            resonator_decay_seconds: 0.3,
            resonator_multirate: true,
            resonator_five_stage: true,
            resonator_timestamp_alignment: true,
            hybrid_fft_window: AnalysisWindow::AsymmetricKaiser8,
            hybrid_half_fft_size: 1024,
            hybrid_full_fft_size: 512,
            fft_display_divisor: 1,
            hybrid_adaptive_matching: true,
            hybrid_match_adapt_seconds: 1.0,
            hybrid_match_max_correction_db: 12.0,
            render_mode: RenderMode::Flat,
            surface_camera_height: 1.2,
            surface_camera_distance: 3.0,
            surface_camera_fov_degrees: 50.0,
            surface_camera_target_y: 0.45,
            surface_bar_depth_px: Some(1.0),
            surface_bar_depth_percent: 100.0,
            surface_peak_depth_brightness_percent: 55.0,
            surface_volume_stage_count: 5,
            surface_reflection_strength_percent: 35.0,
            surface_reflection_fade_percent: 100.0,
            surface_history_enabled: false,
            surface_history_rows: 512,
            surface_history_fade_percent: 100.0,
            surface_history_opacity_percent: 100.0,
            surface_history_grayscale: false,
            surface_history_brightness_percent: 100.0,
            surface_history_contrast_percent: 100.0,
            surface_peak_trail_enabled: false,
            surface_peak_trail_fade_seconds: 0.75,
            surface_peak_trail_movement_percent: 100.0,
            surface_peak_trail_opacity_percent: 70.0,
            surface_peak_trail_capture_interval_frames: 1,
            axis_labels_overlay: false,
            axis_label_mode: AxisLabelMode::Hertz,
            show_frequency_grid: true,
            show_note_ribbon: true,
            note_ribbon_width_percent: 76.0,
            note_midrange_only: true,
            note_fast_response: true,
            note_smoothing_ms: 0.0,
            bottom_mirror: true,
            mirror_baseline_percent: 82.0,
            mirror_initial_fade_percent: 43.0,
            mirror_fade_percent: 2000.0,
            bar_layout: BarLayout::FixedPixels,
            bar_width_px: 3.0,
            bar_spacing_px: 0.0,
            low_db_floor: -51.0,
            high_db_floor: -62.0,
            bar_top_fade_percent: 45.0,
            min_frequency_hz: 16.0,
            max_frequency_hz: 22500.0,
            frequency_focus_strength_percent: 59.0,
            frequency_focus_origin_percent: 50.0,
            transparent_background: false,
        }
    }
}

impl VisualizerSettings {
    fn load(path: &Path) -> Self {
        let source = match fs::read_to_string(path) {
            Ok(source) => source,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(error) => {
                eprintln!("could not read settings from {}: {error}", path.display());
                return Self::default();
            }
        };
        match serde_json::from_str::<Self>(&source) {
            Ok(settings) => settings.sanitized(),
            Err(error) => {
                eprintln!("could not parse settings from {}: {error}", path.display());
                Self::default()
            }
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let parent = path.parent().context("settings path has no parent")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
        let temporary = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(self).context("could not serialize settings")?;
        fs::write(&temporary, json)
            .with_context(|| format!("could not write {}", temporary.display()))?;
        fs::rename(&temporary, path)
            .with_context(|| format!("could not replace {}", path.display()))?;
        Ok(())
    }

    fn sanitized(mut self) -> Self {
        let defaults = Self::default();
        self.fft_display_divisor = self.fft_display_divisor.clamp(1, 4);
        self.gain = finite_clamp(self.gain, defaults.gain, 0.1, 20.0);
        self.visualization_delay_ms = finite_clamp(
            self.visualization_delay_ms,
            defaults.visualization_delay_ms,
            -1000.0,
            1000.0,
        );
        self.min_brightness = finite_clamp(self.min_brightness, defaults.min_brightness, 0.02, 1.0);
        self.max_brightness = finite_clamp(self.max_brightness, defaults.max_brightness, 1.0, 8.0);
        self.rise_highlight_strength = finite_clamp(
            self.rise_highlight_strength,
            defaults.rise_highlight_strength,
            0.0,
            3.0,
        );
        self.rise_highlight_fade_ms = finite_clamp(
            self.rise_highlight_fade_ms,
            defaults.rise_highlight_fade_ms,
            20.0,
            1000.0,
        );
        self.peak_glow_full_scale_db = finite_clamp(
            self.peak_glow_full_scale_db,
            defaults.peak_glow_full_scale_db,
            0.5,
            30.0,
        );
        self.bar_gravity_db_per_second2 = finite_clamp(
            self.bar_gravity_db_per_second2,
            defaults.bar_gravity_db_per_second2,
            10.0,
            5000.0,
        );
        self.peak_hold_ms = finite_clamp(self.peak_hold_ms, defaults.peak_hold_ms, 0.0, 1000.0);
        self.peak_gravity_db_per_second2 = finite_clamp(
            self.peak_gravity_db_per_second2,
            defaults.peak_gravity_db_per_second2,
            5.0,
            5000.0,
        );
        self.bar_width_px = finite_clamp(self.bar_width_px, defaults.bar_width_px, 1.0, 16.0);
        self.bar_spacing_px = finite_clamp(self.bar_spacing_px, defaults.bar_spacing_px, 0.0, 12.0);
        self.note_ribbon_width_percent = finite_clamp(
            self.note_ribbon_width_percent,
            defaults.note_ribbon_width_percent,
            20.0,
            100.0,
        );
        self.note_smoothing_ms = finite_clamp(
            self.note_smoothing_ms,
            defaults.note_smoothing_ms,
            0.0,
            250.0,
        );
        self.mirror_baseline_percent = finite_clamp(
            self.mirror_baseline_percent,
            defaults.mirror_baseline_percent,
            0.0,
            100.0,
        );
        self.mirror_initial_fade_percent = finite_clamp(
            self.mirror_initial_fade_percent,
            defaults.mirror_initial_fade_percent,
            0.0,
            100.0,
        );
        self.mirror_fade_percent = finite_clamp(
            self.mirror_fade_percent,
            defaults.mirror_fade_percent,
            0.0,
            2000.0,
        );
        self.low_db_floor = finite_clamp(self.low_db_floor, defaults.low_db_floor, -120.0, -20.0);
        self.high_db_floor =
            finite_clamp(self.high_db_floor, defaults.high_db_floor, -120.0, -20.0);
        self.bar_top_fade_percent = finite_clamp(
            self.bar_top_fade_percent,
            defaults.bar_top_fade_percent,
            0.0,
            99.0,
        );
        self.min_frequency_hz = finite_clamp(
            self.min_frequency_hz,
            defaults.min_frequency_hz,
            16.0,
            200.0,
        );
        self.max_frequency_hz = finite_clamp(
            self.max_frequency_hz,
            defaults.max_frequency_hz,
            10_000.0,
            40_000.0,
        );
        self.frequency_focus_strength_percent = finite_clamp(
            self.frequency_focus_strength_percent,
            defaults.frequency_focus_strength_percent,
            0.0,
            100.0,
        );
        self.frequency_focus_origin_percent = finite_clamp(
            self.frequency_focus_origin_percent,
            defaults.frequency_focus_origin_percent,
            0.0,
            100.0,
        );
        self.asymmetric_kaiser_beta = finite_clamp(
            self.asymmetric_kaiser_beta,
            defaults.asymmetric_kaiser_beta,
            0.0,
            16.0,
        );
        self.flanagan_alpha = finite_clamp(self.flanagan_alpha, defaults.flanagan_alpha, 2.0, 24.0);
        for index in 0..3 {
            self.multirate_fft_decimations[index] = self.multirate_fft_decimations[index]
                .clamp(1, 128)
                .next_power_of_two();
            let maximum_size = (HISTORY_SAMPLE_COUNT as u32
                / self.multirate_fft_decimations[index])
                .clamp(32, 8192)
                .next_power_of_two()
                .min(8192);
            self.multirate_fft_sizes[index] = self.multirate_fft_sizes[index]
                .clamp(32, maximum_size)
                .next_power_of_two()
                .min(maximum_size);
        }
        self.multirate_fft_crossovers_hz[0] = finite_clamp(
            self.multirate_fft_crossovers_hz[0],
            defaults.multirate_fft_crossovers_hz[0],
            40.0,
            1000.0,
        );
        self.multirate_fft_crossovers_hz[1] = finite_clamp(
            self.multirate_fft_crossovers_hz[1],
            defaults.multirate_fft_crossovers_hz[1],
            (self.multirate_fft_crossovers_hz[0] * 1.2).max(500.0),
            10_000.0,
        );
        for index in 0..5 {
            self.multirate_fft5_decimations[index] = self.multirate_fft5_decimations[index]
                .clamp(1, 128)
                .next_power_of_two();
            let maximum_size = (HISTORY_SAMPLE_COUNT as u32
                / self.multirate_fft5_decimations[index])
                .clamp(32, 8192)
                .next_power_of_two()
                .min(8192);
            self.multirate_fft5_sizes[index] = self.multirate_fft5_sizes[index]
                .clamp(32, maximum_size)
                .next_power_of_two()
                .min(maximum_size);
        }
        let crossover_limits = [
            (40.0, 400.0),
            (100.0, 1200.0),
            (400.0, 4000.0),
            (1500.0, 10_000.0),
        ];
        for (index, &(base_minimum, maximum)) in crossover_limits.iter().enumerate() {
            let lower = if index == 0 {
                base_minimum
            } else {
                (self.multirate_fft5_crossovers_hz[index - 1] * 1.2).max(base_minimum)
            };
            self.multirate_fft5_crossovers_hz[index] = finite_clamp(
                self.multirate_fft5_crossovers_hz[index],
                defaults.multirate_fft5_crossovers_hz[index],
                lower,
                maximum,
            );
        }
        self.auditory_erb_bandwidth_scale = finite_clamp(
            self.auditory_erb_bandwidth_scale,
            defaults.auditory_erb_bandwidth_scale,
            0.1,
            10.0,
        );
        self.auditory_compression_percent = finite_clamp(
            self.auditory_compression_percent,
            defaults.auditory_compression_percent,
            0.0,
            100.0,
        );
        self.auditory_filter_order = self.auditory_filter_order.clamp(1, 4);
        self.auditory_phase_sharpening_percent = finite_clamp(
            self.auditory_phase_sharpening_percent,
            defaults.auditory_phase_sharpening_percent,
            0.0,
            100.0,
        );
        self.auditory_phase_sharpening_10_khz_percent = Some(finite_clamp(
            self.auditory_phase_sharpening_10_khz_percent
                .unwrap_or(self.auditory_phase_sharpening_percent),
            self.auditory_phase_sharpening_percent,
            0.0,
            100.0,
        ));
        self.auditory_tonal_width_percent = finite_clamp(
            self.auditory_tonal_width_percent,
            defaults.auditory_tonal_width_percent,
            5.0,
            100.0,
        );
        self.auditory_tonal_width_10_khz_percent = Some(finite_clamp(
            self.auditory_tonal_width_10_khz_percent
                .unwrap_or(self.auditory_tonal_width_percent),
            self.auditory_tonal_width_percent,
            5.0,
            100.0,
        ));
        self.resonator_q_50_hz = finite_clamp(
            self.resonator_q_50_hz,
            defaults.resonator_q_50_hz,
            0.5,
            32.0,
        );
        self.resonator_q_10_khz = finite_clamp(
            self.resonator_q_10_khz,
            defaults.resonator_q_10_khz,
            4.0,
            512.0,
        );
        self.resonator_decay_db = finite_clamp(
            self.resonator_decay_db,
            defaults.resonator_decay_db,
            1.0,
            60.0,
        );
        self.resonator_decay_seconds = finite_clamp(
            self.resonator_decay_seconds,
            defaults.resonator_decay_seconds,
            0.01,
            5.0,
        );
        self.hybrid_half_fft_size = self
            .hybrid_half_fft_size
            .clamp(128, 4096)
            .next_power_of_two();
        self.hybrid_full_fft_size = self
            .hybrid_full_fft_size
            .clamp(128, 4096)
            .next_power_of_two();
        self.hybrid_match_adapt_seconds = finite_clamp(
            self.hybrid_match_adapt_seconds,
            defaults.hybrid_match_adapt_seconds,
            0.01,
            5.0,
        );
        self.hybrid_match_max_correction_db = finite_clamp(
            self.hybrid_match_max_correction_db,
            defaults.hybrid_match_max_correction_db,
            0.0,
            20.0,
        );
        self.surface_camera_height = finite_clamp(
            self.surface_camera_height,
            defaults.surface_camera_height,
            0.25,
            3.0,
        );
        self.surface_camera_distance = finite_clamp(
            self.surface_camera_distance,
            defaults.surface_camera_distance,
            1.25,
            8.0,
        );
        self.surface_camera_fov_degrees = finite_clamp(
            self.surface_camera_fov_degrees,
            defaults.surface_camera_fov_degrees,
            20.0,
            100.0,
        );
        self.surface_camera_target_y = finite_clamp(
            self.surface_camera_target_y,
            defaults.surface_camera_target_y,
            0.0,
            1.0,
        );
        self.surface_bar_depth_percent = finite_clamp(
            self.surface_bar_depth_percent,
            defaults.surface_bar_depth_percent,
            0.0,
            300.0,
        );
        let migrated_depth_px =
            self.bar_width_px.max(1.0) * (self.surface_bar_depth_percent * 0.01).clamp(0.0, 3.0);
        self.surface_bar_depth_px = Some(match self.surface_bar_depth_px {
            Some(depth_px) => finite_clamp(depth_px, 6.0, 0.0, 300.0),
            None => migrated_depth_px,
        });
        self.surface_peak_depth_brightness_percent = finite_clamp(
            self.surface_peak_depth_brightness_percent,
            defaults.surface_peak_depth_brightness_percent,
            0.0,
            100.0,
        );
        self.surface_volume_stage_count = self.surface_volume_stage_count.clamp(3, 8);
        self.surface_reflection_strength_percent = finite_clamp(
            self.surface_reflection_strength_percent,
            defaults.surface_reflection_strength_percent,
            0.0,
            100.0,
        );
        self.surface_reflection_fade_percent = finite_clamp(
            self.surface_reflection_fade_percent,
            defaults.surface_reflection_fade_percent,
            0.0,
            2000.0,
        );
        self.surface_history_rows = self
            .surface_history_rows
            .clamp(MIN_SURFACE_HISTORY_ROWS, MAX_SURFACE_HISTORY_ROWS);
        self.surface_history_fade_percent = finite_clamp(
            self.surface_history_fade_percent,
            defaults.surface_history_fade_percent,
            0.0,
            200.0,
        );
        self.surface_history_opacity_percent = finite_clamp(
            self.surface_history_opacity_percent,
            defaults.surface_history_opacity_percent,
            0.0,
            100.0,
        );
        self.surface_history_brightness_percent = finite_clamp(
            self.surface_history_brightness_percent,
            defaults.surface_history_brightness_percent,
            0.0,
            300.0,
        );
        self.surface_history_contrast_percent = finite_clamp(
            self.surface_history_contrast_percent,
            defaults.surface_history_contrast_percent,
            25.0,
            300.0,
        );
        self.surface_peak_trail_fade_seconds = finite_clamp(
            self.surface_peak_trail_fade_seconds,
            defaults.surface_peak_trail_fade_seconds,
            0.05,
            5.0,
        );
        self.surface_peak_trail_movement_percent = finite_clamp(
            self.surface_peak_trail_movement_percent,
            defaults.surface_peak_trail_movement_percent,
            0.0,
            300.0,
        );
        self.surface_peak_trail_opacity_percent = finite_clamp(
            self.surface_peak_trail_opacity_percent,
            defaults.surface_peak_trail_opacity_percent,
            0.0,
            200.0,
        );
        self.surface_peak_trail_capture_interval_frames =
            self.surface_peak_trail_capture_interval_frames.clamp(1, 10);
        self
    }
}

fn finite_clamp(value: f32, fallback: f32, minimum: f32, maximum: f32) -> f32 {
    if value.is_finite() {
        value.clamp(minimum, maximum)
    } else {
        fallback
    }
}

fn format_frequency(hz: f32) -> String {
    if hz >= 1000.0 {
        format!("{:.2} kHz", hz / 1000.0)
    } else {
        format!("{hz:.0} Hz")
    }
}

fn settings_path(profile: Option<&str>) -> Option<PathBuf> {
    let file_name = profile
        .map(|profile| format!("settings-{profile}.json"))
        .unwrap_or_else(|| "settings.json".to_owned());
    Some(
        PathBuf::from(std::env::var_os("HOME")?)
            .join("Library/Application Support/VisualAudio")
            .join(file_name),
    )
}

impl UiLayer {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, window: &Window) -> Self {
        let context = egui::Context::default();
        context.set_visuals(egui::Visuals::dark());
        let state = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = EguiRenderer::new(device, format, RendererOptions::default());
        Self {
            context,
            state,
            renderer,
            settings_tab: SettingsTab::Analyzer,
        }
    }

    fn on_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    fn begin_frame(
        &mut self,
        window: &Window,
        render_size: PhysicalSize<u32>,
        settings: &mut VisualizerSettings,
        sample_rate: f32,
        output_latency_ms: Option<f32>,
    ) -> UiFrame {
        let input = self.state.take_egui_input(window);
        let context = self.context.clone();
        let output = context.run(input, |context| {
            draw_settings_panel(
                context,
                settings,
                sample_rate,
                output_latency_ms,
                &mut self.settings_tab,
            );
        });
        self.state
            .handle_platform_output(window, output.platform_output);
        let pixels_per_point = output.pixels_per_point;
        UiFrame {
            paint_jobs: self.context.tessellate(output.shapes, pixels_per_point),
            textures_delta: output.textures_delta,
            screen: ScreenDescriptor {
                size_in_pixels: [render_size.width, render_size.height],
                pixels_per_point,
            },
        }
    }
}

fn visual_delay_ms(settings: &VisualizerSettings, output_latency_ms: Option<f32>) -> f32 {
    let automatic = if settings.auto_output_sync {
        output_latency_ms.unwrap_or(0.0)
    } else {
        0.0
    };
    (automatic + settings.visualization_delay_ms).clamp(0.0, 1000.0)
}

fn fft_cadence_control(ui: &mut egui::Ui, settings: &mut VisualizerSettings) {
    egui::ComboBox::from_label("FFT update rate")
        .selected_text(match settings.fft_display_divisor {
            2 => "½× display",
            3 => "⅓× display",
            4 => "¼× display",
            _ => "1× display",
        })
        .show_ui(ui, |ui| {
            for (divisor, label) in [
                (1, "1× display"),
                (2, "½× display"),
                (3, "⅓× display"),
                (4, "¼× display"),
            ] {
                ui.selectable_value(&mut settings.fft_display_divisor, divisor, label);
            }
        });
    ui.small("Updates FFT every selected number of visual frames; bars and peaks still animate every frame.");
}

fn draw_settings_panel(
    context: &egui::Context,
    settings: &mut VisualizerSettings,
    sample_rate: f32,
    output_latency_ms: Option<f32>,
    settings_tab: &mut SettingsTab,
) {
    egui::CentralPanel::default().show(context, |ui| {
        ui.horizontal(|ui| {
            ui.selectable_value(settings_tab, SettingsTab::Analyzer, "Analyzer");
            ui.selectable_value(settings_tab, SettingsTab::Visual, "UI");
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.label("Changes apply immediately on the GPU.");
                match *settings_tab {
                    SettingsTab::Analyzer => {
                ui.add(
                    egui::Slider::new(&mut settings.gain, 0.1..=20.0)
                        .logarithmic(true)
                        .text("Gain"),
                );
                egui::ComboBox::from_label("Channels")
                    .selected_text(settings.channel_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in [
                            ChannelMode::Mono,
                            ChannelMode::Left,
                            ChannelMode::Right,
                            ChannelMode::StereoMirror,
                        ] {
                            ui.selectable_value(&mut settings.channel_mode, mode, mode.label());
                        }
                });
                if settings.channel_mode == ChannelMode::StereoMirror {
                    ui.checkbox(
                        &mut settings.stereo_treble_center,
                        "Treble at center",
                    );
                    if settings.stereo_treble_center {
                        ui.small("Left: bass ← treble  |  treble → bass :Right");
                    } else {
                        ui.small("Left: treble ← bass  |  bass → treble :Right");
                    }
                }
                egui::ComboBox::from_label("Analysis method")
                    .selected_text(settings.analysis_method.label())
                    .show_ui(ui, |ui| {
                        for method in [
                            AnalysisMethod::MultirateFft,
                            AnalysisMethod::MultirateFft5,
                            AnalysisMethod::ResonatorBank,
                            AnalysisMethod::HybridResonatorFft,
                            AnalysisMethod::AuditoryErb,
                        ] {
                            ui.selectable_value(
                                &mut settings.analysis_method,
                                method,
                                method.label(),
                            );
                        }
                    });
                if matches!(
                    settings.analysis_method,
                    AnalysisMethod::ResonatorBank | AnalysisMethod::HybridResonatorFft
                ) {
                    ui.collapsing("Resonator tuning", |ui| {
                        ui.add(
                            egui::Slider::new(&mut settings.resonator_q_50_hz, 0.5..=32.0)
                                .logarithmic(true)
                                .text("Q at 50 Hz"),
                        );
                        ui.add(
                            egui::Slider::new(&mut settings.resonator_q_10_khz, 4.0..=512.0)
                                .logarithmic(true)
                                .text("Q at 10 kHz"),
                        );
                        ui.small("Q transitions smoothly in square-root frequency.");
                        egui::ComboBox::from_label("Level measurement")
                            .selected_text(settings.resonator_measurement.label())
                            .show_ui(ui, |ui| {
                                for measurement in [
                                    ResonatorMeasurement::RawMagnitude,
                                    ResonatorMeasurement::ImageRejected,
                                    ResonatorMeasurement::PositionVelocity,
                                    ResonatorMeasurement::PositionVelocityPi2,
                                    ResonatorMeasurement::PositionPi2,
                                ] {
                                    ui.selectable_value(
                                        &mut settings.resonator_measurement,
                                        measurement,
                                        measurement.label(),
                                    );
                                }
                            });
                        ui.small(
                            "Phase-corrected magnitude removes the counter-rotating image, accounts for filter startup, and preserves the natural attack without an extra sample buffer.",
                        );
                        if settings.resonator_measurement == ResonatorMeasurement::PositionVelocity {
                            ui.small("Measures position and discrete velocity together to estimate amplitude before position reaches its peak. No peak hold or lookahead; transient response can differ from phase-corrected magnitude.");
                        }
                        if settings.resonator_measurement == ResonatorMeasurement::PositionVelocityPi2 {
                            ui.small("Averages position/velocity energy now and one quarter-cycle earlier. Reduces phase-related ripple; the delayed energy can soften attacks and lengthen the tail, especially in bass.");
                        }
                        if settings.resonator_measurement == ResonatorMeasurement::PositionPi2 {
                            ui.small("Uses position now and one quarter-cycle earlier: sqrt(x_now² + x_delayed²), with gain calibration. No velocity measurement. The delayed position can soften attacks and extend tails.");
                        }
                        ui.checkbox(
                            &mut settings.resonator_multirate,
                            "Multirate resonator optimization",
                        );
                        ui.small(
                            "Uses anti-aliased /2, /4, and /8 audio rates below 8, 4, and 1.8 kHz.",
                        );
                        if settings.resonator_multirate {
                            ui.checkbox(
                                &mut settings.resonator_five_stage,
                                "Add fifth /16 stage",
                            );
                            ui.checkbox(
                                &mut settings.resonator_timestamp_alignment,
                                "Align multirate timestamps",
                            );
                        }
                        ui.small(
                            "The optional /16 stage handles frequencies below about 0.9 kHz.",
                        );
                        ui.small(
                            "Alignment adds about 1.11 ms in 4-stage mode or 2.38 ms in 5-stage mode.",
                        );
                        ui.separator();
                        egui::ComboBox::from_label("Output envelope")
                            .selected_text(settings.resonator_envelope.label())
                            .show_ui(ui, |ui| {
                                for envelope in [
                                    ResonatorEnvelope::DecayThenPeak,
                                    ResonatorEnvelope::PeakThenDecay,
                                ] {
                                    ui.selectable_value(
                                        &mut settings.resonator_envelope,
                                        envelope,
                                        envelope.label(),
                                    );
                                }
                            });
                        ui.small(match settings.resonator_envelope {
                            ResonatorEnvelope::DecayThenPeak => {
                                "current = max(|resonator|, decay(previous))"
                            }
                            ResonatorEnvelope::PeakThenDecay => {
                                "show max(|resonator|, previous); decay its next state"
                            }
                        });
                        ui.add(
                            egui::Slider::new(&mut settings.resonator_decay_db, 1.0..=60.0)
                                .suffix(" dB")
                                .text("Output decay amount"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.resonator_decay_seconds,
                                0.01..=5.0,
                            )
                            .logarithmic(true)
                            .suffix(" s")
                            .text("Output decay time"),
                        );
                        ui.small(format!(
                            "Equivalent to {:.1} dB/s. Bar gravity is applied afterward.",
                            settings.resonator_decay_db / settings.resonator_decay_seconds,
                        ));
                    });
                }
                if settings.analysis_method == AnalysisMethod::HybridResonatorFft {
                    ui.collapsing("Hybrid FFT tuning", |ui| {
                        egui::ComboBox::from_label("High-frequency FFT window")
                            .selected_text(settings.hybrid_fft_window.label())
                            .show_ui(ui, |ui| {
                                for window in [
                                    AnalysisWindow::Hann,
                                    AnalysisWindow::HalfHann,
                                    AnalysisWindow::HalfBlackman,
                                    AnalysisWindow::AsymmetricKaiser8,
                                    AnalysisWindow::Flanagan,
                                ] {
                                    ui.selectable_value(
                                        &mut settings.hybrid_fft_window,
                                        window,
                                        window.label(),
                                    );
                                }
                            });
                        if settings.hybrid_fft_window == AnalysisWindow::AsymmetricKaiser8 {
                            ui.add(egui::Slider::new(&mut settings.asymmetric_kaiser_beta, 0.0..=16.0)
                                .step_by(0.1).text("Kaiser β"));
                        } else if settings.hybrid_fft_window == AnalysisWindow::Flanagan {
                            ui.add(egui::Slider::new(&mut settings.flanagan_alpha, 2.0..=24.0)
                                .step_by(0.1).text("Flanagan α"));
                        }
                        egui::ComboBox::from_label("/2 FFT size")
                            .selected_text(format!("{} samples", settings.hybrid_half_fft_size))
                            .show_ui(ui, |ui| {
                                for size in [128, 256, 512, 1024, 2048, 4096] {
                                    ui.selectable_value(
                                        &mut settings.hybrid_half_fft_size,
                                        size,
                                        format!("{size} samples"),
                                    );
                                }
                            });
                        egui::ComboBox::from_label("/1 FFT size")
                            .selected_text(format!("{} samples", settings.hybrid_full_fft_size))
                            .show_ui(ui, |ui| {
                                for size in [128, 256, 512, 1024, 2048, 4096] {
                                    ui.selectable_value(
                                        &mut settings.hybrid_full_fft_size,
                                        size,
                                        format!("{size} samples"),
                                    );
                                }
                            });
                        ui.collapsing("Adaptive FFT level matching", |ui| {
                            ui.checkbox(
                                &mut settings.hybrid_adaptive_matching,
                                "Match FFT levels to reference resonators",
                            );
                            if settings.hybrid_adaptive_matching {
                                ui.add(
                                    egui::Slider::new(
                                        &mut settings.hybrid_match_adapt_seconds,
                                        0.01..=5.0,
                                    )
                                    .logarithmic(true)
                                    .suffix(" s")
                                    .text("Adapt time"),
                                );
                                ui.add(
                                    egui::Slider::new(
                                        &mut settings.hybrid_match_max_correction_db,
                                        0.0..=20.0,
                                    )
                                    .suffix(" dB")
                                    .text("Maximum correction"),
                                );
                                let references =
                                    automatic_hybrid_match_frequencies(settings, sample_rate);
                                ui.small(format!(
                                    "/2 resonator references: {}, {}, and {}.",
                                    format_frequency(references[0]),
                                    format_frequency(references[1]),
                                    format_frequency(references[2]),
                                ));
                                ui.small(format!(
                                    "/2 → /1 references: {}, {}, and {}.",
                                    format_frequency(references[3]),
                                    format_frequency(references[4]),
                                    format_frequency(references[5]),
                                ));
                            }
                            ui.small(
                                "Matching is chained from the lower-frequency analyzer into each FFT band. Each band receives one stable correction, which pauses automatically below the noise floor.",
                            );
                        });
                        fft_cadence_control(ui, settings);
                        ui.small(
                            "The /8 and /4 stages use resonators. The /2 and full-rate stages use FFT automatically.",
                        );
                    });
                }
                if settings.analysis_method == AnalysisMethod::AuditoryErb {
                    ui.collapsing("Auditory ERB tuning", |ui| {
                        ui.add(
                            egui::Slider::new(
                                &mut settings.auditory_erb_bandwidth_scale,
                                0.1..=10.0,
                            )
                            .logarithmic(true)
                            .suffix("× ERB")
                            .text("Ear bandwidth"),
                        );
                        ui.small(
                            "1× follows human critical bandwidth. Lower is sharper and slower; higher is smoother and faster.",
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.auditory_compression_percent,
                                0.0..=100.0,
                            )
                            .suffix("%")
                            .text("Cochlear compression"),
                        );
                        ui.small(
                            "Raises quieter energy within each auditory band, similar to the ear's compressive response.",
                        );
                        ui.add(
                            egui::Slider::new(&mut settings.auditory_filter_order, 1..=4)
                                .text("Filter order"),
                        );
                        ui.small(
                            "Order 1 is fastest; order 4 is the standard smooth gammatone response.",
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.auditory_phase_sharpening_percent,
                                0.0..=100.0,
                            )
                            .suffix("%")
                            .text("Phase sharpening at 50 Hz"),
                        );
                        let phase_sharpening_50_hz =
                            settings.auditory_phase_sharpening_percent;
                        ui.add(
                            egui::Slider::new(
                                settings
                                    .auditory_phase_sharpening_10_khz_percent
                                    .get_or_insert(phase_sharpening_50_hz),
                                0.0..=100.0,
                            )
                            .suffix("%")
                            .text("Phase sharpening at 10 kHz"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.auditory_tonal_width_percent,
                                5.0..=100.0,
                            )
                            .suffix("% ERB")
                            .text("Tonal width at 50 Hz"),
                        );
                        let tonal_width_50_hz = settings.auditory_tonal_width_percent;
                        ui.add(
                            egui::Slider::new(
                                settings
                                    .auditory_tonal_width_10_khz_percent
                                    .get_or_insert(tonal_width_50_hz),
                                5.0..=100.0,
                            )
                            .suffix("% ERB")
                            .text("Tonal width at 10 kHz"),
                        );
                        ui.small(
                            "Both values transition smoothly in log frequency. Stable tonal phase is narrowed; incoherent noise keeps its auditory bandwidth.",
                        );
                    });
                }
                if settings.analysis_method == AnalysisMethod::MultirateFft {
                    ui.collapsing("Multirate FFT tuning", |ui| {
                        egui::ComboBox::from_label("FFT window")
                            .selected_text(settings.multirate_fft_window.label())
                            .show_ui(ui, |ui| {
                                for window in [
                                    AnalysisWindow::Hann,
                                    AnalysisWindow::HalfHann,
                                    AnalysisWindow::HalfBlackman,
                                    AnalysisWindow::AsymmetricKaiser8,
                                    AnalysisWindow::Flanagan,
                                ] {
                                    ui.selectable_value(
                                        &mut settings.multirate_fft_window,
                                        window,
                                        window.label(),
                                    );
                                }
                            });
                        if settings.multirate_fft_window == AnalysisWindow::AsymmetricKaiser8 {
                            ui.add(
                                egui::Slider::new(
                                    &mut settings.asymmetric_kaiser_beta,
                                    0.0..=16.0,
                                )
                                .step_by(0.1)
                                .text("Kaiser β"),
                            );
                        } else if settings.multirate_fft_window == AnalysisWindow::Flanagan {
                            ui.add(
                                egui::Slider::new(&mut settings.flanagan_alpha, 2.0..=24.0)
                                    .step_by(0.1)
                                    .text("Flanagan α"),
                            );
                        }
                        fft_cadence_control(ui, settings);
                        ui.add(
                            egui::Slider::new(
                                &mut settings.multirate_fft_crossovers_hz[0],
                                40.0..=1000.0,
                            )
                            .logarithmic(true)
                            .suffix(" Hz")
                            .text("Bass crossover"),
                        );
                        let minimum_mid =
                            (settings.multirate_fft_crossovers_hz[0] * 1.2).max(500.0);
                        ui.add(
                            egui::Slider::new(
                                &mut settings.multirate_fft_crossovers_hz[1],
                                minimum_mid..=10_000.0,
                            )
                            .logarithmic(true)
                            .suffix(" Hz")
                            .text("High crossover"),
                        );
                        ui.separator();
                        for (index, name) in ["Bass", "Mid", "High"].into_iter().enumerate() {
                            ui.label(name);
                            ui.horizontal(|ui| {
                                egui::ComboBox::from_id_salt(("fft decimation", index))
                                    .selected_text(format!(
                                        "÷{}",
                                        settings.multirate_fft_decimations[index]
                                    ))
                                    .show_ui(ui, |ui| {
                                        for factor in [1, 2, 4, 8, 16, 32, 64, 128] {
                                            ui.selectable_value(
                                                &mut settings.multirate_fft_decimations[index],
                                                factor,
                                                format!("÷{factor}"),
                                            );
                                        }
                                    });
                                let maximum_size =
                                    (HISTORY_SAMPLE_COUNT as u32
                                        / settings.multirate_fft_decimations[index])
                                        .clamp(32, 8192);
                                egui::ComboBox::from_id_salt(("fft size", index))
                                    .selected_text(format!(
                                        "{} samples",
                                        settings.multirate_fft_sizes[index]
                                    ))
                                    .show_ui(ui, |ui| {
                                        for size in [
                                            32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                                        ] {
                                            if size <= maximum_size {
                                                ui.selectable_value(
                                                    &mut settings.multirate_fft_sizes[index],
                                                    size,
                                                    size.to_string(),
                                                );
                                            }
                                        }
                                    });
                                settings.multirate_fft_sizes[index] =
                                    settings.multirate_fft_sizes[index].min(maximum_size);
                            });
                            let effective_rate =
                                sample_rate / settings.multirate_fft_decimations[index] as f32;
                            let fft_size = settings.multirate_fft_sizes[index] as f32;
                            ui.small(format!(
                                "{:.1} ms history, {:.2} Hz bins, {:.1} kHz Nyquist",
                                fft_size / effective_rate * 1000.0,
                                effective_rate / fft_size,
                                effective_rate * 0.0005,
                            ));
                        }
                        ui.small(
                            "Decimation reduces FFT work but keeps the selected real-time history. Invalid crossover/Nyquist combinations fade to zero.",
                        );
                    });
                }
                if settings.analysis_method == AnalysisMethod::MultirateFft5 {
                    ui.collapsing("5-band Multirate FFT tuning", |ui| {
                        egui::ComboBox::from_label("FFT window")
                            .selected_text(settings.multirate_fft5_window.label())
                            .show_ui(ui, |ui| {
                                for window in [
                                    AnalysisWindow::Hann,
                                    AnalysisWindow::HalfHann,
                                    AnalysisWindow::HalfBlackman,
                                    AnalysisWindow::AsymmetricKaiser8,
                                    AnalysisWindow::Flanagan,
                                ] {
                                    ui.selectable_value(
                                        &mut settings.multirate_fft5_window,
                                        window,
                                        window.label(),
                                    );
                                }
                            });
                        if settings.multirate_fft5_window == AnalysisWindow::AsymmetricKaiser8 {
                            ui.add(
                                egui::Slider::new(
                                    &mut settings.asymmetric_kaiser_beta,
                                    0.0..=16.0,
                                )
                                .step_by(0.1)
                                .text("Kaiser β"),
                            );
                        } else if settings.multirate_fft5_window == AnalysisWindow::Flanagan {
                            ui.add(
                                egui::Slider::new(&mut settings.flanagan_alpha, 2.0..=24.0)
                                    .step_by(0.1)
                                    .text("Flanagan α"),
                            );
                        }
                        fft_cadence_control(ui, settings);
                        let crossover_labels = [
                            "Sub-bass crossover",
                            "Bass crossover",
                            "Mid crossover",
                            "Presence crossover",
                        ];
                        let crossover_limits = [
                            (40.0, 400.0),
                            (100.0, 1200.0),
                            (400.0, 4000.0),
                            (1500.0, 10_000.0),
                        ];
                        for (index, &(base_minimum, maximum)) in
                            crossover_limits.iter().enumerate()
                        {
                            let minimum = if index == 0 {
                                base_minimum
                            } else {
                                (settings.multirate_fft5_crossovers_hz[index - 1] * 1.2)
                                    .max(base_minimum)
                            };
                            ui.add(
                                egui::Slider::new(
                                    &mut settings.multirate_fft5_crossovers_hz[index],
                                    minimum..=maximum,
                                )
                                .logarithmic(true)
                                .suffix(" Hz")
                                .text(crossover_labels[index]),
                            );
                        }
                        ui.separator();
                        for (index, name) in
                            ["Sub-bass", "Bass", "Low-mid", "High-mid", "High"]
                                .into_iter()
                                .enumerate()
                        {
                            ui.label(name);
                            ui.horizontal(|ui| {
                                egui::ComboBox::from_id_salt(("fft5 decimation", index))
                                    .selected_text(format!(
                                        "÷{}",
                                        settings.multirate_fft5_decimations[index]
                                    ))
                                    .show_ui(ui, |ui| {
                                        for factor in [1, 2, 4, 8, 16, 32, 64, 128] {
                                            ui.selectable_value(
                                                &mut settings.multirate_fft5_decimations[index],
                                                factor,
                                                format!("÷{factor}"),
                                            );
                                        }
                                    });
                                let maximum_size = (HISTORY_SAMPLE_COUNT as u32
                                    / settings.multirate_fft5_decimations[index])
                                    .clamp(32, 8192);
                                egui::ComboBox::from_id_salt(("fft5 size", index))
                                    .selected_text(format!(
                                        "{} samples",
                                        settings.multirate_fft5_sizes[index]
                                    ))
                                    .show_ui(ui, |ui| {
                                        for size in
                                            [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
                                        {
                                            if size <= maximum_size {
                                                ui.selectable_value(
                                                    &mut settings.multirate_fft5_sizes[index],
                                                    size,
                                                    size.to_string(),
                                                );
                                            }
                                        }
                                    });
                                settings.multirate_fft5_sizes[index] =
                                    settings.multirate_fft5_sizes[index].min(maximum_size);
                            });
                            let effective_rate =
                                sample_rate / settings.multirate_fft5_decimations[index] as f32;
                            let fft_size = settings.multirate_fft5_sizes[index] as f32;
                            ui.small(format!(
                                "{:.1} ms history, {:.2} Hz bins, {:.1} kHz Nyquist",
                                fft_size / effective_rate * 1000.0,
                                effective_rate / fft_size,
                                effective_rate * 0.0005,
                            ));
                        }
                        ui.small(
                            "Five independently decimated FFTs crossfade smoothly between adjacent frequency ranges.",
                        );
                    });
                }
                    }
                    SettingsTab::Visual => {
                ui.checkbox(&mut settings.auto_output_sync, "Auto sync to audio output");
                ui.add(
                    egui::Slider::new(&mut settings.visualization_delay_ms, -1000.0..=1000.0)
                        .step_by(1.0)
                        .suffix(" ms")
                        .text("Sync adjustment"),
                );
                let automatic = if settings.auto_output_sync { output_latency_ms } else { None };
                ui.small(match automatic {
                    Some(ms) => format!("Output latency estimate: {ms:.0} ms"),
                    None if settings.auto_output_sync => "Output latency unavailable; using manual adjustment.".to_owned(),
                    None => "Automatic output sync is off.".to_owned(),
                });
                ui.small(format!("Applied visual delay: {:.0} ms", visual_delay_ms(settings, output_latency_ms)));
                ui.small("Increase adjustment if visuals lead the sound; decrease if they lag.");
                ui.separator();
                ui.label("Frequency range");
                ui.add(
                    egui::Slider::new(&mut settings.min_frequency_hz, 16.0..=200.0)
                        .logarithmic(true)
                        .step_by(1.0)
                        .suffix(" Hz")
                        .text("Lowest frequency"),
                );
                ui.add(
                    egui::Slider::new(&mut settings.max_frequency_hz, 10_000.0..=40_000.0)
                        .logarithmic(true)
                        .step_by(100.0)
                        .custom_formatter(|value, _| format!("{:.1} kHz", value / 1000.0))
                        .text("Highest frequency"),
                );
                let effective_max = settings.max_frequency_hz.min(sample_rate * 0.48);
                ui.small(format!(
                    "Effective maximum: {:.1} kHz (limited by the {:.1} kHz input sample rate).",
                    effective_max / 1000.0,
                    sample_rate / 1000.0,
                ));
                if settings.bar_layout == BarLayout::FixedPixels {
                    ui.add(
                    egui::Slider::new(
                        &mut settings.frequency_focus_strength_percent,
                        0.0..=100.0,
                    )
                    .step_by(1.0)
                    .suffix("%")
                    .text("Focus strength"),
                    );
                }
                if settings.bar_layout == BarLayout::FixedPixels {
                    ui.add(
                    egui::Slider::new(
                        &mut settings.frequency_focus_origin_percent,
                        0.0..=100.0,
                    )
                    .step_by(1.0)
                    .suffix("%")
                    .text("Focus origin"),
                    );
                }
                let origin_t = settings.frequency_focus_origin_percent * 0.01;
                let origin_hz = settings.min_frequency_hz
                    * (effective_max / settings.min_frequency_hz).powf(origin_t);
                if settings.bar_layout == BarLayout::MusicalNotes {
                    ui.small("Frequency focus is disabled because every semitone has equal width.");
                } else {
                    ui.small(format!(
                        "50% strength is unchanged. The current origin is {}.",
                        format_frequency(origin_hz),
                    ));
                }
                ui.separator();
                ui.label("dB crop and rescale");
                ui.add(
                    egui::Slider::new(&mut settings.low_db_floor, -120.0..=-20.0)
                        .suffix(" dB")
                        .text("Low-frequency floor"),
                );
                ui.add(
                    egui::Slider::new(&mut settings.high_db_floor, -120.0..=-20.0)
                        .suffix(" dB")
                        .text("High-frequency floor"),
                );
                ui.small("The floor changes smoothly across the logarithmic frequency range.");
                ui.separator();
                ui.label("Frequency axis");
                ui.checkbox(&mut settings.show_note_ribbon, "Show note activity at top");
                if settings.show_note_ribbon {
                    ui.checkbox(&mut settings.note_fast_response, "Fast note response");
                    ui.small("Uses short audio windows for quick note changes; close low notes may be less precise.");
                    ui.add(
                        egui::Slider::new(&mut settings.note_smoothing_ms, 0.0..=250.0)
                            .step_by(1.0)
                            .suffix(" ms")
                            .text("Note smoothing"),
                    );
                    ui.small("0 ms keeps the snappy look. Higher values soften glow and label transitions.");
                    ui.checkbox(&mut settings.note_midrange_only, "Midrange notes only (200 Hz–2 kHz)");
                    ui.add(
                        egui::Slider::new(&mut settings.note_ribbon_width_percent, 20.0..=100.0)
                            .step_by(1.0)
                            .suffix("%")
                            .text("Note ribbon width"),
                    );
                }
                ui.small("C–B combines octaves and channels; the strongest note glows brighter. Harmonics also contribute.");
                egui::ComboBox::from_label("Labels")
                    .selected_text(settings.axis_label_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in [AxisLabelMode::Notes, AxisLabelMode::Hertz] {
                            ui.selectable_value(&mut settings.axis_label_mode, mode, mode.label());
                        }
                    });
                ui.checkbox(&mut settings.axis_labels_overlay, "Overlay labels on visual");
                ui.checkbox(&mut settings.show_frequency_grid, "Show frequency grid");
                ui.small(if settings.axis_labels_overlay {
                    "Bars extend behind the frequency labels."
                } else {
                    "Space is reserved below the bars for frequency labels."
                });
                ui.separator();
                ui.label("Bar appearance");
                egui::ComboBox::from_label("Render mode")
                    .selected_text(settings.render_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in [RenderMode::Flat, RenderMode::Surface3d] {
                            ui.selectable_value(&mut settings.render_mode, mode, mode.label());
                        }
                    });
                if settings.render_mode == RenderMode::Surface3d {
                    ui.add(
                        egui::Slider::new(&mut settings.surface_camera_height, 0.25..=3.0)
                            .step_by(0.01)
                            .text("Camera Y position"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut settings.surface_camera_distance,
                            1.25..=8.0,
                        )
                        .step_by(0.05)
                        .text("Camera distance"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut settings.surface_camera_fov_degrees,
                            20.0..=100.0,
                        )
                        .step_by(1.0)
                        .suffix("°")
                        .text("Field of view"),
                    );
                    ui.add(
                        egui::Slider::new(&mut settings.surface_camera_target_y, 0.0..=1.0)
                            .step_by(0.01)
                            .text("Camera Y anchor"),
                    );
                    let depth_px = settings.surface_bar_depth_px.get_or_insert(6.0);
                    ui.add(
                        egui::Slider::new(depth_px, 0.0..=300.0)
                            .step_by(0.5)
                            .suffix(" px")
                            .text("3D bar depth"),
                    );
                    if settings.surface_bar_depth_px.unwrap_or(0.0) > 0.0 {
                    ui.add(
                        egui::Slider::new(&mut settings.surface_volume_stage_count, 3..=8)
                            .text("3D volume stages"),
                    );
                    ui.small(
                        "Each stage is twice as far forward as the previous one; the final stage reaches full depth.",
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut settings.surface_peak_depth_brightness_percent,
                            0.0..=100.0,
                        )
                        .step_by(1.0)
                        .suffix("%")
                        .text("Peak depth brightness"),
                    );
                    ui.small(
                        "0% hides rear peak markers; the front peak remains fully visible.",
                    );
                    ui.small(format!(
                        "Camera ({:.2}, {:.2}, {:.2}) looks at (0.00, {:.2}, 0.00).",
                        0.0,
                        settings.surface_camera_height,
                        settings.surface_camera_distance,
                        settings.surface_camera_target_y,
                    ));
                    }
                    ui.add(
                        egui::Slider::new(
                            &mut settings.surface_reflection_strength_percent,
                            0.0..=100.0,
                        )
                        .step_by(1.0)
                        .suffix("%")
                        .text("Surface reflection"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut settings.surface_reflection_fade_percent,
                            0.0..=2000.0,
                        )
                        .step_by(1.0)
                        .suffix("%")
                        .text("Reflection fade strength"),
                    );
                    ui.separator();
                    ui.checkbox(
                        &mut settings.surface_history_enabled,
                        "Rendered spectrum history",
                    );
                    if settings.surface_history_enabled {
                        egui::ComboBox::from_label("Maximum history")
                            .selected_text(format!("{} rows", settings.surface_history_rows))
                            .show_ui(ui, |ui| {
                                for rows in [64, 128, 256, 512, 1024, 2048] {
                                    ui.selectable_value(
                                        &mut settings.surface_history_rows,
                                        rows,
                                        format!("{rows} rows"),
                                    );
                                }
                            });
                        ui.add(
                            egui::Slider::new(
                                &mut settings.surface_history_fade_percent,
                                0.0..=200.0,
                            )
                            .step_by(0.1)
                            .suffix("%")
                            .text("History fade strength"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.surface_history_opacity_percent,
                                0.0..=100.0,
                            )
                            .step_by(1.0)
                            .suffix("%")
                            .text("History opacity"),
                        );
                        ui.checkbox(
                            &mut settings.surface_history_grayscale,
                            "Grayscale history",
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.surface_history_brightness_percent,
                                0.0..=300.0,
                            )
                            .step_by(1.0)
                            .suffix("%")
                            .text("History brightness"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.surface_history_contrast_percent,
                                25.0..=300.0,
                            )
                            .step_by(1.0)
                            .suffix("%")
                            .text("History contrast"),
                        );
                    }
                    ui.small(
                        "Captures the rendered bar baseline into a circular GPU texture once per visual frame. 100% fade leaves 1% brightness at the oldest edge.",
                    );
                    ui.separator();
                    ui.checkbox(
                        &mut settings.surface_peak_trail_enabled,
                        "Peak moving trail",
                    );
                    if settings.surface_peak_trail_enabled {
                        ui.add(
                            egui::Slider::new(
                                &mut settings.surface_peak_trail_fade_seconds,
                                0.05..=5.0,
                            )
                            .logarithmic(true)
                            .suffix(" s")
                            .text("Trail fade time"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.surface_peak_trail_movement_percent,
                                0.0..=300.0,
                            )
                            .step_by(1.0)
                            .suffix("%")
                            .text("Trail movement"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.surface_peak_trail_opacity_percent,
                                0.0..=200.0,
                            )
                            .step_by(1.0)
                            .suffix("%")
                            .text("Trail opacity"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut settings.surface_peak_trail_capture_interval_frames,
                                1..=10,
                            )
                            .text("Capture every N frames"),
                        );
                    }
                    ui.small(
                        "Transforms the previous peak image toward the viewer, fades it by elapsed time, then overlays the newest peak.",
                    );
                    ui.small(
                        "The live spectrum, history floor, and reflection share one perspective camera.",
                    );
                }
                ui.checkbox(
                    &mut settings.transparent_background,
                    "Transparent background",
                );
                ui.small("Converts rendered light into window transparency.");
                if settings.render_mode == RenderMode::Flat {
                    ui.checkbox(&mut settings.bottom_mirror, "Bottom mirror");
                }
                if settings.render_mode == RenderMode::Flat && settings.bottom_mirror {
                    ui.add(
                        egui::Slider::new(&mut settings.mirror_baseline_percent, 0.0..=100.0)
                            .step_by(1.0)
                            .suffix("%")
                            .text("Mirror baseline"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut settings.mirror_initial_fade_percent,
                            0.0..=100.0,
                        )
                        .step_by(1.0)
                        .suffix("%")
                        .text("Reflection initial fade"),
                    );
                    ui.add(
                        egui::Slider::new(&mut settings.mirror_fade_percent, 0.0..=2000.0)
                            .step_by(1.0)
                            .suffix("%")
                            .text("Reflection fade strength"),
                    );
                    }
                if settings.render_mode == RenderMode::Flat {
                    ui.small(
                        "The reflection matches the original height and fades exponentially from its initial brightness.",
                    );
                }
                egui::ComboBox::from_label("Bar layout")
                    .selected_text(settings.bar_layout.label())
                    .show_ui(ui, |ui| {
                        for mode in [BarLayout::FixedPixels, BarLayout::MusicalNotes] {
                            ui.selectable_value(&mut settings.bar_layout, mode, mode.label());
                        }
                    });
                if settings.bar_layout == BarLayout::FixedPixels {
                    ui.add(
                    egui::Slider::new(&mut settings.bar_width_px, 1.0..=16.0)
                        .suffix(" px")
                        .text("Fixed bar width"),
                    );
                }
                ui.add(
                    egui::Slider::new(&mut settings.bar_spacing_px, 0.0..=12.0)
                        .step_by(0.1)
                        .suffix(" px")
                        .text("Bar spacing"),
                );
                ui.add(
                    egui::Slider::new(&mut settings.bar_top_fade_percent, 0.0..=99.0)
                        .step_by(1.0)
                        .suffix("%")
                        .text("Bar top fade"),
                );
                if settings.bar_layout == BarLayout::MusicalNotes {
                    ui.small(
                        "Each chromatic semitone is one bar (A4 = 440 Hz); one octave has 12 bars.",
                    );
                } else {
                    ui.small("The number of frequency bars follows the available window width.");
                }
                ui.add(
                    egui::Slider::new(&mut settings.min_brightness, 0.02..=1.0)
                        .text("Minimum brightness"),
                );
                ui.add(
                    egui::Slider::new(&mut settings.max_brightness, 1.0..=8.0)
                        .text("Maximum brightness"),
                );
                ui.small("Bar light maps from the local dB floor to 0 dB.");
                ui.checkbox(&mut settings.rise_highlight, "Highlight fast rises");
                if settings.rise_highlight {
                    ui.add(
                        egui::Slider::new(&mut settings.rise_highlight_strength, 0.0..=3.0)
                            .text("Peak glow strength"),
                    );
                    ui.add(
                        egui::Slider::new(&mut settings.rise_highlight_fade_ms, 20.0..=1000.0)
                            .logarithmic(true)
                            .suffix(" ms")
                            .text("Peak highlight fade"),
                    );
                    ui.add(
                        egui::Slider::new(&mut settings.peak_glow_full_scale_db, 0.5..=30.0)
                            .logarithmic(true)
                            .suffix(" dB")
                            .text("Full glow rise"),
                    );
                }
                ui.small(
                    "Peak and bar glow follow the same dB rise. Bar glow fades about 3× faster to preserve detail.",
                );
                ui.add(
                    egui::Slider::new(&mut settings.bar_gravity_db_per_second2, 10.0..=5000.0)
                        .logarithmic(true)
                        .suffix(" dB/s²")
                        .text("Bar gravity"),
                );
                ui.small("Bars rise instantly, have zero hold, then accelerate downward.");
                ui.add(
                    egui::Slider::new(&mut settings.peak_hold_ms, 0.0..=1000.0)
                        .suffix(" ms")
                        .text("Peak hold"),
                );
                ui.add(
                    egui::Slider::new(&mut settings.peak_gravity_db_per_second2, 5.0..=5000.0)
                        .logarithmic(true)
                        .suffix(" dB/s²")
                        .text("Peak gravity"),
                );
                ui.checkbox(&mut settings.peak_momentum, "Peak momentum");
                ui.small(
                    "Peak keeps a little upward velocity, pauses at its apex, then falls under gravity.",
                );
                ui.separator();
                    }
                }
                ui.separator();
                if ui.button("Reset defaults").clicked() {
                    *settings = VisualizerSettings::default();
                }
                ui.small(
                    "Press H or right-click the visualizer to show this window. Up/Down adjust gain.",
                );
            });
    });
}

pub struct GpuState {
    surface: wgpu::Surface<'static>,
    controls_surface: wgpu::Surface<'static>,
    compute_pipeline_layout: wgpu::PipelineLayout,
    render_pipeline_layout: wgpu::PipelineLayout,
    composite_pipeline_layout: wgpu::PipelineLayout,
    peak_feedback_pipeline_layout: wgpu::PipelineLayout,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    controls_config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    controls_size: PhysicalSize<u32>,
    pending_size: Option<PhysicalSize<u32>>,
    last_surface_configure: Instant,
    resonator_pipeline: Option<wgpu::ComputePipeline>,
    gammatone_pipeline: Option<wgpu::ComputePipeline>,
    multirate_fft_pipeline: Option<wgpu::ComputePipeline>,
    hybrid_fft_pipeline: Option<wgpu::ComputePipeline>,
    compose_display_pipeline: wgpu::ComputePipeline,
    static_grid: StaticGrid,
    axis_overlay_pipeline: Option<wgpu::RenderPipeline>,
    note_ribbon: Option<NoteRibbon>,
    glow_pipeline: wgpu::RenderPipeline,
    render_pipeline: wgpu::RenderPipeline,
    prism_pipeline: Option<wgpu::RenderPipeline>,
    peak_glow_pipeline: wgpu::RenderPipeline,
    peak_pipeline: wgpu::RenderPipeline,
    peak_volume_pipeline: Option<wgpu::RenderPipeline>,
    peak_feedback_pipeline: Option<wgpu::RenderPipeline>,
    peak_feedback_bind_group_layout: wgpu::BindGroupLayout,
    peak_feedback_bind_groups: [wgpu::BindGroup; 2],
    peak_trail_sampler: wgpu::Sampler,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_bind_groups: [wgpu::BindGroup; 2],
    composite_params_buffer: wgpu::Buffer,
    visual_texture: wgpu::Texture,
    visual_view: wgpu::TextureView,
    surface_history_texture: wgpu::Texture,
    _surface_history_view: wgpu::TextureView,
    surface_history_rows: u32,
    surface_history_write_row: u32,
    surface_history_valid_rows: u32,
    _peak_source_texture: wgpu::Texture,
    peak_source_view: wgpu::TextureView,
    _peak_trail_textures: [wgpu::Texture; 2],
    peak_trail_views: [wgpu::TextureView; 2],
    peak_trail_previous_index: usize,
    peak_trail_was_enabled: bool,
    peak_trail_capture_frame: u32,
    analyzers: Vec<AnalyzerChannel>,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,
    resource_mode: ResourceMode,
    target_features: (bool, bool, bool),
    left_history: Vec<f32>,
    right_history: Vec<f32>,
    // Follows capture: an output device or format change restarts the
    // history, and `resource_mode` then rebuilds the analyzers.
    sample_rate: f32,
    capture_stall: CaptureStall,
    started: Instant,
    output_latency_ms: Option<f32>,
    last_frame: Instant,
    last_bin_count: u32,
    last_db_floors: [f32; 2],
    last_frequency_mapping: [f32; 4],
    last_analysis_method: AnalysisMethod,
    last_channel_mode: ChannelMode,
    last_delay_samples: usize,
    last_resonator_q: [f32; 2],
    last_resonator_measurement: ResonatorMeasurement,
    last_resonator_envelope: ResonatorEnvelope,
    last_resonator_multirate: bool,
    last_resonator_five_stage: bool,
    last_resonator_timestamp_alignment: bool,
    last_hybrid_adaptive_matching: bool,
    last_auditory_filter_order: u32,
    last_auditory_display_settings: [f32; 6],
    last_multirate_crossovers: [f32; 4],
    last_peak_momentum: bool,
    resonator_needs_reset: bool,
    // Samples were drained without analysis (hidden window); streaming
    // analyzers must restart from the contiguous history.
    audio_gap: bool,
    settings_path: PathBuf,
    settings: VisualizerSettings,
    settings_dirty: bool,
    settings_save_deadline: Option<Instant>,
    ui: UiLayer,
}

struct AnalyzerChannel {
    compute_bind_group: wgpu::BindGroup,
    fft_compute_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    sample_buffer: wgpu::Buffer,
    fft_magnitude_buffer: wgpu::Buffer,
    spectrum_buffer: wgpu::Buffer,
    resonator_half_buffer: wgpu::Buffer,
    resonator_quarter_buffer: wgpu::Buffer,
    resonator_eighth_buffer: wgpu::Buffer,
    resonator_sixteenth_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    render_channel_buffer: wgpu::Buffer,
    multirate_fft: MultirateFft,
    hybrid_band_matcher: HybridBandMatcher,
    fft_match_elapsed: f32,
    resonator_resampler: ResonatorResampler,
    audio_history: Vec<f32>,
    samples: Vec<f32>,
}

impl GpuState {
    pub async fn new(
        window: Arc<Window>,
        controls_window: Arc<Window>,
        sample_rate: u32,
        profile: Option<&str>,
    ) -> Result<Self> {
        let size = window.inner_size();
        let controls_size = controls_window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("could not create the GPU window surface")?;
        let controls_surface = instance
            .create_surface(Arc::clone(&controls_window))
            .context("could not create the controls GPU surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("no compatible GPU adapter found")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("visual-audio device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .context("could not create a GPU device")?;

        let caps = surface.get_capabilities(&adapter);
        let controls_caps = controls_surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb() && controls_caps.formats.contains(format))
            .unwrap_or(caps.formats[0]);
        // FIFO's drawable wait paces frames to the display refresh (the event
        // loop relies on it). Metal offers no Mailbox mode in any case.
        let present_mode = wgpu::PresentMode::Fifo;
        let alpha_mode = if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
        {
            wgpu::CompositeAlphaMode::PreMultiplied
        } else if caps
            .alpha_modes
            .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
        {
            wgpu::CompositeAlphaMode::PostMultiplied
        } else {
            caps.alpha_modes[0]
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            // At most one queued frame (two drawables on Metal): if frames are
            // produced faster than the display, a full queue adds one refresh
            // of latency rather than two. Frames take well under a millisecond.
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &config);
        let controls_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: controls_size.width.max(1),
            height: controls_size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: controls_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        controls_surface.configure(&device, &controls_config);

        let history_sample_count = history_sample_count(sample_rate);
        let settings_path =
            settings_path(profile).context("could not find the macOS home directory")?;
        let settings = VisualizerSettings::load(&settings_path);
        let initial_bin_count = display_bin_count(&settings, size, sample_rate as f32);
        let initial_params = analysis_params(
            &settings,
            [sample_rate as f32, 0.0, 0.0],
            initial_bin_count,
            true,
            ([0; 5], true),
            [0.0; 4],
            size,
        );
        let (compute_bind_group_layout, render_bind_group_layout) = analyzer_layouts(&device);
        let analyzers = (0..settings.channel_mode.channel_count())
            .map(|index| {
                create_analyzer_channel(
                    &device,
                    &compute_bind_group_layout,
                    &render_bind_group_layout,
                    &initial_params,
                    history_sample_count,
                    index,
                    settings.channel_mode.channel_count() as f32,
                    &settings,
                )
            })
            .collect();
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("analysis pipeline layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visualizer pipeline layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("analysis.wgsl"));

        let compose_display_pipeline = compute_pipeline(
            &device,
            &compute_pipeline_layout,
            &compute_shader,
            "spectrum display composition",
            "compose_display",
        );

        let render_shader = device.create_shader_module(wgpu::include_wgsl!("visualizer.wgsl"));
        let static_grid = StaticGrid::new(
            &device,
            &render_bind_group_layout,
            &render_shader,
            format,
        );

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("spectrum renderer"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let glow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("spectrum additive glow"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("glow_vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("glow_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let peak_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("per-bin peak indicators"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("peak_vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("peak_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let peak_glow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("peak indicator onset glow"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("peak_glow_vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("peak_glow_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visual compositor bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    uniform_entry(1, wgpu::ShaderStages::FRAGMENT),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visual compositor pipeline layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let peak_feedback_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("peak feedback bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    uniform_entry(2, wgpu::ShaderStages::FRAGMENT),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let peak_feedback_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("peak feedback pipeline layout"),
                bind_group_layouts: &[&peak_feedback_bind_group_layout],
                push_constant_ranges: &[],
            });

        let composite_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("visual compositor parameters"),
                contents: bytemuck::bytes_of(&CompositeParams {
                    transparent_background: if settings.transparent_background {
                        1.0
                    } else {
                        0.0
                    },
                    surface_mode: if settings.render_mode == RenderMode::Surface3d {
                        1.0
                    } else {
                        0.0
                    },
                    surface_camera_height: settings.surface_camera_height,
                    surface_camera_distance: settings.surface_camera_distance,
                    surface_camera_fov_degrees: settings.surface_camera_fov_degrees,
                    surface_camera_target_y: settings.surface_camera_target_y,
                    surface_reflection_strength_percent: settings
                        .surface_reflection_strength_percent,
                    surface_reflection_fade_percent: settings.surface_reflection_fade_percent,
                    surface_history_enabled: 0.0,
                    surface_history_write_row: 0.0,
                    surface_history_valid_rows: 0.0,
                    surface_history_fade_percent: settings.surface_history_fade_percent,
                    surface_history_opacity_percent: settings.surface_history_opacity_percent,
                    surface_history_grayscale: if settings.surface_history_grayscale {
                        1.0
                    } else {
                        0.0
                    },
                    surface_history_brightness_percent: settings.surface_history_brightness_percent,
                    surface_history_contrast_percent: settings.surface_history_contrast_percent,
                    peak_trail_enabled: 0.0,
                    peak_trail_delta_seconds: 0.0,
                    peak_trail_fade_seconds: settings.surface_peak_trail_fade_seconds,
                    peak_trail_movement_percent: settings.surface_peak_trail_movement_percent,
                    peak_trail_opacity_percent: settings.surface_peak_trail_opacity_percent,
                    peak_trail_history_rows: settings
                        .surface_history_rows
                        .clamp(MIN_SURFACE_HISTORY_ROWS, MAX_SURFACE_HISTORY_ROWS)
                        as f32,
                    peak_trail_capture_current: 0.0,
                    _peak_trail_padding_2: 0.0,
                }),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let peak_trail_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("peak trail linear sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let initial_surface_history_rows = settings
            .surface_history_rows
            .clamp(MIN_SURFACE_HISTORY_ROWS, MAX_SURFACE_HISTORY_ROWS);
        let (
            peak_source_texture,
            peak_source_view,
            peak_trail_textures,
            peak_trail_views,
            peak_feedback_bind_groups,
        ) = create_peak_trail_targets(
            &device,
            &peak_feedback_bind_group_layout,
            &composite_params_buffer,
            &peak_trail_sampler,
            format,
            target_size(size, visual_target_features(&settings).2),
        );
        let (
            visual_texture,
            visual_view,
            surface_history_texture,
            surface_history_view,
            composite_bind_groups,
        ) = create_visual_targets(
            &device,
            &composite_bind_group_layout,
            &composite_params_buffer,
            format,
            target_size(size, visual_target_features(&settings).0),
            initial_surface_history_rows,
            visual_target_features(&settings).1,
            &peak_trail_views,
        );
        let ui = UiLayer::new(&device, format, controls_window.as_ref());

        Ok(Self {
            surface,
            controls_surface,
            compute_pipeline_layout,
            render_pipeline_layout,
            composite_pipeline_layout,
            peak_feedback_pipeline_layout,
            device,
            queue,
            config,
            controls_config,
            size,
            controls_size,
            pending_size: None,
            last_surface_configure: Instant::now(),
            resonator_pipeline: None,
            gammatone_pipeline: None,
            multirate_fft_pipeline: None,
            hybrid_fft_pipeline: None,
            compose_display_pipeline,
            static_grid,
            axis_overlay_pipeline: None,
            note_ribbon: None,
            glow_pipeline,
            render_pipeline,
            prism_pipeline: None,
            peak_glow_pipeline,
            peak_pipeline,
            peak_volume_pipeline: None,
            peak_feedback_pipeline: None,
            peak_feedback_bind_group_layout,
            peak_feedback_bind_groups,
            peak_trail_sampler,
            composite_pipeline: None,
            composite_bind_group_layout,
            composite_bind_groups,
            composite_params_buffer,
            visual_texture,
            visual_view,
            surface_history_texture,
            _surface_history_view: surface_history_view,
            surface_history_rows: initial_surface_history_rows,
            surface_history_write_row: 0,
            surface_history_valid_rows: 0,
            _peak_source_texture: peak_source_texture,
            peak_source_view,
            _peak_trail_textures: peak_trail_textures,
            peak_trail_views,
            peak_trail_previous_index: 0,
            peak_trail_was_enabled: false,
            peak_trail_capture_frame: 0,
            analyzers,
            compute_bind_group_layout,
            render_bind_group_layout,
            resource_mode: resource_mode(&settings, sample_rate as f32),
            target_features: visual_target_features(&settings),
            left_history: vec![0.0; history_sample_count],
            right_history: vec![0.0; history_sample_count],
            sample_rate: sample_rate as f32,
            capture_stall: CaptureStall::new(Instant::now()),
            output_latency_ms: None,
            started: Instant::now(),
            last_frame: Instant::now(),
            last_bin_count: initial_bin_count,
            last_db_floors: [settings.low_db_floor, settings.high_db_floor],
            last_frequency_mapping: [
                settings.min_frequency_hz,
                settings.max_frequency_hz,
                settings.frequency_focus_strength_percent,
                settings.frequency_focus_origin_percent,
            ],
            last_analysis_method: settings.analysis_method,
            last_channel_mode: settings.channel_mode,
            last_delay_samples: 0,
            last_resonator_q: [settings.resonator_q_50_hz, settings.resonator_q_10_khz],
            last_resonator_measurement: settings.resonator_measurement,
            last_resonator_envelope: settings.resonator_envelope,
            last_resonator_multirate: settings.resonator_multirate,
            last_resonator_five_stage: settings.resonator_five_stage,
            last_resonator_timestamp_alignment: settings.resonator_timestamp_alignment,
            last_hybrid_adaptive_matching: settings.hybrid_adaptive_matching,
            last_auditory_filter_order: settings.auditory_filter_order,
            last_auditory_display_settings: [
                settings.auditory_erb_bandwidth_scale,
                settings.auditory_compression_percent,
                settings.auditory_phase_sharpening_percent,
                settings
                    .auditory_phase_sharpening_10_khz_percent
                    .unwrap_or(settings.auditory_phase_sharpening_percent),
                settings.auditory_tonal_width_percent,
                settings
                    .auditory_tonal_width_10_khz_percent
                    .unwrap_or(settings.auditory_tonal_width_percent),
            ],
            last_multirate_crossovers: multirate_crossovers(&settings),
            last_peak_momentum: settings.peak_momentum,
            resonator_needs_reset: true,
            audio_gap: false,
            settings_path,
            settings,
            settings_dirty: false,
            settings_save_deadline: None,
            ui,
        })
    }

    pub fn resize_visualizer(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.pending_size = Some(size);
        }
    }

    pub fn resize_controls(&mut self, size: PhysicalSize<u32>) {
        if size.width > 0 && size.height > 0 {
            self.controls_size = size;
            self.controls_config.width = size.width;
            self.controls_config.height = size.height;
            self.controls_surface
                .configure(&self.device, &self.controls_config);
        }
    }

    fn apply_pending_resize(&mut self) {
        if self.last_surface_configure.elapsed() < std::time::Duration::from_millis(50) {
            return;
        }
        if let Some(size) = self.pending_size.take() {
            self.size = size;
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
            self.recreate_visual_targets();
            self.last_surface_configure = Instant::now();
        }
    }

    fn prepare_resources(&mut self) {
        let resource_mode = resource_mode(&self.settings, self.sample_rate);
        let count = self.settings.channel_mode.channel_count();
        if self.resource_mode != resource_mode {
            self.analyzers.clear();
            self.note_ribbon = None;
            self.resource_mode = resource_mode;
            self.resonator_needs_reset = true;
        }
        if self.analyzers.len() != count {
            self.note_ribbon = None;
        }
        self.analyzers.truncate(count);
        while self.analyzers.len() < count {
            let params = analysis_params(
                &self.settings,
                [self.sample_rate, 0.0, 0.0],
                display_bin_count(&self.settings, self.size, self.sample_rate),
                true,
                ([0; 5], true),
                [0.0; 4],
                self.size,
            );
            self.analyzers.push(create_analyzer_channel(
                &self.device,
                &self.compute_bind_group_layout,
                &self.render_bind_group_layout,
                &params,
                self.left_history.len(),
                self.analyzers.len(),
                count as f32,
                &self.settings,
            ));
            self.resonator_needs_reset = true;
        }
        if self.settings.show_note_ribbon {
            if self.note_ribbon.is_none() {
                self.note_ribbon = Some(NoteRibbon::new(
                    &self.device,
                    &self.queue,
                    self.config.format,
                    &self.analyzers[0].params_buffer,
                    &self.analyzers[0].spectrum_buffer,
                    &self.analyzers[count - 1].spectrum_buffer,
                ));
            }
            self.note_ribbon.as_mut().unwrap().configure(
                &self.queue,
                self.settings.note_ribbon_width_percent,
                self.settings.note_midrange_only,
                self.settings.note_fast_response,
            );
            self.note_ribbon.as_mut().unwrap().set_smoothing(
                &self.queue,
                self.settings.note_smoothing_ms,
            );
        } else {
            self.note_ribbon = None;
        }
        if matches!(
            self.settings.analysis_method,
            AnalysisMethod::ResonatorBank | AnalysisMethod::HybridResonatorFft
        ) {
            if self.resonator_pipeline.is_none() {
                self.resonator_pipeline = Some(build_resonator_pipeline(
                    &self.device,
                    &self.compute_pipeline_layout,
                ));
            }
        } else {
            self.resonator_pipeline = None;
        }
        if self.settings.analysis_method == AnalysisMethod::AuditoryErb {
            if self.gammatone_pipeline.is_none() {
                self.gammatone_pipeline = Some(build_gammatone_pipeline(
                    &self.device,
                    &self.compute_pipeline_layout,
                ));
            }
        } else {
            self.gammatone_pipeline = None;
        }
        if matches!(
            self.settings.analysis_method,
            AnalysisMethod::MultirateFft | AnalysisMethod::MultirateFft5
        ) {
            if self.multirate_fft_pipeline.is_none() {
                self.multirate_fft_pipeline = Some(build_multirate_fft_pipeline(
                    &self.device,
                    &self.compute_pipeline_layout,
                ));
            }
        } else {
            self.multirate_fft_pipeline = None;
        }
        if self.settings.analysis_method == AnalysisMethod::HybridResonatorFft {
            if self.hybrid_fft_pipeline.is_none() {
                self.hybrid_fft_pipeline = Some(build_hybrid_fft_pipeline(
                    &self.device,
                    &self.compute_pipeline_layout,
                ));
            }
        } else {
            self.hybrid_fft_pipeline = None;
        }
        if self.settings.axis_labels_overlay {
            if self.axis_overlay_pipeline.is_none() {
                self.axis_overlay_pipeline = Some(build_axis_overlay_pipeline(
                    &self.device,
                    &self.render_pipeline_layout,
                    self.config.format,
                ));
            }
        } else {
            self.axis_overlay_pipeline = None;
        }
        if self.settings.render_mode == RenderMode::Surface3d
            && self.settings.surface_bar_depth_px.unwrap_or(0.0) > 0.0
        {
            if self.prism_pipeline.is_none() {
                self.prism_pipeline = Some(build_prism_pipeline(
                    &self.device,
                    &self.render_pipeline_layout,
                    self.config.format,
                ));
            }
        } else {
            self.prism_pipeline = None;
        }
        if self.settings.render_mode == RenderMode::Surface3d
            && self.settings.surface_bar_depth_px.unwrap_or(0.0) > 0.0
        {
            if self.peak_volume_pipeline.is_none() {
                self.peak_volume_pipeline = Some(build_peak_volume_pipeline(
                    &self.device,
                    &self.render_pipeline_layout,
                    self.config.format,
                ));
            }
        } else {
            self.peak_volume_pipeline = None;
        }
        if visual_target_features(&self.settings).0 {
            if self.composite_pipeline.is_none() {
                self.composite_pipeline = Some(build_composite_pipeline(
                    &self.device,
                    &self.composite_pipeline_layout,
                    self.config.format,
                ));
            }
        } else {
            self.composite_pipeline = None;
        }
        if visual_target_features(&self.settings).2 {
            if self.peak_feedback_pipeline.is_none() {
                self.peak_feedback_pipeline = Some(build_peak_feedback_pipeline(
                    &self.device,
                    &self.peak_feedback_pipeline_layout,
                    self.config.format,
                ));
            }
        } else {
            self.peak_feedback_pipeline = None;
        }
    }

    fn recreate_visual_targets(&mut self) {
        let desired_history_rows = self
            .settings
            .surface_history_rows
            .clamp(MIN_SURFACE_HISTORY_ROWS, MAX_SURFACE_HISTORY_ROWS);
        let (
            peak_source_texture,
            peak_source_view,
            peak_trail_textures,
            peak_trail_views,
            peak_feedback_bind_groups,
        ) = create_peak_trail_targets(
            &self.device,
            &self.peak_feedback_bind_group_layout,
            &self.composite_params_buffer,
            &self.peak_trail_sampler,
            self.config.format,
            target_size(self.size, visual_target_features(&self.settings).2),
        );
        let (
            visual_texture,
            visual_view,
            surface_history_texture,
            surface_history_view,
            composite_bind_groups,
        ) = create_visual_targets(
            &self.device,
            &self.composite_bind_group_layout,
            &self.composite_params_buffer,
            self.config.format,
            target_size(self.size, visual_target_features(&self.settings).0),
            desired_history_rows,
            visual_target_features(&self.settings).1,
            &peak_trail_views,
        );
        self.target_features = visual_target_features(&self.settings);
        self.visual_texture = visual_texture;
        self.visual_view = visual_view;
        self.surface_history_texture = surface_history_texture;
        self._surface_history_view = surface_history_view;
        self.composite_bind_groups = composite_bind_groups;
        self.surface_history_rows = desired_history_rows;
        self.surface_history_write_row = 0;
        self.surface_history_valid_rows = 0;
        self._peak_source_texture = peak_source_texture;
        self.peak_source_view = peak_source_view;
        self._peak_trail_textures = peak_trail_textures;
        self.peak_trail_views = peak_trail_views;
        self.peak_feedback_bind_groups = peak_feedback_bind_groups;
        self.peak_trail_previous_index = 0;
        self.peak_trail_was_enabled = false;
        self.peak_trail_capture_frame = 0;
    }

    fn apply_surface_history_size(&mut self) {
        let desired_history_rows = self
            .settings
            .surface_history_rows
            .clamp(MIN_SURFACE_HISTORY_ROWS, MAX_SURFACE_HISTORY_ROWS);
        if self.target_features != visual_target_features(&self.settings)
            || (visual_target_features(&self.settings).1
                && desired_history_rows != self.surface_history_rows)
        {
            self.recreate_visual_targets();
        }
    }

    /// Keep the capture ring from filling while the visualizer is hidden. The
    /// samples join the shared history, so the next frame analyzes current audio.
    pub fn drain_hidden_audio(&mut self, ring: &SampleRing) {
        self.follow_sample_rate(ring);
        if self.drain_audio(ring) > 0 {
            self.audio_gap = true;
        }
    }

    /// Follow a capture sample-rate change (a new output device or format).
    /// The history restarts empty at the new rate; analyzers, the note ribbon
    /// and every rate-derived setting are rebuilt by `prepare_resources`,
    /// because the rate is part of `resource_mode`.
    fn follow_sample_rate(&mut self, ring: &SampleRing) {
        let Some(sample_rate) = ring.take_sample_rate_change() else {
            return;
        };
        // Queued samples from the previous rate were discarded.
        self.audio_gap = true;
        if sample_rate == 0 || sample_rate as f32 == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate as f32;
        let count = history_sample_count(sample_rate);
        self.left_history = vec![0.0; count];
        self.right_history = vec![0.0; count];
        self.capture_stall = CaptureStall::new(Instant::now());
    }

    /// Append newly captured samples to the history and return how many are
    /// new. A capture stall adds silence, so levels decay instead of freezing.
    fn drain_audio(&mut self, ring: &SampleRing) -> usize {
        let drained = ring.drain_latest(&mut self.left_history, &mut self.right_history);
        let history_len = self.left_history.len();
        let already_silent = self.capture_stall.filled() >= history_len as u64;
        let silent =
            self.capture_stall
                .silent_samples(drained, Instant::now(), self.sample_rate);
        if silent > 0 && !already_silent {
            append_silence(&mut self.left_history, silent);
            append_silence(&mut self.right_history, silent);
        }
        (drained + silent).min(history_len)
    }

    pub fn adjust_gain(&mut self, multiplier: f32) {
        self.settings.gain = (self.settings.gain * multiplier).clamp(0.1, 20.0);
        self.mark_settings_dirty();
        eprintln!("visual gain: {:.2}x", self.settings.gain);
    }

    fn mark_settings_dirty(&mut self) {
        self.settings_dirty = true;
        self.settings_save_deadline = Some(Instant::now() + Duration::from_millis(300));
    }

    fn save_settings_if_due(&mut self) {
        if !self.settings_dirty
            || self
                .settings_save_deadline
                .is_some_and(|deadline| Instant::now() < deadline)
        {
            return;
        }
        match self.settings.save(&self.settings_path) {
            Ok(()) => {
                self.settings_dirty = false;
                self.settings_save_deadline = None;
            }
            Err(error) => {
                eprintln!("could not save visualizer settings: {error:#}");
                self.settings_save_deadline = Some(Instant::now() + Duration::from_secs(5));
            }
        }
    }

    pub fn handle_controls_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        self.ui.on_event(window, event)
    }

    pub fn render_visualizer(
        &mut self,
        ring: &SampleRing,
        output_latency_ms: Option<f32>,
    ) -> Result<(), wgpu::SurfaceError> {
        self.output_latency_ms = output_latency_ms;
        self.follow_sample_rate(ring);
        self.save_settings_if_due();
        self.apply_pending_resize();
        self.apply_surface_history_size();
        self.prepare_resources();
        // Waiting for a drawable paces frames to the display (FIFO on Metal).
        // Wait before reading audio so the analyzed samples are as recent as
        // possible when this frame is presented, rather than aging during the wait.
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let now = Instant::now();
        let frame_delta = now.duration_since(self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;
        let drained_sample_count = self.drain_audio(ring);
        let active_channel_count = self.settings.channel_mode.channel_count();
        for channel_index in 0..active_channel_count {
            let analyzer = &mut self.analyzers[channel_index];
            match self.settings.channel_mode {
                ChannelMode::Mono => {
                    for ((output, left), right) in analyzer
                        .audio_history
                        .iter_mut()
                        .zip(&self.left_history)
                        .zip(&self.right_history)
                    {
                        *output = (*left + *right) * 0.5;
                    }
                }
                ChannelMode::Left | ChannelMode::StereoMirror if channel_index == 0 => {
                    analyzer.audio_history.copy_from_slice(&self.left_history);
                }
                ChannelMode::Right | ChannelMode::StereoMirror => {
                    analyzer.audio_history.copy_from_slice(&self.right_history);
                }
                ChannelMode::Left => unreachable!(),
            }
        }
        let requested_delay =
            (visual_delay_ms(&self.settings, self.output_latency_ms) * self.sample_rate / 1000.0)
                .round() as usize;
        let history_sample_count = self.left_history.len();
        let resonator_alignment_delay = if matches!(
            self.settings.analysis_method,
            AnalysisMethod::ResonatorBank | AnalysisMethod::HybridResonatorFft
        ) && self.settings.resonator_multirate
            && self.settings.resonator_timestamp_alignment
        {
            if self.settings.resonator_five_stage {
                105
            } else {
                49
            }
        } else {
            0
        };
        let maximum_delay =
            history_sample_count.saturating_sub(SAMPLE_COUNT + resonator_alignment_delay);
        let delay_samples = requested_delay.min(maximum_delay);
        let end = history_sample_count - delay_samples;
        for analyzer in &mut self.analyzers[..active_channel_count] {
            analyzer
                .samples
                .copy_from_slice(&analyzer.audio_history[end - SAMPLE_COUNT..end]);
        }
        let bin_count = display_bin_count(&self.settings, self.size, self.sample_rate);
        let db_floors = [self.settings.low_db_floor, self.settings.high_db_floor];
        let frequency_mapping = [
            self.settings.min_frequency_hz,
            self.settings.max_frequency_hz,
            self.settings.frequency_focus_strength_percent,
            if self.settings.bar_layout == BarLayout::MusicalNotes {
                -self.settings.frequency_focus_origin_percent
            } else {
                self.settings.frequency_focus_origin_percent
            },
        ];
        let mapping_changed =
            bin_count != self.last_bin_count || frequency_mapping != self.last_frequency_mapping;
        let method_changed = self.settings.analysis_method != self.last_analysis_method;
        let channel_mode_changed = self.settings.channel_mode != self.last_channel_mode;
        // An unanalyzed gap breaks streaming state just like a delay change.
        let delay_changed =
            delay_samples != self.last_delay_samples || std::mem::take(&mut self.audio_gap);
        if let Some(ribbon) = &mut self.note_ribbon {
            ribbon.update_audio(
                &self.queue,
                [&self.analyzers[0].samples, &self.analyzers[active_channel_count - 1].samples],
                drained_sample_count,
                self.sample_rate,
                self.settings.gain,
                [self.settings.min_frequency_hz, self.settings.max_frequency_hz],
                delay_changed || channel_mode_changed,
            );
        }
        let fft_config = multirate_config(&self.settings);
        let multirate_fft_needs_reset = self.settings.analysis_method.is_multirate_fft()
            && (self.analyzers[..active_channel_count]
                .iter()
                .any(|analyzer| analyzer.multirate_fft.requires_reset(fft_config))
                || method_changed
                || channel_mode_changed
                || delay_changed
                || (self.settings.analysis_method == AnalysisMethod::HybridResonatorFft
                    && (self.settings.resonator_five_stage != self.last_resonator_five_stage
                        || self.settings.resonator_timestamp_alignment
                            != self.last_resonator_timestamp_alignment)));
        let resonator_q = [
            self.settings.resonator_q_50_hz,
            self.settings.resonator_q_10_khz,
        ];
        let resonator_q_changed = resonator_q != self.last_resonator_q;
        let resonator_measurement_changed =
            self.settings.resonator_measurement != self.last_resonator_measurement;
        let resonator_envelope_changed =
            self.settings.resonator_envelope != self.last_resonator_envelope;
        let resonator_multirate_changed =
            self.settings.resonator_multirate != self.last_resonator_multirate;
        let resonator_five_stage_changed =
            self.settings.resonator_five_stage != self.last_resonator_five_stage;
        let resonator_timestamp_alignment_changed =
            self.settings.resonator_timestamp_alignment != self.last_resonator_timestamp_alignment;
        let hybrid_adaptive_matching_changed =
            self.settings.hybrid_adaptive_matching != self.last_hybrid_adaptive_matching;
        let auditory_filter_order_changed =
            self.settings.auditory_filter_order != self.last_auditory_filter_order;
        let auditory_display_settings = [
            self.settings.auditory_erb_bandwidth_scale,
            self.settings.auditory_compression_percent,
            self.settings.auditory_phase_sharpening_percent,
            self.settings
                .auditory_phase_sharpening_10_khz_percent
                .unwrap_or(self.settings.auditory_phase_sharpening_percent),
            self.settings.auditory_tonal_width_percent,
            self.settings
                .auditory_tonal_width_10_khz_percent
                .unwrap_or(self.settings.auditory_tonal_width_percent),
        ];
        let auditory_display_changed =
            auditory_display_settings != self.last_auditory_display_settings;
        let current_multirate_crossovers = multirate_crossovers(&self.settings);
        let multirate_crossovers_changed =
            current_multirate_crossovers != self.last_multirate_crossovers;
        let reset_envelopes = mapping_changed
            || db_floors != self.last_db_floors
            || method_changed
            || channel_mode_changed
            || delay_changed
            || resonator_q_changed
            || resonator_measurement_changed
            || resonator_envelope_changed
            || resonator_multirate_changed
            || resonator_five_stage_changed
            || resonator_timestamp_alignment_changed
            || hybrid_adaptive_matching_changed
            || auditory_filter_order_changed
            || auditory_display_changed
            || multirate_fft_needs_reset
            || multirate_crossovers_changed
            || self.settings.peak_momentum != self.last_peak_momentum;
        let reset_resonators = self.resonator_needs_reset
            || mapping_changed
            || method_changed
            || channel_mode_changed
            || delay_changed
            || resonator_q_changed
            || resonator_measurement_changed
            || resonator_envelope_changed
            || resonator_multirate_changed
            || resonator_five_stage_changed
            || resonator_timestamp_alignment_changed
            || auditory_filter_order_changed;
        let resonator_sample_count = if reset_resonators {
            if self.settings.analysis_method == AnalysisMethod::AuditoryErb {
                GAMMATONE_WARMUP_SAMPLES
            } else {
                SAMPLE_COUNT
            }
        } else {
            drained_sample_count.min(SAMPLE_COUNT)
        } as u32;
        self.last_bin_count = bin_count;
        self.last_db_floors = db_floors;
        self.last_frequency_mapping = frequency_mapping;
        self.last_analysis_method = self.settings.analysis_method;
        self.last_channel_mode = self.settings.channel_mode;
        self.last_delay_samples = delay_samples;
        self.last_resonator_q = resonator_q;
        self.last_resonator_measurement = self.settings.resonator_measurement;
        self.last_resonator_envelope = self.settings.resonator_envelope;
        self.last_resonator_multirate = self.settings.resonator_multirate;
        self.last_resonator_five_stage = self.settings.resonator_five_stage;
        self.last_resonator_timestamp_alignment = self.settings.resonator_timestamp_alignment;
        self.last_hybrid_adaptive_matching = self.settings.hybrid_adaptive_matching;
        self.last_auditory_filter_order = self.settings.auditory_filter_order;
        self.last_auditory_display_settings = auditory_display_settings;
        self.last_multirate_crossovers = current_multirate_crossovers;
        self.last_peak_momentum = self.settings.peak_momentum;

        let aligned_end = end - resonator_alignment_delay;
        let hybrid_match_config = hybrid_match_config(&self.settings, self.sample_rate);
        let mut stream_counts_per_channel = [[resonator_sample_count, 0, 0, 0, 0]; 2];
        for (channel_index, analyzer) in self.analyzers[..active_channel_count]
            .iter_mut()
            .enumerate()
        {
            let uploaded_samples = &analyzer.audio_history[aligned_end - SAMPLE_COUNT..aligned_end];
            if self.settings.analysis_method == AnalysisMethod::HybridResonatorFft
                && self.settings.hybrid_adaptive_matching
            {
                if reset_resonators
                    || multirate_fft_needs_reset
                    || analyzer
                        .hybrid_band_matcher
                        .requires_reset(hybrid_match_config)
                {
                    analyzer.hybrid_band_matcher.reset(
                        hybrid_match_config,
                        uploaded_samples,
                        self.sample_rate,
                    );
                } else {
                    let new_count = resonator_sample_count as usize;
                    analyzer.hybrid_band_matcher.ingest(
                        &uploaded_samples[SAMPLE_COUNT - new_count..],
                        self.sample_rate,
                    );
                }
            } else {
                analyzer.hybrid_band_matcher.clear();
            }

            analyzer.fft_match_elapsed += frame_delta;
            if self.settings.analysis_method.is_multirate_fft() {
                let fft_end = if self.settings.analysis_method == AnalysisMethod::HybridResonatorFft
                {
                    aligned_end
                } else {
                    end
                };
                if multirate_fft_needs_reset {
                    analyzer
                        .multirate_fft
                        .reset(fft_config, &analyzer.audio_history[..fft_end]);
                } else {
                    let new_count = drained_sample_count.min(fft_end);
                    analyzer
                        .multirate_fft
                        .ingest(&analyzer.audio_history[fft_end - new_count..fft_end]);
                }
                if let Some(magnitudes) = analyzer
                    .multirate_fft
                    .analyze_if_due(multirate_fft_needs_reset, self.settings.fft_display_divisor)
                {
                    if self.settings.analysis_method == AnalysisMethod::HybridResonatorFft
                        && self.settings.hybrid_adaptive_matching
                    {
                        analyzer.hybrid_band_matcher.update(
                            magnitudes,
                            fft_config,
                            self.sample_rate,
                            analyzer.fft_match_elapsed,
                        );
                    }
                    analyzer.fft_match_elapsed = 0.0;
                    self.queue.write_buffer(
                        &analyzer.fft_magnitude_buffer,
                        0,
                        bytemuck::cast_slice(magnitudes),
                    );
                }
            }
            if matches!(
                self.settings.analysis_method,
                AnalysisMethod::ResonatorBank
                    | AnalysisMethod::HybridResonatorFft
                    | AnalysisMethod::AuditoryErb
            ) {
                // Streaming shaders read only this suffix. A reset already
                // expands the count to the full analyzer warm-up window.
                upload_audio_samples(
                    &self.queue,
                    &analyzer.sample_buffer,
                    uploaded_samples,
                    resonator_sample_count as usize,
                );
            }

            let resonator_stream_counts = &mut stream_counts_per_channel[channel_index];
            if matches!(
                self.settings.analysis_method,
                AnalysisMethod::ResonatorBank | AnalysisMethod::HybridResonatorFft
            ) && self.settings.resonator_multirate
            {
                let new_count = resonator_sample_count as usize;
                analyzer.resonator_resampler.process(
                    &analyzer.samples[SAMPLE_COUNT - new_count..],
                    reset_resonators,
                    self.settings.resonator_five_stage,
                    self.settings.resonator_timestamp_alignment,
                );
                let streams = analyzer.resonator_resampler.streams();
                resonator_stream_counts[1] = streams[0].len() as u32;
                resonator_stream_counts[2] = streams[1].len() as u32;
                resonator_stream_counts[3] = streams[2].len() as u32;
                resonator_stream_counts[4] = streams[3].len() as u32;
                for (stream, buffer) in streams.iter().zip([
                    &analyzer.resonator_half_buffer,
                    &analyzer.resonator_quarter_buffer,
                    &analyzer.resonator_eighth_buffer,
                    &analyzer.resonator_sixteenth_buffer,
                ]) {
                    if !stream.is_empty() {
                        self.queue
                            .write_buffer(buffer, 0, bytemuck::cast_slice(stream));
                    }
                }
            }
        }
        let mut corrections = [[0.0; 4]; 2];
        for (output, analyzer) in corrections
            .iter_mut()
            .zip(&self.analyzers[..active_channel_count])
        {
            *output = analyzer.hybrid_band_matcher.corrections_db();
        }
        if active_channel_count == 2 && self.settings.hybrid_adaptive_matching {
            let (left, right) = corrections.split_at_mut(1);
            for (left_band, right_band) in left[0].iter_mut().zip(&mut right[0]) {
                let shared = (*left_band + *right_band) * 0.5;
                *left_band = shared;
                *right_band = shared;
            }
        }
        for channel_index in 0..active_channel_count {
            let analyzer = &self.analyzers[channel_index];
            let params = analysis_params(
                &self.settings,
                [
                    self.sample_rate,
                    self.started.elapsed().as_secs_f32(),
                    frame_delta,
                ],
                bin_count,
                reset_envelopes,
                (stream_counts_per_channel[channel_index], reset_resonators),
                corrections[channel_index],
                self.size,
            );
            self.queue
                .write_buffer(&analyzer.params_buffer, 0, bytemuck::bytes_of(&params));
            self.queue.write_buffer(
                &analyzer.render_channel_buffer,
                0,
                bytemuck::bytes_of(&RenderChannelParams {
                    channel_index: channel_index as f32,
                    channel_count: active_channel_count as f32,
                    bottom_mirror: if self.settings.render_mode == RenderMode::Flat
                        && self.settings.bottom_mirror
                    {
                        1.0
                    } else {
                        0.0
                    },
                    stereo_treble_center: if self.settings.stereo_treble_center {
                        1.0
                    } else {
                        0.0
                    },
                    mirror_baseline_percent: self.settings.mirror_baseline_percent,
                    mirror_initial_fade_percent: self.settings.mirror_initial_fade_percent,
                    mirror_fade_percent: self.settings.mirror_fade_percent,
                    axis_labels_overlay: if self.settings.axis_labels_overlay {
                        1.0
                    } else {
                        0.0
                    },
                    axis_label_mode: self.settings.axis_label_mode.shader_value(),
                    show_frequency_grid: if self.settings.show_frequency_grid {
                        1.0
                    } else {
                        0.0
                    },
                    bar_depth_pixels: if self.settings.render_mode == RenderMode::Surface3d {
                        self.settings.surface_bar_depth_px.unwrap_or(0.0)
                    } else {
                        0.0
                    },
                    peak_depth_brightness_percent: self
                        .settings
                        .surface_peak_depth_brightness_percent,
                    volume_stage_count: self.settings.surface_volume_stage_count as f32,
                    _padding_0: 0.0,
                    _padding_1: 0.0,
                    _padding_2: 0.0,
                }),
            );
        }
        let transparent_background = self.settings.transparent_background;
        let surface_mode = self.settings.render_mode == RenderMode::Surface3d;
        let peak_trail_enabled = surface_mode && self.settings.surface_peak_trail_enabled;
        let peak_trail_input_index = self.peak_trail_previous_index;
        let peak_trail_output_index = 1 - peak_trail_input_index;
        let reset_peak_trail = peak_trail_enabled && !self.peak_trail_was_enabled;
        let peak_trail_capture_interval = self
            .settings
            .surface_peak_trail_capture_interval_frames
            .clamp(1, 10);
        let peak_trail_capture_current = peak_trail_enabled
            && (reset_peak_trail
                || self
                    .peak_trail_capture_frame
                    .is_multiple_of(peak_trail_capture_interval));
        let surface_history_enabled = surface_mode && self.settings.surface_history_enabled;
        let surface_history_destination_row = if surface_history_enabled {
            let destination_row = self.surface_history_write_row;
            self.surface_history_write_row =
                (self.surface_history_write_row + 1) % self.surface_history_rows;
            self.surface_history_valid_rows =
                (self.surface_history_valid_rows + 1).min(self.surface_history_rows);
            Some(destination_row)
        } else {
            self.surface_history_write_row = 0;
            self.surface_history_valid_rows = 0;
            None
        };
        self.queue.write_buffer(
            &self.composite_params_buffer,
            0,
            bytemuck::bytes_of(&CompositeParams {
                transparent_background: if transparent_background { 1.0 } else { 0.0 },
                surface_mode: if surface_mode { 1.0 } else { 0.0 },
                surface_camera_height: self.settings.surface_camera_height,
                surface_camera_distance: self.settings.surface_camera_distance,
                surface_camera_fov_degrees: self.settings.surface_camera_fov_degrees,
                surface_camera_target_y: self.settings.surface_camera_target_y,
                surface_reflection_strength_percent: self
                    .settings
                    .surface_reflection_strength_percent,
                surface_reflection_fade_percent: self.settings.surface_reflection_fade_percent,
                surface_history_enabled: if surface_history_enabled { 1.0 } else { 0.0 },
                surface_history_write_row: self.surface_history_write_row as f32,
                surface_history_valid_rows: self.surface_history_valid_rows as f32,
                surface_history_fade_percent: self.settings.surface_history_fade_percent,
                surface_history_opacity_percent: self.settings.surface_history_opacity_percent,
                surface_history_grayscale: if self.settings.surface_history_grayscale {
                    1.0
                } else {
                    0.0
                },
                surface_history_brightness_percent: self
                    .settings
                    .surface_history_brightness_percent,
                surface_history_contrast_percent: self.settings.surface_history_contrast_percent,
                peak_trail_enabled: if peak_trail_enabled { 1.0 } else { 0.0 },
                peak_trail_delta_seconds: frame_delta,
                peak_trail_fade_seconds: self.settings.surface_peak_trail_fade_seconds,
                peak_trail_movement_percent: self.settings.surface_peak_trail_movement_percent,
                peak_trail_opacity_percent: self.settings.surface_peak_trail_opacity_percent,
                peak_trail_history_rows: self.surface_history_rows as f32,
                peak_trail_capture_current: if peak_trail_capture_current { 1.0 } else { 0.0 },
                _peak_trail_padding_2: 0.0,
            }),
        );
        let use_visual_target = transparent_background || surface_mode;
        let visual_target = if use_visual_target {
            &self.visual_view
        } else {
            &view
        };
        let visual_clear = if use_visual_target {
            wgpu::Color::TRANSPARENT
        } else {
            wgpu::Color::BLACK
        };
        let flat_bottom_mirror =
            self.settings.render_mode == RenderMode::Flat && self.settings.bottom_mirror;
        let rendered_bin_count = bin_count * if flat_bottom_mirror { 2 } else { 1 };
        let volume_vertex_count =
            if surface_mode && self.settings.surface_bar_depth_px.unwrap_or(0.0) > 0.0 {
                self.settings.surface_volume_stage_count.clamp(3, 8) * 6
            } else {
                6
            };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("visual-audio frame"),
            });
        self.static_grid.prepare(
            &self.device,
            &mut encoder,
            &self.analyzers[0].render_bind_group,
            self.size,
            &self.settings,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU spectrum analysis"),
                timestamp_writes: None,
            });
            for analyzer in &self.analyzers[..active_channel_count] {
                match self.settings.analysis_method {
                    AnalysisMethod::ResonatorBank => {
                        pass.set_bind_group(0, &analyzer.compute_bind_group, &[]);
                        pass.set_pipeline(
                            self.resonator_pipeline.as_ref().expect("active pipeline"),
                        );
                        pass.dispatch_workgroups(bin_count.div_ceil(64), 1, 1);
                    }
                    AnalysisMethod::HybridResonatorFft => {
                        pass.set_bind_group(0, &analyzer.compute_bind_group, &[]);
                        pass.set_pipeline(
                            self.resonator_pipeline.as_ref().expect("active pipeline"),
                        );
                        pass.dispatch_workgroups(bin_count.div_ceil(64), 1, 1);
                        pass.set_bind_group(0, &analyzer.fft_compute_bind_group, &[]);
                        pass.set_pipeline(
                            self.hybrid_fft_pipeline.as_ref().expect("active pipeline"),
                        );
                        pass.dispatch_workgroups(bin_count.div_ceil(64), 1, 1);
                    }
                    AnalysisMethod::AuditoryErb => {
                        pass.set_bind_group(0, &analyzer.compute_bind_group, &[]);
                        pass.set_pipeline(
                            self.gammatone_pipeline.as_ref().expect("active pipeline"),
                        );
                        pass.dispatch_workgroups(bin_count.div_ceil(64), 1, 1);
                    }
                    AnalysisMethod::MultirateFft | AnalysisMethod::MultirateFft5 => {
                        pass.set_bind_group(0, &analyzer.fft_compute_bind_group, &[]);
                        pass.set_pipeline(
                            self.multirate_fft_pipeline
                                .as_ref()
                                .expect("active pipeline"),
                        );
                        pass.dispatch_workgroups(bin_count.div_ceil(64), 1, 1);
                    }
                }
                pass.set_pipeline(&self.compose_display_pipeline);
                pass.dispatch_workgroups(bin_count.div_ceil(64), 1, 1);
            }
            if matches!(
                self.settings.analysis_method,
                AnalysisMethod::ResonatorBank
                    | AnalysisMethod::HybridResonatorFft
                    | AnalysisMethod::AuditoryErb
            ) {
                self.resonator_needs_reset = false;
            }
        }
        if let Some(ribbon) = &self.note_ribbon {
            ribbon.analyze(&mut encoder);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("spectrum visualization"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: visual_target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(visual_clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.static_grid
                .draw(&mut pass, &self.analyzers[0].render_bind_group);
            for analyzer in &self.analyzers[..active_channel_count] {
                pass.set_pipeline(&self.glow_pipeline);
                pass.set_bind_group(0, &analyzer.render_bind_group, &[]);
                pass.draw(0..6, 0..rendered_bin_count);
                if surface_mode && self.settings.surface_bar_depth_px.unwrap_or(0.0) > 0.0 {
                    pass.set_pipeline(self.prism_pipeline.as_ref().expect("active pipeline"));
                    pass.draw(0..volume_vertex_count, 0..rendered_bin_count);
                } else {
                    pass.set_pipeline(&self.render_pipeline);
                    pass.draw(0..6, 0..rendered_bin_count);
                }
                pass.set_pipeline(&self.peak_glow_pipeline);
                pass.draw(0..6, 0..rendered_bin_count);
                if surface_mode && self.settings.surface_bar_depth_px.unwrap_or(0.0) > 0.0 {
                    pass.set_pipeline(self.peak_volume_pipeline.as_ref().expect("active pipeline"));
                    pass.draw(0..volume_vertex_count, 0..rendered_bin_count);
                } else {
                    pass.set_pipeline(&self.peak_pipeline);
                    pass.draw(0..6, 0..rendered_bin_count);
                }
            }
            if self.settings.axis_labels_overlay {
                // The overlay font occupies only these ten physical pixel rows.
                // Scissoring also avoids launching fragments elsewhere.
                let label_y = self.size.height.saturating_sub(14);
                pass.set_scissor_rect(
                    0,
                    label_y,
                    self.size.width,
                    10.min(self.size.height - label_y),
                );
                pass.set_pipeline(
                    self.axis_overlay_pipeline
                        .as_ref()
                        .expect("active pipeline"),
                );
                pass.set_bind_group(0, &self.analyzers[0].render_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            if !use_visual_target {
                if let Some(ribbon) = &self.note_ribbon {
                    // The flat opaque view needs no composition afterwards.
                    // Draw its final overlay here to avoid another attachment
                    // store/load and render pass. Axis labels leave a scissor.
                    pass.set_scissor_rect(0, 0, self.size.width, self.size.height);
                    ribbon.draw(&mut pass);
                }
            }
        }
        if peak_trail_enabled {
            if reset_peak_trail {
                let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("clear previous peak feedback"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.peak_trail_views[peak_trail_input_index],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            if peak_trail_capture_current {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("current peak isolation"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.peak_source_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                for analyzer in &self.analyzers[..active_channel_count] {
                    pass.set_bind_group(0, &analyzer.render_bind_group, &[]);
                    pass.set_pipeline(&self.peak_glow_pipeline);
                    pass.draw(0..6, 0..rendered_bin_count);
                    // Capture one peak face only. The temporal feedback pass
                    // creates the depth trail; capturing the dyadic volume
                    // here would apply the depth fan twice.
                    pass.set_pipeline(&self.peak_pipeline);
                    pass.draw(0..6, 0..rendered_bin_count);
                }
            }
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("moving peak temporal feedback"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.peak_trail_views[peak_trail_output_index],
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(
                    self.peak_feedback_pipeline
                        .as_ref()
                        .expect("active pipeline"),
                );
                pass.set_bind_group(
                    0,
                    &self.peak_feedback_bind_groups[peak_trail_input_index],
                    &[],
                );
                pass.draw(0..3, 0..1);
            }
        }
        if let Some(destination_row) = surface_history_destination_row {
            let bottom_padding = if self.settings.axis_labels_overlay {
                0
            } else {
                PLOT_BOTTOM_PADDING
            };
            let baseline_row = self
                .config
                .height
                .saturating_sub(bottom_padding.saturating_add(1));
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.visual_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: baseline_row,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &self.surface_history_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: destination_row,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.config.width,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        if use_visual_target {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("visual surface composition"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(self.composite_pipeline.as_ref().expect("active pipeline"));
            pass.set_bind_group(0, &self.composite_bind_groups[peak_trail_input_index], &[]);
            pass.draw(0..3, 0..1);
        }
        if use_visual_target {
            if let Some(ribbon) = &self.note_ribbon {
                ribbon.render(&mut encoder, &view);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        if peak_trail_enabled {
            self.peak_trail_previous_index = peak_trail_output_index;
            self.peak_trail_capture_frame =
                (self.peak_trail_capture_frame + 1) % peak_trail_capture_interval;
        } else {
            self.peak_trail_capture_frame = 0;
        }
        self.peak_trail_was_enabled = peak_trail_enabled;
        Ok(())
    }

    pub fn render_controls(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let settings_before = self.settings.clone();
        let ui_frame = self.ui.begin_frame(
            window,
            self.controls_size,
            &mut self.settings,
            self.sample_rate,
            self.output_latency_ms,
        );
        if self.settings != settings_before {
            self.mark_settings_dirty();
        }
        let frame = self.controls_surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("visual-audio controls frame"),
            });
        for (id, delta) in &ui_frame.textures_delta.set {
            self.ui
                .renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let user_commands = self.ui.renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &ui_frame.paint_jobs,
            &ui_frame.screen,
        );
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("detached visualizer controls"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.008,
                            g: 0.012,
                            b: 0.025,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.ui.renderer.render(
                &mut pass.forget_lifetime(),
                &ui_frame.paint_jobs,
                &ui_frame.screen,
            );
        }
        self.queue
            .submit(user_commands.into_iter().chain(Some(encoder.finish())));
        frame.present();
        for id in &ui_frame.textures_delta.free {
            self.ui.renderer.free_texture(id);
        }
        Ok(())
    }
}

impl Drop for GpuState {
    fn drop(&mut self) {
        if self.settings_dirty
            && let Err(error) = self.settings.save(&self.settings_path)
        {
            eprintln!("could not save visualizer settings during shutdown: {error:#}");
        }
    }
}

fn storage_entry(
    binding: u32,
    read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn visual_target_features(settings: &VisualizerSettings) -> (bool, bool, bool) {
    let surface = settings.render_mode == RenderMode::Surface3d;
    (
        surface || settings.transparent_background,
        surface && settings.surface_history_enabled,
        surface && settings.surface_peak_trail_enabled,
    )
}

fn target_size(size: PhysicalSize<u32>, enabled: bool) -> PhysicalSize<u32> {
    if enabled {
        size
    } else {
        PhysicalSize::new(1, 1)
    }
}

fn create_visual_targets(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    composite_params_buffer: &wgpu::Buffer,
    format: wgpu::TextureFormat,
    size: PhysicalSize<u32>,
    history_rows: u32,
    history_enabled: bool,
    peak_trail_views: &[wgpu::TextureView; 2],
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    [wgpu::BindGroup; 2],
) {
    let visual_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("opaque visualizer render target"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let visual_view = visual_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let surface_history_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rendered spectrum history ring"),
        size: wgpu::Extent3d {
            width: if history_enabled {
                size.width.max(1)
            } else {
                1
            },
            height: if history_enabled {
                history_rows.clamp(MIN_SURFACE_HISTORY_ROWS, MAX_SURFACE_HISTORY_ROWS)
            } else {
                1
            },
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let surface_history_view =
        surface_history_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let peak_trail_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("composite peak trail linear sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    let bind_groups = std::array::from_fn(|index| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("visual compositor bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&visual_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: composite_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&surface_history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&peak_trail_views[index]),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&peak_trail_sampler),
                },
            ],
        })
    });
    (
        visual_texture,
        visual_view,
        surface_history_texture,
        surface_history_view,
        bind_groups,
    )
}

fn create_peak_trail_targets(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
    composite_params_buffer: &wgpu::Buffer,
    peak_trail_sampler: &wgpu::Sampler,
    format: wgpu::TextureFormat,
    size: PhysicalSize<u32>,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    [wgpu::Texture; 2],
    [wgpu::TextureView; 2],
    [wgpu::BindGroup; 2],
) {
    let descriptor = |label| wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    let peak_source_texture = device.create_texture(&descriptor("current peak render target"));
    let peak_source_view = peak_source_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let peak_trail_textures = [
        device.create_texture(&descriptor("peak feedback A")),
        device.create_texture(&descriptor("peak feedback B")),
    ];
    let peak_trail_views = peak_trail_textures
        .each_ref()
        .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()));
    let bind_groups = std::array::from_fn(|index| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("peak feedback bind group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&peak_trail_views[index]),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&peak_source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: composite_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(peak_trail_sampler),
                },
            ],
        })
    });
    (
        peak_source_texture,
        peak_source_view,
        peak_trail_textures,
        peak_trail_views,
        bind_groups,
    )
}

fn analysis_params(
    settings: &VisualizerSettings,
    timing: [f32; 3],
    bin_count: u32,
    reset_envelopes: bool,
    resonator_runtime: ([u32; 5], bool),
    hybrid_match_corrections: [f32; 4],
    size: PhysicalSize<u32>,
) -> AnalysisParams {
    let (resonator_stream, reset_resonators) = resonator_runtime;
    let fft_config = multirate_config(settings);
    let fft_crossovers = multirate_crossovers(settings);
    AnalysisParams {
        sample_rate: timing[0],
        elapsed: timing[1],
        gain: settings.gain,
        bin_count,
        resolution: [size.width as f32, size.height as f32],
        axis_focus: [
            settings.frequency_focus_strength_percent * 0.01,
            settings.frequency_focus_origin_percent * 0.01,
        ],
        perception: [
            if settings.bar_layout == BarLayout::MusicalNotes {
                1.0
            } else {
                0.0
            },
            if reset_envelopes { 1.0 } else { 0.0 },
            settings.bar_gravity_db_per_second2,
            timing[2],
        ],
        visual: [
            settings.min_brightness,
            settings.max_brightness,
            if settings.peak_momentum {
                -settings.peak_gravity_db_per_second2
            } else {
                settings.peak_gravity_db_per_second2
            },
            settings.peak_hold_ms * 0.001,
        ],
        capture: [
            0,
            resonator_stream[0],
            u32::from(reset_resonators),
            u32::from(settings.resonator_multirate),
        ],
        db_scale: [
            settings.low_db_floor,
            settings.high_db_floor,
            0.0,
            settings.bar_top_fade_percent * 0.01,
        ],
        frequency_range: [
            settings.min_frequency_hz,
            settings.max_frequency_hz,
            settings.bar_spacing_px,
            if settings.rise_highlight {
                settings.rise_highlight_strength
            } else {
                0.0
            },
        ],
        auditory_0: [
            settings.auditory_erb_bandwidth_scale,
            settings.auditory_compression_percent * 0.01,
            settings.auditory_filter_order as f32,
            settings.auditory_phase_sharpening_percent * 0.01,
        ],
        auditory_1: [
            settings.auditory_tonal_width_percent * 0.01,
            settings
                .auditory_phase_sharpening_10_khz_percent
                .unwrap_or(settings.auditory_phase_sharpening_percent)
                * 0.01,
            settings
                .auditory_tonal_width_10_khz_percent
                .unwrap_or(settings.auditory_tonal_width_percent)
                * 0.01,
            settings.rise_highlight_fade_ms * 0.001,
        ],
        fft_sizes_0: [
            fft_config.fft_sizes[0] as u32,
            fft_config.fft_sizes[1] as u32,
            fft_config.fft_sizes[2] as u32,
            fft_config.fft_sizes[3] as u32,
        ],
        fft_sizes_1: [
            fft_config.fft_sizes[4] as u32,
            fft_config.band_count as u32,
            0,
            0,
        ],
        fft_decimations_0: [
            fft_config.decimations[0] as u32,
            fft_config.decimations[1] as u32,
            fft_config.decimations[2] as u32,
            fft_config.decimations[3] as u32,
        ],
        fft_decimations_1: [fft_config.decimations[4] as u32, 0, 0, 0],
        fft_crossovers,
        resonator: [
            settings.resonator_q_50_hz,
            settings.resonator_q_10_khz,
            settings.resonator_measurement.shader_value(),
            0.0,
        ],
        resonator_decay: [
            settings.resonator_decay_db,
            settings.resonator_decay_seconds,
            settings.resonator_envelope.shader_value(),
            settings.peak_glow_full_scale_db,
        ],
        resonator_stream: resonator_stream[..4].try_into().unwrap(),
        hybrid: [
            0.0,
            0.0,
            if settings.analysis_method == AnalysisMethod::HybridResonatorFft {
                1.0
            } else {
                0.0
            },
            1.0,
        ],
        resonator_stream_extra: [
            resonator_stream[4],
            u32::from(settings.resonator_five_stage),
            quarter_energy_stride(settings, timing[0]) as u32,
            0,
        ],
        hybrid_match_corrections,
    }
}

fn hybrid_match_config(settings: &VisualizerSettings, sample_rate: f32) -> HybridMatchConfig {
    HybridMatchConfig {
        frequencies_hz: automatic_hybrid_match_frequencies(settings, sample_rate),
        q_50_hz: settings.resonator_q_50_hz,
        q_10_khz: settings.resonator_q_10_khz,
        adapt_seconds: settings.hybrid_match_adapt_seconds,
        max_correction_db: settings.hybrid_match_max_correction_db,
    }
}

fn automatic_hybrid_match_frequencies(settings: &VisualizerSettings, sample_rate: f32) -> [f32; 6] {
    let effective_max = settings.max_frequency_hz.min(sample_rate * 0.48);
    let resonator_edge = sample_rate * 0.091;
    let fft_edge = sample_rate * 0.181;
    let half_low = settings.min_frequency_hz.max(resonator_edge);
    let half_high = effective_max.min(fft_edge);
    let (low, high) = if half_high > half_low * 1.02 {
        (half_low, half_high)
    } else {
        (resonator_edge, fft_edge)
    };
    let ratio = high / low;
    [
        low * ratio.powf(0.1),
        low * ratio.powf(0.2),
        low * ratio.powf(0.3),
        fft_edge * 0.94,
        fft_edge * 0.97,
        fft_edge,
    ]
}

fn multirate_config(settings: &VisualizerSettings) -> MultirateConfig {
    if settings.analysis_method == AnalysisMethod::HybridResonatorFft {
        MultirateConfig {
            band_count: 2,
            fft_sizes: [
                settings.hybrid_half_fft_size as usize,
                settings.hybrid_full_fft_size as usize,
                0,
                0,
                0,
            ],
            decimations: [2, 1, 1, 1, 1],
            window_kind: settings.hybrid_fft_window.shader_value() as u32,
            kaiser_beta: settings.asymmetric_kaiser_beta,
            flanagan_alpha: settings.flanagan_alpha,
        }
    } else if settings.analysis_method == AnalysisMethod::MultirateFft5 {
        MultirateConfig {
            band_count: 5,
            fft_sizes: settings.multirate_fft5_sizes.map(|value| value as usize),
            decimations: settings
                .multirate_fft5_decimations
                .map(|value| value as usize),
            window_kind: settings.multirate_fft5_window.shader_value() as u32,
            kaiser_beta: settings.asymmetric_kaiser_beta,
            flanagan_alpha: settings.flanagan_alpha,
        }
    } else {
        MultirateConfig {
            band_count: 3,
            fft_sizes: [
                settings.multirate_fft_sizes[0] as usize,
                settings.multirate_fft_sizes[1] as usize,
                settings.multirate_fft_sizes[2] as usize,
                0,
                0,
            ],
            decimations: [
                settings.multirate_fft_decimations[0] as usize,
                settings.multirate_fft_decimations[1] as usize,
                settings.multirate_fft_decimations[2] as usize,
                1,
                1,
            ],
            window_kind: settings.multirate_fft_window.shader_value() as u32,
            kaiser_beta: settings.asymmetric_kaiser_beta,
            flanagan_alpha: settings.flanagan_alpha,
        }
    }
}

fn quarter_energy_stride(settings: &VisualizerSettings, sample_rate: f32) -> usize {
    if !matches!(
        settings.resonator_measurement,
        ResonatorMeasurement::PositionVelocityPi2 | ResonatorMeasurement::PositionPi2
    ) || !matches!(
        settings.analysis_method,
        AnalysisMethod::ResonatorBank | AnalysisMethod::HybridResonatorFft
    ) {
        return 0;
    }
    let frequency = settings.min_frequency_hz.max(16.0);
    let divisor = if !settings.resonator_multirate {
        1.0
    } else if settings.resonator_five_stage && frequency <= sample_rate * 0.0205 {
        16.0
    } else if frequency <= sample_rate * 0.041 {
        8.0
    } else if frequency <= sample_rate * 0.091 {
        4.0
    } else if frequency <= sample_rate * 0.181 {
        2.0
    } else {
        1.0
    };
    (sample_rate / (4.0 * frequency * divisor)).ceil() as usize + 2
}

/// Everything the analyzer buffers are sized from; a change rebuilds them.
type ResourceMode = (AnalysisMethod, bool, bool, usize, usize, u32);

fn resource_mode(settings: &VisualizerSettings, sample_rate: f32) -> ResourceMode {
    (
        settings.analysis_method,
        settings.resonator_multirate,
        settings.resonator_five_stage,
        fft_magnitude_count(settings),
        quarter_energy_stride(settings, sample_rate),
        sample_rate as u32,
    )
}

/// Shared capture history: at least one second beyond the analyzer block, so
/// the output delay (up to one second) always has samples behind it.
fn history_sample_count(sample_rate: u32) -> usize {
    HISTORY_SAMPLE_COUNT.max(SAMPLE_COUNT + sample_rate as usize)
}

/// Shift `count` zero samples into the end of a history.
fn append_silence(history: &mut [f32], count: usize) {
    let count = count.min(history.len());
    history.copy_within(count.., 0);
    let start = history.len() - count;
    history[start..].fill(0.0);
}

fn fft_magnitude_count(settings: &VisualizerSettings) -> usize {
    if !settings.analysis_method.is_multirate_fft() {
        return 1;
    }
    let config = multirate_config(settings);
    config.fft_sizes[..config.band_count]
        .iter()
        .map(|size| size / 2 + 1)
        .sum()
}

fn multirate_crossovers(settings: &VisualizerSettings) -> [f32; 4] {
    if settings.analysis_method == AnalysisMethod::HybridResonatorFft {
        [0.0; 4]
    } else if settings.analysis_method == AnalysisMethod::MultirateFft5 {
        settings.multirate_fft5_crossovers_hz
    } else {
        [
            settings.multirate_fft_crossovers_hz[0],
            settings.multirate_fft_crossovers_hz[1],
            0.0,
            0.0,
        ]
    }
}

fn display_bin_count(
    settings: &VisualizerSettings,
    size: PhysicalSize<u32>,
    sample_rate: f32,
) -> u32 {
    if settings.bar_layout == BarLayout::MusicalNotes {
        let lower_midi = 69.0 + 12.0 * (settings.min_frequency_hz / 440.0).log2();
        let effective_upper = settings.max_frequency_hz.min(sample_rate * 0.48);
        let upper_midi = 69.0 + 12.0 * (effective_upper / 440.0).log2();
        let first_midi = (lower_midi - 1.0e-4).ceil();
        return (upper_midi - first_midi)
            .ceil()
            .clamp(1.0, MAX_BIN_COUNT as f32) as u32;
    }
    let plot_width = (size.width as f32 - PLOT_HORIZONTAL_MARGIN).max(1.0)
        / settings.channel_mode.channel_count() as f32;
    (plot_width / (settings.bar_width_px + settings.bar_spacing_px))
        .floor()
        .clamp(32.0, MAX_BIN_COUNT as f32) as u32
}

fn analyzer_layouts(device: &wgpu::Device) -> (wgpu::BindGroupLayout, wgpu::BindGroupLayout) {
    let compute_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("audio compute layout"),
            entries: &[
                storage_entry(0, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, false, wgpu::ShaderStages::COMPUTE),
                uniform_entry(2, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(4, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(5, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(6, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(7, false, wgpu::ShaderStages::COMPUTE),
                storage_entry(8, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
    let render_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("audio render layout"),
            entries: &[
                storage_entry(7, true, wgpu::ShaderStages::VERTEX),
                storage_entry(8, true, wgpu::ShaderStages::VERTEX),
                uniform_entry(2, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                uniform_entry(9, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
            ],
        });
    (compute_bind_group_layout, render_bind_group_layout)
}

fn create_analyzer_channel(
    device: &wgpu::Device,
    compute_layout: &wgpu::BindGroupLayout,
    render_layout: &wgpu::BindGroupLayout,
    initial_params: &AnalysisParams,
    history_sample_count: usize,
    channel_index: usize,
    channel_count: f32,
    settings: &VisualizerSettings,
) -> AnalyzerChannel {
    let prefix = format!("channel {channel_index}");
    let uses_resampler = matches!(
        settings.analysis_method,
        AnalysisMethod::ResonatorBank | AnalysisMethod::HybridResonatorFft
    ) && settings.resonator_multirate;
    let stream_size = |size| if uses_resampler { size } else { 1 };
    let uses_samples = !matches!(
        settings.analysis_method,
        AnalysisMethod::MultirateFft | AnalysisMethod::MultirateFft5
    );

    let sample_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{prefix} audio samples")),
        size: ((if uses_samples { SAMPLE_COUNT } else { 1 }) * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fft_magnitude_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{prefix} FFT magnitudes")),
        size: ((fft_magnitude_count(settings)) * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let spectrum_buffer = storage_buffer(
        device,
        &format!("{prefix} spectrum bins"),
        MAX_BIN_COUNT as u64 * 4,
    );
    let display_spectrum_buffer = storage_buffer(
        device,
        &format!("{prefix} display spectrum bins"),
        MAX_BIN_COUNT as u64 * 4,
    );
    let resonator_half_buffer =
        storage_buffer(device, &format!("{prefix} /2 samples"), stream_size(8192));
    let resonator_quarter_buffer =
        storage_buffer(device, &format!("{prefix} /4 samples"), stream_size(4096));
    let resonator_eighth_buffer =
        storage_buffer(device, &format!("{prefix} /8 samples"), stream_size(2048));
    let resonator_sixteenth_buffer = storage_buffer(
        device,
        &format!("{prefix} /16 samples"),
        if uses_resampler && settings.resonator_five_stage {
            1024
        } else {
            1
        },
    );
    let resonator_state_buffer = storage_buffer(
        device,
        &format!("{prefix} analysis auxiliary state"),
        MAX_BIN_COUNT as u64 * 16
            + (quarter_energy_stride(settings, initial_params.sample_rate) as u64
                * MAX_BIN_COUNT as u64)
                .max(4),
    );
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{prefix} analysis parameters")),
        contents: bytemuck::bytes_of(initial_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let render_channel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{prefix} render channel parameters")),
        contents: bytemuck::bytes_of(&RenderChannelParams {
            channel_index: channel_index as f32,
            channel_count,
            bottom_mirror: 0.0,
            stereo_treble_center: 0.0,
            mirror_baseline_percent: 50.0,
            mirror_initial_fade_percent: 0.0,
            mirror_fade_percent: 0.0,
            axis_labels_overlay: 0.0,
            axis_label_mode: 0.0,
            show_frequency_grid: 1.0,
            bar_depth_pixels: 0.0,
            peak_depth_brightness_percent: 0.0,
            volume_stage_count: 3.0,
            _padding_0: 0.0,
            _padding_1: 0.0,
            _padding_2: 0.0,
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let compute_resources = [
        &spectrum_buffer,
        &params_buffer,
        &resonator_quarter_buffer,
        &resonator_eighth_buffer,
        &resonator_half_buffer,
        &resonator_sixteenth_buffer,
        &display_spectrum_buffer,
        &resonator_state_buffer,
    ];
    let compute_bind_group = create_compute_channel_bind_group(
        device,
        compute_layout,
        &format!("{prefix} compute bindings"),
        &sample_buffer,
        compute_resources,
    );
    let fft_compute_bind_group = create_compute_channel_bind_group(
        device,
        compute_layout,
        &format!("{prefix} FFT compute bindings"),
        &fft_magnitude_buffer,
        compute_resources,
    );
    let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{prefix} render bindings")),
        layout: render_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 7,
                resource: display_spectrum_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: resonator_state_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: render_channel_buffer.as_entire_binding(),
            },
        ],
    });

    AnalyzerChannel {
        compute_bind_group,
        fft_compute_bind_group,
        render_bind_group,
        sample_buffer,
        fft_magnitude_buffer,
        spectrum_buffer,
        resonator_half_buffer,
        resonator_quarter_buffer,
        resonator_eighth_buffer,
        resonator_sixteenth_buffer,
        params_buffer,
        render_channel_buffer,
        multirate_fft: MultirateFft::default(),
        hybrid_band_matcher: HybridBandMatcher::default(),
        fft_match_elapsed: 0.0,
        resonator_resampler: ResonatorResampler::default(),
        audio_history: vec![0.0; history_sample_count],
        samples: vec![0.0; SAMPLE_COUNT],
    }
}

fn create_compute_channel_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    input: &wgpu::Buffer,
    resources: [&wgpu::Buffer; 8],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: resources[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: resources[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: resources[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: resources[3].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: resources[4].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: resources[5].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: resources[6].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: resources[7].as_entire_binding(),
            },
        ],
    })
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn upload_audio_samples(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    samples: &[f32],
    new_count: usize,
) {
    if new_count > 0 {
        let start = samples.len() - new_count;
        queue.write_buffer(
            buffer,
            (start * std::mem::size_of::<f32>()) as u64,
            bytemuck::cast_slice(&samples[start..]),
        );
    }
}

fn storage_buffer(device: &wgpu::Device, label: &str, samples: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: samples * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn build_resonator_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("resonator.wgsl"));
    compute_pipeline(device, layout, &shader, "resonator_pipeline", "resonate")
}

fn build_gammatone_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("gammatone.wgsl"));
    compute_pipeline(device, layout, &shader, "gammatone_pipeline", "analyze_erb")
}

fn build_multirate_fft_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("multirate_fft.wgsl"));
    compute_pipeline(device, layout, &shader, "multirate_fft_pipeline", "map_fft")
}

fn build_hybrid_fft_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("multirate_fft.wgsl"));
    compute_pipeline(device, layout, &shader, "hybrid_fft_pipeline", "map_hybrid")
}

fn build_axis_overlay_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("visualizer.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("frequency axis label overlay"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("grid_vs"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("axis_overlay_fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn build_prism_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("visualizer.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("dyadic layered spectrum volume"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("prism_vs"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("prism_fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn build_peak_volume_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("visualizer.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("dyadic layered peak indicators"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("peak_volume_vs"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("peak_fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn build_composite_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("composite.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("luminance-to-alpha compositor"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

fn build_peak_feedback_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("peak_feedback.wgsl"));
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("moving peak temporal feedback"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use winit::dpi::PhysicalSize;

    #[test]
    fn removed_settings_migrate_without_losing_active_settings() {
        let old = r#"{"analysis_method":"WindowedDft","gain":2.4,"bottom_mirror":true,
            "temporal_smoothing":true,"frequency_smoothing_strength_percent":90.0,
            "resonator_stage_count":2,"resonator_transient_assist_percent":70.0,
            "hybrid_half_fft_gain_db":9.0,"fft_sync_to_display":false,
            "hybrid_analysis_rate_hz":60.0}"#;
        let settings = serde_json::from_str::<VisualizerSettings>(old)
            .unwrap()
            .sanitized();
        assert_eq!(settings.analysis_method, AnalysisMethod::ResonatorBank);
        assert_eq!(settings.gain, 2.4);
        assert!(settings.bottom_mirror);
        assert_eq!(settings.fft_display_divisor, 1);
        let saved = serde_json::to_value(&settings).unwrap();
        for field in [
            "temporal_smoothing",
            "frequency_smoothing_strength_percent",
            "resonator_stage_count",
            "resonator_transient_assist_percent",
            "hybrid_half_fft_gain_db",
            "fft_sync_to_display",
            "hybrid_analysis_rate_hz",
        ] {
            assert!(saved.get(field).is_none(), "obsolete field {field}");
        }
        let invalid = VisualizerSettings {
            fft_display_divisor: 99,
            ..settings
        };
        assert_eq!(invalid.sanitized().fft_display_divisor, 4);
    }

    #[test]
    fn quarter_cycle_history_is_allocated_only_for_selected_resonators() {
        let mut settings = VisualizerSettings::default();
        assert_eq!(quarter_energy_stride(&settings, 48000.0), 0);
        settings.resonator_measurement = ResonatorMeasurement::PositionVelocityPi2;
        assert_eq!(quarter_energy_stride(&settings, 48000.0), 49);
        settings.resonator_measurement = ResonatorMeasurement::PositionPi2;
        assert_eq!(quarter_energy_stride(&settings, 48000.0), 49);
        settings.resonator_multirate = false;
        assert_eq!(quarter_energy_stride(&settings, 48000.0), 752);
        settings.analysis_method = AnalysisMethod::MultirateFft;
        assert_eq!(quarter_energy_stride(&settings, 48000.0), 0);
    }

    #[test]
    fn visual_resources_follow_enabled_effects() {
        let mut settings = VisualizerSettings::default();
        assert_eq!(visual_target_features(&settings), (false, false, false));
        settings.surface_history_enabled = true;
        settings.surface_peak_trail_enabled = true;
        assert_eq!(visual_target_features(&settings), (false, false, false));
        settings.transparent_background = true;
        assert_eq!(visual_target_features(&settings), (true, false, false));
        settings.render_mode = RenderMode::Surface3d;
        assert_eq!(visual_target_features(&settings), (true, true, true));
        assert_eq!(
            target_size(PhysicalSize::new(1920, 1080), false),
            PhysicalSize::new(1, 1)
        );
    }

    #[test]
    #[ignore = "requires a Metal GPU"]
    fn gpu_analyzers_validate_with_lazy_buffers() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                ..Default::default()
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .unwrap();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .unwrap();
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let (compute_layout, render_layout) = analyzer_layouts(&device);
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&compute_layout],
                push_constant_ranges: &[],
            });
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&render_layout],
                    push_constant_ranges: &[],
                });
            let shader = device.create_shader_module(wgpu::include_wgsl!("analysis.wgsl"));
            let compose = compute_pipeline(
                &device,
                &pipeline_layout,
                &shader,
                "compose",
                "compose_display",
            );
            // Capture follows output changes, so analyzers are rebuilt at each rate.
            for (sample_rate, method) in [44_100, 48_000, 96_000].into_iter().flat_map(|rate| {
                [
                    AnalysisMethod::ResonatorBank,
                    AnalysisMethod::HybridResonatorFft,
                    AnalysisMethod::MultirateFft,
                    AnalysisMethod::MultirateFft5,
                    AnalysisMethod::AuditoryErb,
                ]
                .map(|method| (rate, method))
            }) {
                for multirate in [false, true] {
                    for five_stage in [false, true] {
                        let settings = VisualizerSettings {
                            analysis_method: method,
                            resonator_multirate: multirate,
                            resonator_measurement: if five_stage && multirate {
                                ResonatorMeasurement::PositionVelocityPi2
                            } else if five_stage {
                                ResonatorMeasurement::PositionPi2
                            } else {
                                ResonatorMeasurement::ImageRejected
                            },
                            resonator_five_stage: five_stage,
                            ..VisualizerSettings::default()
                        };
                        let params = analysis_params(
                            &settings,
                            [sample_rate as f32, 0.0, 1.0 / 60.0],
                            16,
                            true,
                            ([0; 5], true),
                            [0.0; 4],
                            PhysicalSize::new(320, 180),
                        );
                        let analyzer = create_analyzer_channel(
                            &device,
                            &compute_layout,
                            &render_layout,
                            &params,
                            history_sample_count(sample_rate),
                            0,
                            1.0,
                            &settings,
                        );
                        assert_eq!(
                            analyzer.fft_magnitude_buffer.size() > 4,
                            method.is_multirate_fft()
                        );
                        assert_eq!(
                            analyzer.resonator_half_buffer.size() > 4,
                            multirate
                                && matches!(
                                    method,
                                    AnalysisMethod::ResonatorBank
                                        | AnalysisMethod::HybridResonatorFft
                                )
                        );
                        let pipeline = match method {
                            AnalysisMethod::ResonatorBank | AnalysisMethod::HybridResonatorFft => {
                                build_resonator_pipeline(&device, &pipeline_layout)
                            }
                            AnalysisMethod::AuditoryErb => {
                                build_gammatone_pipeline(&device, &pipeline_layout)
                            }
                            _ => build_multirate_fft_pipeline(&device, &pipeline_layout),
                        };
                        let hybrid = (method == AnalysisMethod::HybridResonatorFft)
                            .then(|| build_hybrid_fft_pipeline(&device, &pipeline_layout));
                        let mut encoder = device.create_command_encoder(&Default::default());
                        {
                            let mut pass = encoder.begin_compute_pass(&Default::default());
                            pass.set_pipeline(&pipeline);
                            pass.set_bind_group(
                                0,
                                if matches!(
                                    method,
                                    AnalysisMethod::MultirateFft | AnalysisMethod::MultirateFft5
                                ) {
                                    &analyzer.fft_compute_bind_group
                                } else {
                                    &analyzer.compute_bind_group
                                },
                                &[],
                            );
                            pass.dispatch_workgroups(1, 1, 1);
                            if let Some(hybrid) = &hybrid {
                                pass.set_pipeline(hybrid);
                                pass.set_bind_group(0, &analyzer.fft_compute_bind_group, &[]);
                                pass.dispatch_workgroups(1, 1, 1);
                            }
                            pass.set_pipeline(&compose);
                            pass.set_bind_group(0, &analyzer.compute_bind_group, &[]);
                            pass.dispatch_workgroups(1, 1, 1);
                        }
                        queue.submit([encoder.finish()]);
                    }
                }
            }
            let format = wgpu::TextureFormat::Bgra8UnormSrgb;
            let _ = build_prism_pipeline(&device, &render_pipeline_layout, format);
            let _ = build_peak_volume_pipeline(&device, &render_pipeline_layout, format);
            let _ = build_axis_overlay_pipeline(&device, &render_pipeline_layout, format);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let error = device.pop_error_scope().await;
            assert!(error.is_none(), "GPU validation failed: {error:?}");
        });
    }

    #[test]
    fn analyzer_resources_follow_the_capture_sample_rate() {
        let settings = VisualizerSettings::default();
        assert_ne!(
            resource_mode(&settings, 44_100.0),
            resource_mode(&settings, 48_000.0)
        );
        assert_eq!(
            resource_mode(&settings, 48_000.0),
            resource_mode(&settings, 48_000.0)
        );
        assert_eq!(history_sample_count(44_100), HISTORY_SAMPLE_COUNT);
        assert_eq!(history_sample_count(48_000), HISTORY_SAMPLE_COUNT);
        for rate in [8_000, 16_000, 44_100, 48_000, 88_200, 96_000, 192_000] {
            // The full one-second output delay stays available at every rate.
            assert!(history_sample_count(rate) >= SAMPLE_COUNT + rate as usize);
        }
    }

    #[test]
    fn appended_silence_shifts_the_history() {
        let mut history = [1.0, 2.0, 3.0, 4.0];
        append_silence(&mut history, 1);
        assert_eq!(history, [2.0, 3.0, 4.0, 0.0]);
        append_silence(&mut history, 0);
        assert_eq!(history, [2.0, 3.0, 4.0, 0.0]);
        append_silence(&mut history, 10);
        assert_eq!(history, [0.0; 4]);
    }

    #[test]
    fn settings_json_round_trip() {
        let expected = VisualizerSettings::default();
        let json = serde_json::to_string(&expected).unwrap();
        let actual: VisualizerSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn output_sync_combines_reported_latency_and_signed_adjustment() {
        let mut settings = VisualizerSettings::default();
        assert_eq!(visual_delay_ms(&settings, Some(240.0)), 240.0);
        settings.visualization_delay_ms = -40.0;
        assert_eq!(visual_delay_ms(&settings, Some(240.0)), 200.0);
        assert_eq!(visual_delay_ms(&settings, Some(10.0)), 0.0);
        settings.visualization_delay_ms = 900.0;
        assert_eq!(visual_delay_ms(&settings, Some(240.0)), 1000.0);
        settings.auto_output_sync = false;
        assert_eq!(visual_delay_ms(&settings, Some(240.0)), 900.0);
        settings.auto_output_sync = true;
        assert_eq!(visual_delay_ms(&settings, None), 900.0);
    }

    #[test]
    fn existing_settings_enable_output_sync_without_losing_manual_delay() {
        let settings: VisualizerSettings =
            serde_json::from_str(r#"{"visualization_delay_ms":150.0}"#).unwrap();
        assert!(settings.auto_output_sync);
        assert_eq!(visual_delay_ms(&settings, Some(100.0)), 250.0);
    }

    #[test]
    fn legacy_hybrid_fft_size_migrates_to_half_rate_stage() {
        let actual: VisualizerSettings =
            serde_json::from_str(r#"{"hybrid_fft_size":1024}"#).unwrap();
        assert_eq!(actual.hybrid_half_fft_size, 1024);
        assert_eq!(actual.hybrid_full_fft_size, 512);
    }

    #[test]
    fn percentage_bar_depth_migrates_to_fixed_pixels() {
        let actual: VisualizerSettings =
            serde_json::from_str(r#"{"bar_width_px":6.0,"surface_bar_depth_percent":200.0}"#)
                .unwrap();
        let actual = actual.sanitized();
        assert_eq!(actual.surface_bar_depth_px, Some(12.0));
    }

    #[test]
    fn musical_note_layout_has_twelve_bars_per_octave() {
        let settings = VisualizerSettings {
            bar_layout: BarLayout::MusicalNotes,
            min_frequency_hz: 130.812_79, // C3
            max_frequency_hz: 261.625_58, // C4, exclusive
            ..VisualizerSettings::default()
        };
        assert_eq!(
            display_bin_count(&settings, PhysicalSize::new(1600, 900), 48_000.0),
            12
        );
    }

    #[test]
    fn wgsl_shaders_parse() {
        for (name, source) in [
            ("analysis.wgsl", include_str!("analysis.wgsl")),
            ("resonator.wgsl", include_str!("resonator.wgsl")),
            ("gammatone.wgsl", include_str!("gammatone.wgsl")),
            ("multirate_fft.wgsl", include_str!("multirate_fft.wgsl")),
            ("visualizer.wgsl", include_str!("visualizer.wgsl")),
            ("composite.wgsl", include_str!("composite.wgsl")),
            ("peak_feedback.wgsl", include_str!("peak_feedback.wgsl")),
            ("note_activity.wgsl", include_str!("note_activity.wgsl")),
            ("note_ribbon.wgsl", include_str!("note_ribbon.wgsl")),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name} failed to parse: {error}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name} failed validation: {error}"));
        }
    }
}

#[cfg(test)]
#[path = "resonator_measurement_tests.rs"]
mod resonator_measurement_tests;

#[cfg(test)]
#[path = "gpu_performance_tests.rs"]
mod gpu_performance_tests;
