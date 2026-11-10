use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use egui::{Color32, Pos2, Rect, Stroke};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use visual_audio::{
    audio::AudioInput,
    waveform::{ReferenceSweep, WaveformHistory, beating_sample, tone_sample},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

struct Graphics {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    context: egui::Context,
    input: egui_winit::State,
    renderer: Renderer,
}

impl Graphics {
    async fn new(window: Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .context("no compatible graphics adapter")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("no compatible window format")?;
        let caps = surface.get_capabilities(&adapter);
        if let Some(format) = caps.formats.iter().find(|format| format.is_srgb()) {
            config.format = *format;
        }
        config.present_mode = wgpu::PresentMode::Fifo;
        surface.configure(&device, &config);
        let context = egui::Context::default();
        context.set_visuals(egui::Visuals::dark());
        let input = egui_winit::State::new(
            context.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = Renderer::new(&device, config.format, RendererOptions::default());
        Ok(Self {
            surface,
            device,
            queue,
            config,
            context,
            input,
            renderer,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self, window: &Window, model: &mut Model) -> Result<(), wgpu::SurfaceError> {
        // Acquire before running egui, so a lost surface cannot discard texture updates.
        let frame = self.surface.get_current_texture()?;
        let output = self
            .context
            .run(self.input.take_egui_input(window), |context| {
                model.ui(context)
            });
        self.input
            .handle_platform_output(window, output.platform_output);
        let jobs = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);
        let screen = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: output.pixels_per_point,
        };
        for (id, delta) in &output.textures_delta.set {
            self.renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let commands =
            self.renderer
                .update_buffers(&self.device, &self.queue, &mut encoder, &jobs, &screen);
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("WaveAudio"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.018,
                            g: 0.025,
                            b: 0.04,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            self.renderer
                .render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        self.queue
            .submit(commands.into_iter().chain(Some(encoder.finish())));
        frame.present();
        for id in output.textures_delta.free {
            self.renderer.free_texture(&id);
        }
        Ok(())
    }
}

struct Model {
    history: WaveformHistory,
    sample_rate: u32,
    min_hz: f64,
    max_hz: f64,
    logarithmic: bool,
    phase_lock: bool,
    target_fps: u32,
    right_channel: bool,
    gain: f32,
    auto_sync: bool,
    adjustment_ms: f32,
    latency_ms: Option<f32>,
    demo: bool,
    demo_beats: bool,
    demo_hz: f64,
}

impl Model {
    fn new(sample_rate: u32, demo: bool) -> Self {
        Self {
            history: WaveformHistory::new(sample_rate),
            sample_rate,
            min_hz: 60.0_f64.min((sample_rate as f64 * 0.45).max(1.0)),
            max_hz: 6000.0_f64.min((sample_rate as f64 * 0.45).max(1.0)),
            logarithmic: true,
            phase_lock: false,
            target_fps: 120,
            right_channel: false,
            gain: 1.0,
            auto_sync: true,
            adjustment_ms: 0.0,
            latency_ms: None,
            demo,
            demo_beats: false,
            demo_hz: 60.0,
        }
    }

    /// Follow a capture rate change (a new output device or format): the
    /// history restarts and the reference range stays below Nyquist.
    fn set_sample_rate(&mut self, sample_rate: u32) {
        if sample_rate == 0 || sample_rate == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        self.history = WaveformHistory::new(sample_rate);
        let limit = (sample_rate as f64 * 0.45).min(20_000.0).max(1.0);
        self.max_hz = self.max_hz.min(limit);
        self.min_hz = self.min_hz.min(self.max_hz);
    }

    fn ui(&mut self, context: &egui::Context) {
        let previous_channel = self.right_channel;
        let max_hz = (self.sample_rate as f64 * 0.45).min(20_000.0).max(1.0);
        egui::TopBottomPanel::top("controls").show(context, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("WaveAudio");
                ui.separator();
                if self.demo {
                    if self.demo_beats { ui.label("Demo: 440 + 442 Hz"); }
                    else { ui.label(format!("Demo: {:.2} Hz", self.demo_hz)); }
                } else {
                    ui.selectable_value(&mut self.right_channel, false, "Left");
                    ui.selectable_value(&mut self.right_channel, true, "Right");
                }
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Reference F: Min");
                ui.add(egui::DragValue::new(&mut self.min_hz).range(1.0..=max_hz).speed(0.1).min_decimals(2).max_decimals(4).suffix(" Hz"));
                self.max_hz = self.max_hz.clamp(self.min_hz, max_hz);
                ui.label("Max");
                ui.add(egui::DragValue::new(&mut self.max_hz).range(self.min_hz..=max_hz).speed(0.1).min_decimals(2).max_decimals(4).suffix(" Hz"));
                ui.checkbox(&mut self.logarithmic, "Log spacing");
            });
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.phase_lock, "Lock phase to Min")
                    .on_hover_text("Use one shared read endpoint aligned to the Min period. Individual points use F(x) only for their delay, keeping the curve continuous. This does not lock every frequency at once.");
                ui.add(egui::Slider::new(&mut self.gain, 0.1..=10.0).logarithmic(true).text("gain"));
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Target FPS");
                ui.add(egui::DragValue::new(&mut self.target_fps).range(30..=240).speed(1.0));
                ui.small("Actual presentation depends on the display; audio time sets the window.");
            });
            if !self.demo {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut self.auto_sync, "Follow output delay");
                    ui.add(egui::Slider::new(&mut self.adjustment_ms, -500.0..=1000.0).text("adjustment (ms)"));
                    match self.latency_ms {
                        Some(ms) => { ui.label(format!("Output: {ms:.0} ms")); }
                        None => { ui.label("Output delay unavailable"); }
                    }
                });
            }
            ui.small("One line: Reference F increases from Min at left to Max at right. All points read the same raw audio.");
        });
        if self.right_channel != previous_channel {
            self.history = WaveformHistory::new(self.sample_rate);
        }
        let delay_ms = if self.demo {
            0.0
        } else {
            ((if self.auto_sync {
                self.latency_ms.unwrap_or(0.0)
            } else {
                0.0
            }) + self.adjustment_ms)
                .clamp(0.0, 1500.0)
        };
        let available_end = self.history.sample_count as f64
            - 1.0
            - delay_ms as f64 * self.sample_rate as f64 / 1000.0;
        egui::CentralPanel::default().show(context, |ui| {
            let height = (ui.available_height() - 24.0).max(30.0);
            let sweep = self.history.sweep(
                available_end,
                self.min_hz,
                self.max_hz,
                self.phase_lock,
                self.logarithmic,
            );
            self.wave(ui, sweep, height);
            ui.small(if self.phase_lock {
                "Shared phase anchor at Min / same audio, varying timebase / newest at right"
            } else {
                "Continuous scrolling / same audio, varying timebase / newest at right"
            });
        });
    }

    fn wave(&self, ui: &mut egui::Ui, sweep: ReferenceSweep, height: f32) {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), height),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::from_rgb(13, 20, 30));
        let plot = Rect::from_min_max(
            Pos2::new(rect.left() + 8.0, rect.top() + 4.0),
            Pos2::new(rect.right() - 8.0, rect.bottom() - 24.0),
        );
        if plot.width() < 1.0 || !ui.is_rect_visible(rect) {
            return;
        }
        for division in 0..=8 {
            let x = plot.left() + plot.width() * division as f32 / 8.0;
            painter.line_segment(
                [Pos2::new(x, plot.top()), Pos2::new(x, plot.bottom())],
                Stroke::new(0.5, Color32::from_gray(35)),
            );
            let frequency = sweep.frequency_at(division as f64 / 8.0);
            let label = if frequency >= 1000.0 {
                format!("{:.2}k", frequency / 1000.0)
            } else {
                format!("{frequency:.0} Hz")
            };
            let alignment = if division == 0 {
                egui::Align2::LEFT_BOTTOM
            } else if division == 8 {
                egui::Align2::RIGHT_BOTTOM
            } else {
                egui::Align2::CENTER_BOTTOM
            };
            painter.text(
                Pos2::new(x, rect.bottom() - 3.0),
                alignment,
                label,
                egui::FontId::proportional(11.0),
                Color32::from_gray(160),
            );
        }
        painter.line_segment(
            [
                Pos2::new(plot.left(), plot.center().y),
                Pos2::new(plot.right(), plot.center().y),
            ],
            Stroke::new(0.5, Color32::from_gray(65)),
        );
        let painter = painter.with_clip_rect(plot);
        let columns = plot.width().ceil().clamp(1.0, 1800.0) as usize;
        let scale = plot.height() * 0.44 * self.gain;
        let y = |value: f32| plot.center().y - value * scale;
        let color = Color32::from_rgb(84, 220, 209);
        let mut line = Vec::new();
        for point in 0..=columns {
            let fraction = point as f64 / columns as f64;
            let position = sweep.sample_position(fraction);
            if let Some(value) = self.history.raw.interpolate(position) {
                let x = plot.left() + plot.width() * fraction as f32;
                line.push(Pos2::new(x, y(value)));
            } else if line.len() > 1 {
                painter.add(egui::Shape::line(
                    std::mem::take(&mut line),
                    Stroke::new(1.0, color),
                ));
            } else {
                line.clear();
            }
            if point < columns {
                let next = sweep.sample_position((point + 1) as f64 / columns as f64);
                if next - position > 1.0 {
                    if let Some((low, high)) = self.history.raw.segment_range(position, next) {
                        let x = plot.left() + plot.width() * fraction as f32;
                        painter.line_segment(
                            [Pos2::new(x, y(low)), Pos2::new(x, y(high))],
                            Stroke::new(1.0, color),
                        );
                    }
                }
            }
        }
        if line.len() > 1 {
            painter.add(egui::Shape::line(line, Stroke::new(1.0, color)));
        }
    }
}

struct App {
    audio: Option<AudioInput>,
    model: Model,
    left: Vec<f32>,
    right: Vec<f32>,
    demo_started: Instant,
    demo_index: u64,
    window: Option<Arc<Window>>,
    graphics: Option<Graphics>,
    next_frame: Instant,
    minimized: bool,
}

impl App {
    fn ingest(&mut self) {
        if let Some(audio) = &mut self.audio {
            self.model.latency_ms = audio.output_latency_ms();
            if let Some(sample_rate) = audio.ring.take_sample_rate_change() {
                self.model.set_sample_rate(sample_rate);
            }
            let count = audio.ring.drain_latest(&mut self.left, &mut self.right);
            let samples = if self.model.right_channel {
                &self.right
            } else {
                &self.left
            };
            self.model.history.ingest(&samples[samples.len() - count..]);
        } else {
            let target =
                (self.demo_started.elapsed().as_secs_f64() * self.model.sample_rate as f64) as u64;
            // A suspended app skips old demo data instead of doing unbounded catch-up.
            self.demo_index = self
                .demo_index
                .max(target.saturating_sub(self.left.len() as u64));
            let count = target.saturating_sub(self.demo_index) as usize;
            for (offset, sample) in self.left[..count].iter_mut().enumerate() {
                *sample = if self.model.demo_beats {
                    beating_sample(self.demo_index + offset as u64, self.model.sample_rate)
                } else {
                    tone_sample(
                        self.demo_index + offset as u64,
                        self.model.sample_rate,
                        self.model.demo_hz,
                    )
                };
            }
            self.demo_index = target;
            self.model.history.ingest(&self.left[..count]);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = match event_loop.create_window(
            WindowAttributes::default()
                .with_title(if self.model.demo {
                    "WaveAudio — Timebase demo"
                } else {
                    "WaveAudio — Live audio"
                })
                .with_inner_size(LogicalSize::new(1060.0, 820.0))
                .with_min_inner_size(LogicalSize::new(540.0, 340.0)),
        ) {
            Ok(window) => Arc::new(window),
            Err(error) => {
                eprintln!("window creation failed: {error}");
                event_loop.exit();
                return;
            }
        };
        match pollster::block_on(Graphics::new(window.clone())) {
            Ok(graphics) => self.graphics = Some(graphics),
            Err(error) => {
                eprintln!("graphics initialization failed: {error:#}");
                event_loop.exit();
                return;
            }
        }
        self.demo_started = Instant::now();
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else {
            return;
        };
        if window.id() != window_id {
            return;
        }
        let graphics = self.graphics.as_mut().unwrap();
        let _ = graphics.input.on_window_event(window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                graphics.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested if !self.minimized => {
                self.ingest();
                let window = self.window.as_ref().unwrap();
                let graphics = self.graphics.as_mut().unwrap();
                match graphics.render(window, &mut self.model) {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = window.inner_size();
                        graphics.resize(size.width, size.height);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(error) => eprintln!("WaveAudio rendering error: {error}"),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.minimized {
            // Keep consuming capture while hidden so the callback queue stays current.
            self.ingest();
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_millis(50),
            ));
        } else {
            if Instant::now() >= self.next_frame {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
                self.next_frame = Instant::now()
                    + Duration::from_secs_f64(1.0 / self.model.target_fps.max(1) as f64);
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
        }
    }
}

fn main() -> Result<()> {
    let mut demo = false;
    let mut demo_beats = false;
    let mut demo_hz = 60.0_f64;
    let mut min_hz: Option<f64> = None;
    let mut max_hz: Option<f64> = None;
    let mut device = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => demo = true,
            "--demo-beats" => {
                demo = true;
                demo_beats = true;
            }
            "--demo-tone" => {
                demo = true;
                demo_hz = args
                    .next()
                    .context("--demo-tone needs a frequency")?
                    .parse()
                    .context("invalid demo frequency")?;
                anyhow::ensure!(
                    demo_hz.is_finite() && (1.0..=20_000.0).contains(&demo_hz),
                    "demo frequency must be 1–20000 Hz"
                );
            }
            "--min-hz" => {
                min_hz = Some(
                    args.next()
                        .context("--min-hz needs a frequency")?
                        .parse()
                        .context("invalid minimum frequency")?,
                )
            }
            "--max-hz" => {
                max_hz = Some(
                    args.next()
                        .context("--max-hz needs a frequency")?
                        .parse()
                        .context("invalid maximum frequency")?,
                )
            }
            "--device" => device = Some(args.next().context("--device needs a name fragment")?),
            "--help" | "-h" => {
                println!(
                    "wave-audio [--demo | --demo-tone HZ | --demo-beats | --device NAME] [--min-hz HZ] [--max-hz HZ]\n\nDefault: native system audio capture (macOS 14.2+).\n--demo: silent visual test with a pure 60 Hz tone.\n--demo-tone HZ: visual test with a chosen pure tone.\n--demo-beats: visual test with 440 + 442 Hz.\n--min-hz / --max-hz: reference frequency range along one curve.\nDefault: Min=60 Hz, Max=6000 Hz, log spacing, continuous scrolling, target 120 FPS.\nEach x reads audio(t_anchor - (1-u) / F(u)), with u from left to right.\nVisualAudio settings are not changed."
                );
                return Ok(());
            }
            _ => anyhow::bail!("unknown argument {arg:?}; use --help"),
        }
    }
    anyhow::ensure!(
        !(demo && device.is_some()),
        "choose a demo or --device, not both"
    );
    let audio = if demo {
        None
    } else {
        Some(match device {
            Some(name) => AudioInput::open(&name)?,
            None => AudioInput::system()?,
        })
    };
    let sample_rate = audio.as_ref().map_or(48_000, AudioInput::sample_rate);
    let mut model = Model::new(sample_rate, demo);
    model.demo_beats = demo_beats;
    model.demo_hz = demo_hz;
    if demo {
        model.min_hz = if demo_beats { 440.0 } else { demo_hz };
        model.max_hz = model.max_hz.max(model.min_hz);
    }
    if let Some(value) = min_hz {
        model.min_hz = value;
    }
    if let Some(value) = max_hz {
        model.max_hz = value;
    }
    let limit = (sample_rate as f64 * 0.45).min(20_000.0);
    anyhow::ensure!(
        model.min_hz.is_finite()
            && model.max_hz.is_finite()
            && model.min_hz >= 1.0
            && model.max_hz <= limit
            && model.max_hz >= model.min_hz,
        "reference limits must satisfy 1 <= Min <= Max <= {limit} Hz"
    );
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App {
        audio,
        model,
        left: vec![0.0; 131_072],
        right: vec![0.0; 131_072],
        demo_started: Instant::now(),
        demo_index: 0,
        window: None,
        graphics: None,
        next_frame: Instant::now(),
        minimized: false,
    })?;
    Ok(())
}
