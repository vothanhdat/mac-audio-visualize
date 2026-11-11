use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use egui::{Color32, Pos2, Rect, Stroke};
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use visual_audio::{
    audio::{AudioInput, CaptureStall},
    waveform::{ReferenceSweep, Trace, WaveformHistory, beating_sample, tone_sample},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes, WindowId},
};

#[path = "wave_audio/pitch.rs"]
mod pitch;
use pitch::{DEFAULT_FILTER_DELAY_MS, NOTES, PitchHistory};

const NOTE_WINDOW_MS: std::ops::RangeInclusive<u32> = 20..=500;
const RAW_WAVE_COLOR: Color32 = Color32::from_rgb(84, 220, 209);
const NOTE_CAPTURE_GRACE: Duration = Duration::from_millis(40);

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

#[derive(Clone, Copy)]
enum Demo {
    Tone(f64),
    Beats,
    Chord,
    Bass,
}

impl Demo {
    fn sample(self, index: u64, sample_rate: u32) -> f32 {
        match self {
            Self::Tone(hz) => tone_sample(index, sample_rate, hz),
            Self::Beats => beating_sample(index, sample_rate),
            Self::Chord => [60, 64, 67]
                .into_iter()
                .map(|midi| tone_sample(index, sample_rate, pitch::frequency(midi)) / 3.0)
                .sum(),
            Self::Bass => {
                let beat_samples = u64::from(sample_rate) * 3 / 4;
                let beat = index / beat_samples;
                let time = (index % beat_samples) as f64 / f64::from(sample_rate);
                let envelope =
                    (time / 0.01).clamp(0.0, 1.0) * ((0.65 - time) / 0.12).clamp(0.0, 1.0);
                let midi = [36, 43, 45, 41][(beat % 4) as usize];
                tone_sample(index, sample_rate, pitch::frequency(midi)) * envelope as f32
            }
        }
    }
}

struct Model {
    history: WaveformHistory,
    notes: Option<PitchHistory>,
    lock_notes: bool,
    note_cycles: u32,
    note_window_ms: u32,
    max_filter_delay_ms: u32,
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
    demo: Option<Demo>,
}

impl Model {
    fn new(sample_rate: u32, demo: Option<Demo>, sweep: bool) -> Self {
        Self {
            history: WaveformHistory::new(sample_rate),
            notes: (!sweep).then(|| PitchHistory::new(sample_rate, 0, DEFAULT_FILTER_DELAY_MS)),
            lock_notes: true,
            note_cycles: 3,
            note_window_ms: *NOTE_WINDOW_MS.end(),
            max_filter_delay_ms: DEFAULT_FILTER_DELAY_MS,
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
        }
    }

    fn ingest(&mut self, samples: &[f32]) {
        self.history.ingest(samples);
        if let Some(notes) = &mut self.notes {
            notes.ingest(samples);
        }
    }

    fn reset_history(&mut self) {
        self.history = WaveformHistory::new(self.sample_rate);
        if self.notes.is_some() {
            self.notes = Some(PitchHistory::new(
                self.sample_rate,
                0,
                self.max_filter_delay_ms,
            ));
        }
    }

    /// Follow a capture rate change (a new output device or format): the
    /// history restarts and the reference range stays below Nyquist.
    fn set_sample_rate(&mut self, sample_rate: u32) {
        if sample_rate == 0 || sample_rate == self.sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        self.reset_history();
        let limit = (sample_rate as f64 * 0.45).min(20_000.0).max(1.0);
        self.max_hz = self.max_hz.min(limit);
        self.min_hz = self.min_hz.min(self.max_hz);
    }

    fn ui(&mut self, context: &egui::Context) {
        let previous_channel = self.right_channel;
        let previous_delay_limit = self.max_filter_delay_ms;
        let mut show_notes = self.notes.is_some();
        let max_hz = (self.sample_rate as f64 * 0.45).min(20_000.0).max(1.0);
        egui::TopBottomPanel::top("controls").show(context, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("WaveAudio");
                ui.separator();
                if let Some(demo) = self.demo {
                    let label = match demo {
                        Demo::Tone(hz) => format!("Demo: {hz:.2} Hz"),
                        Demo::Beats => "Demo: 440 + 442 Hz".into(),
                        Demo::Chord => "Demo: C major".into(),
                        Demo::Bass => "Demo: bass · C2 / G2 / A2 / F2".into(),
                    };
                    ui.label(label);
                } else {
                    ui.selectable_value(&mut self.right_channel, false, "Left");
                    ui.selectable_value(&mut self.right_channel, true, "Right");
                }
                ui.separator();
                ui.selectable_value(&mut show_notes, true, "12 notes");
                ui.selectable_value(&mut show_notes, false, "Time sweep");
            });
            if !show_notes {
                ui.horizontal_wrapped(|ui| {
                    ui.label("Reference F: Min");
                    ui.add(
                        egui::DragValue::new(&mut self.min_hz)
                            .range(1.0..=max_hz)
                            .speed(0.1)
                            .min_decimals(2)
                            .max_decimals(4)
                            .suffix(" Hz"),
                    );
                    self.max_hz = self.max_hz.clamp(self.min_hz, max_hz);
                    ui.label("Max");
                    ui.add(
                        egui::DragValue::new(&mut self.max_hz)
                            .range(self.min_hz..=max_hz)
                            .speed(0.1)
                            .min_decimals(2)
                            .max_decimals(4)
                            .suffix(" Hz"),
                    );
                    ui.checkbox(&mut self.logarithmic, "Log spacing");
                });
            }
            ui.horizontal_wrapped(|ui| {
                if show_notes {
                    ui.checkbox(&mut self.lock_notes, "Phase lock");
                    if self.lock_notes {
                        ui.add(egui::Slider::new(&mut self.note_cycles, 1..=12).text("Cycles"));
                    } else {
                        ui.add(
                            egui::Slider::new(&mut self.note_window_ms, NOTE_WINDOW_MS)
                                .suffix(" ms")
                                .text("Time window"),
                        );
                    }
                } else {
                    ui.checkbox(&mut self.phase_lock, "Lock phase to Min")
                        .on_hover_text("Align the shared read endpoint to the Min period.");
                }
                ui.add(
                    egui::Slider::new(&mut self.gain, 0.1..=10.0)
                        .logarithmic(true)
                        .text("gain"),
                );
            });
            if show_notes {
                ui.add(
                    egui::Slider::new(&mut self.max_filter_delay_ms, 60..=250)
                        .suffix(" ms")
                        .text("Max filter delay"),
                )
                .on_hover_text("Lower values speed up bass response and allow more spill between neighboring notes.");
            }
            ui.horizontal_wrapped(|ui| {
                ui.label("Target FPS");
                ui.add(
                    egui::DragValue::new(&mut self.target_fps)
                        .range(30..=240)
                        .speed(1.0),
                );
                ui.small("Actual presentation depends on the display; audio time sets the window.");
            });
            if self.demo.is_none() {
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut self.auto_sync, "Follow output delay");
                    ui.add(
                        egui::Slider::new(&mut self.adjustment_ms, -500.0..=1000.0)
                            .text("adjustment (ms)"),
                    );
                    match self.latency_ms {
                        Some(ms) => {
                            ui.label(format!("Output: {ms:.0} ms"));
                        }
                        None => {
                            ui.label("Output delay unavailable");
                        }
                    }
                });
            }
        });
        if self.right_channel != previous_channel {
            self.reset_history();
        }
        if show_notes != self.notes.is_some() {
            self.notes = show_notes.then(|| {
                PitchHistory::new(
                    self.sample_rate,
                    self.history.sample_count,
                    self.max_filter_delay_ms,
                )
            });
        } else if previous_delay_limit != self.max_filter_delay_ms
            && let Some(notes) = &mut self.notes
        {
            notes.set_delay_limit(self.sample_rate, self.max_filter_delay_ms);
        }
        let delay_ms = if self.demo.is_some() {
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
            if let Some(notes) = &self.notes {
                self.note_plot(ui, notes, available_end);
                return;
            }
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

    fn note_plot(&self, ui: &mut egui::Ui, notes: &PitchHistory, end: f64) {
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::from_rgb(13, 20, 30));
        let left = rect.left() + 44.0;
        let right = rect.right() - 10.0;
        let top = rect.top() + 8.0;
        let bottom = rect.bottom() - 23.0;
        let width = right - left;
        let height = bottom - top;
        if width < 1.0 || height < 1.0 {
            return;
        }

        let lane_height = height / NOTES.len() as f32;
        let label = egui::FontId::proportional(11.0);
        for note in 0..NOTES.len() {
            if note % 2 == 0 {
                let y = top + note as f32 * lane_height;
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(left, y), Pos2::new(right, y + lane_height)),
                    0.0,
                    Color32::from_rgb(17, 27, 39),
                );
            }
        }
        for step in 0..=4 {
            let fraction = f64::from(step) / 4.0;
            let x = left + width * fraction as f32;
            painter.line_segment(
                [Pos2::new(x, top), Pos2::new(x, bottom)],
                Stroke::new(0.5, Color32::from_gray(49)),
            );
            let align = match step {
                0 => egui::Align2::LEFT_BOTTOM,
                4 => egui::Align2::RIGHT_BOTTOM,
                _ => egui::Align2::CENTER_BOTTOM,
            };
            let text = if step == 4 {
                if self.lock_notes { "Trigger" } else { "Now" }.to_owned()
            } else if self.lock_notes {
                let cycles = (1.0 - fraction) * f64::from(self.note_cycles);
                format!("−{cycles} cycles")
            } else {
                let age_ms = (1.0 - fraction) * f64::from(self.note_window_ms);
                format!("−{age_ms} ms")
            };
            painter.text(
                Pos2::new(x, rect.bottom() - 4.0),
                align,
                text,
                label.clone(),
                Color32::GRAY,
            );
        }

        let sample_rate = f64::from(self.sample_rate);
        let shared_span = f64::from(self.note_window_ms) * sample_rate / 1000.0;
        let frame = if self.lock_notes {
            notes.frame_at(end)
        } else {
            Default::default()
        };
        for (note, name) in NOTES.iter().enumerate() {
            let lane_top = top + note as f32 * lane_height;
            let lane = Rect::from_min_max(
                Pos2::new(left, lane_top),
                Pos2::new(right, lane_top + lane_height),
            );
            let color: Color32 =
                egui::ecolor::Hsva::new(note as f32 / 12.0, 0.56, 0.94, 1.0).into();
            let trigger = frame.triggers[note];
            let lane_end = trigger.map_or(end, |trigger| trigger.end);
            let span = if self.lock_notes {
                let period = trigger.map_or_else(
                    || sample_rate / pitch::frequency(60 + note as u8),
                    |trigger| trigger.period,
                );
                period * f64::from(self.note_cycles)
            } else {
                shared_span
            };
            let name = trigger.map_or_else(
                || (*name).to_owned(),
                |trigger| format!("{name}{}", trigger.midi / 12 - 1),
            );
            painter.text(
                Pos2::new(left - 8.0, lane.center().y),
                egui::Align2::RIGHT_CENTER,
                name,
                label.clone(),
                color,
            );
            let trace = (!frame.silent).then(|| notes.trace(note));
            self.draw_trace(&painter, lane, trace, color, |u| {
                lane_end - (1.0 - u) * span
            });
        }
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
        self.draw_trace(
            &painter,
            plot,
            Some(&self.history.raw),
            RAW_WAVE_COLOR,
            |u| sweep.sample_position(u),
        );
    }

    fn draw_trace(
        &self,
        painter: &egui::Painter,
        plot: Rect,
        trace: Option<&Trace>,
        color: Color32,
        position_at: impl Fn(f64) -> f64,
    ) {
        painter.line_segment(
            [
                Pos2::new(plot.left(), plot.center().y),
                Pos2::new(plot.right(), plot.center().y),
            ],
            Stroke::new(0.5, Color32::from_gray(65)),
        );
        let Some(trace) = trace else {
            return;
        };
        let painter = painter.with_clip_rect(plot);
        let columns = plot.width().ceil().clamp(1.0, 1800.0) as usize;
        let scale = plot.height() * 0.44 * self.gain;
        let y = |value: f32| plot.center().y - value * scale;
        let mut line = Vec::with_capacity(columns + 1);
        for point in 0..=columns {
            let fraction = point as f64 / columns as f64;
            let position = position_at(fraction);
            if let Some(value) = trace.interpolate(position) {
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
            let next = position_at((point + 1) as f64 / columns as f64);
            if point < columns
                && next - position > 1.0
                && let Some((low, high)) = trace.segment_range(position, next)
                && low < high
            {
                let x = plot.left() + plot.width() * fraction as f32;
                painter.line_segment(
                    [Pos2::new(x, y(low)), Pos2::new(x, y(high))],
                    Stroke::new(1.0, color),
                );
            }
        }
        if line.len() > 1 {
            painter.add(egui::Shape::line(line, Stroke::new(1.0, color)));
        }
    }
}

struct App {
    audio: Option<AudioInput>,
    capture_stall: CaptureStall,
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
                self.capture_stall =
                    CaptureStall::with_grace_period(Instant::now(), NOTE_CAPTURE_GRACE);
            }
            let count = audio.ring.drain_latest(&mut self.left, &mut self.right);
            let history_samples = 4 * u64::from(self.model.sample_rate);
            let remaining = history_samples.saturating_sub(self.capture_stall.filled()) as usize;
            let silent = self.capture_stall.silent_samples(
                count,
                Instant::now(),
                self.model.sample_rate as f32,
            );
            let samples = if self.model.right_channel {
                &self.right
            } else {
                &self.left
            };
            self.model.ingest(&samples[samples.len() - count..]);
            if self.model.notes.is_some() {
                let mut silent = silent.min(remaining);
                while silent > 0 {
                    let chunk = silent.min(self.left.len());
                    self.left[..chunk].fill(0.0);
                    self.model.ingest(&self.left[..chunk]);
                    silent -= chunk;
                }
            }
        } else {
            let target =
                (self.demo_started.elapsed().as_secs_f64() * self.model.sample_rate as f64) as u64;
            // A suspended app skips old demo data instead of doing unbounded catch-up.
            self.demo_index = self
                .demo_index
                .max(target.saturating_sub(self.left.len() as u64));
            let count = target.saturating_sub(self.demo_index) as usize;
            for (offset, sample) in self.left[..count].iter_mut().enumerate() {
                *sample = self
                    .model
                    .demo
                    .unwrap()
                    .sample(self.demo_index + offset as u64, self.model.sample_rate);
            }
            self.demo_index = target;
            self.model.ingest(&self.left[..count]);
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
                .with_title(if self.model.demo.is_some() {
                    "WaveAudio — Demo"
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
    let mut demo = None;
    let mut sweep = false;
    let mut min_hz: Option<f64> = None;
    let mut max_hz: Option<f64> = None;
    let mut device = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--demo" => demo = Some(Demo::Tone(60.0)),
            "--demo-beats" => demo = Some(Demo::Beats),
            "--demo-chord" => demo = Some(Demo::Chord),
            "--demo-bass" => demo = Some(Demo::Bass),
            "--sweep" => sweep = true,
            "--demo-tone" => {
                let hz: f64 = args
                    .next()
                    .context("--demo-tone needs a frequency")?
                    .parse()
                    .context("invalid demo frequency")?;
                anyhow::ensure!(
                    hz.is_finite() && (1.0..=20_000.0).contains(&hz),
                    "demo frequency must be 1–20000 Hz"
                );
                demo = Some(Demo::Tone(hz));
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
                    "wave-audio [--device NAME | --demo | --demo-tone HZ | --demo-chord | --demo-bass | --demo-beats]\n           [--sweep] [--min-hz HZ] [--max-hz HZ]\n\nDefault: 12 filtered waveforms using native macOS audio capture.\nDemos are silent. --demo uses 60 Hz, --demo-chord uses C major, and --demo-bass loops a low phrase.\n--sweep opens the variable-timebase waveform. --min-hz / --max-hz imply --sweep."
                );
                return Ok(());
            }
            _ => anyhow::bail!("unknown argument {arg:?}; use --help"),
        }
    }
    anyhow::ensure!(
        !(demo.is_some() && device.is_some()),
        "choose a demo or --device, not both"
    );
    let audio = if demo.is_some() {
        None
    } else {
        Some(match device {
            Some(name) => AudioInput::open(&name)?,
            None => AudioInput::system()?,
        })
    };
    let sample_rate = audio.as_ref().map_or(48_000, AudioInput::sample_rate);
    let mut model = Model::new(
        sample_rate,
        demo,
        sweep || min_hz.is_some() || max_hz.is_some(),
    );
    model.min_hz = match demo {
        Some(Demo::Tone(hz)) => hz,
        Some(Demo::Beats) => 440.0,
        _ => model.min_hz,
    };
    model.max_hz = model.max_hz.max(model.min_hz);
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
        capture_stall: CaptureStall::with_grace_period(Instant::now(), NOTE_CAPTURE_GRACE),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bass_demo_moves_through_the_expected_lanes() {
        let sample_rate = 48_000;
        let mut history = PitchHistory::new(sample_rate, 0, DEFAULT_FILTER_DELAY_MS);
        let mut start = 0;
        for (step, expected) in [0, 7, 9, 5].into_iter().enumerate() {
            let end = step as u64 * 36_000 + 28_800;
            let samples: Vec<_> = (start..end)
                .map(|index| Demo::Bass.sample(index, sample_rate))
                .collect();
            history.ingest(&samples);
            let peaks: [f32; 12] = std::array::from_fn(|note| {
                let (low, high) = history
                    .trace(note)
                    .segment_range((end - 960) as f64, (end - 1) as f64)
                    .unwrap();
                low.abs().max(high.abs())
            });
            assert!(peaks[expected] > 0.35, "{peaks:?}");
            for (note, peak) in peaks.into_iter().enumerate() {
                if note != expected {
                    assert!(peak < peaks[expected] * 0.55, "{peaks:?}");
                }
            }
            start = end;
        }
    }
}
