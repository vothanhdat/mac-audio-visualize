mod gpu;
mod hybrid_matcher;
mod multirate_fft;
mod resonator_resampler;
mod window_state;

use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result};
use gpu::GpuState;
use visual_audio::audio::AudioInput;
use window_state::{Frame, WindowState};
#[cfg(target_os = "macos")]
use winit::platform::macos::WindowAttributesExtMacOS;
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalPosition, LogicalSize},
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

struct App {
    audio: AudioInput,
    profile: Option<String>,
    window: Option<Arc<Window>>,
    controls_window: Option<Arc<Window>>,
    controls_visible: bool,
    gpu: Option<GpuState>,
    frame_interval: Duration,
    next_frame: Instant,
    interaction_until: Option<Instant>,
    occluded: bool,
    window_state: WindowState,
    window_state_path: Option<PathBuf>,
    window_state_save_at: Option<Instant>,
}

// Window moves arrive continuously while dragging; save once they settle.
const WINDOW_STATE_SAVE_DELAY: Duration = Duration::from_millis(500);

impl App {
    fn schedule_window_state_save(&mut self) {
        self.window_state_save_at = Some(Instant::now() + WINDOW_STATE_SAVE_DELAY);
    }

    /// Save both windows' frames once a move or resize has settled. When
    /// exiting (`now`), save the current frames even without a reported move:
    /// macOS reports a drag only when it ends. A frame that cannot be read
    /// keeps its previous value, and unchanged frames are not rewritten.
    fn save_window_state_if_due(&mut self, now: bool) {
        let due = self
            .window_state_save_at
            .is_some_and(|deadline| Instant::now() >= deadline);
        if !now && !due {
            return;
        }
        self.window_state_save_at = None;
        let previous = self.window_state.clone();
        if let Some(frame) = self.window.as_deref().and_then(Frame::of) {
            self.window_state.visualizer = Some(frame);
        }
        if let Some(frame) = self.controls_window.as_deref().and_then(Frame::of) {
            self.window_state.controls = Some(frame);
        }
        if self.window_state == previous && !due {
            return;
        }
        if let Some(path) = &self.window_state_path
            && let Err(error) = self.window_state.save(path)
        {
            eprintln!("could not save window positions: {error:#}");
        }
    }
}

// While hidden, macOS may withhold drawables; skip rendering but keep the
// capture ring from filling so analysis resumes with current audio.
const HIDDEN_AUDIO_DRAIN_INTERVAL: Duration = Duration::from_millis(250);
// While idle in silence, how often to check for audio. Sound that returns
// is drawn within this interval plus one display refresh.
const IDLE_AUDIO_POLL_INTERVAL: Duration = Duration::from_millis(20);

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let profile_suffix = self
            .profile
            .as_deref()
            .map(|profile| format!(" — {profile}"))
            .unwrap_or_default();
        // Reopen windows where they were last, unless that display is gone.
        let screens = window_state::screens(event_loop.available_monitors());
        let restore = |attributes: WindowAttributes, frame: Option<Frame>| match frame
            .filter(|frame| frame.restorable(&screens))
        {
            Some(frame) => attributes
                .with_position(LogicalPosition::new(frame.x, frame.y))
                .with_inner_size(LogicalSize::new(frame.width, frame.height)),
            None => attributes,
        };
        let attributes = restore(
            WindowAttributes::default()
                .with_title(format!(
                    "VisualAudio — {}{profile_suffix}",
                    self.audio.device_name
                ))
                .with_transparent(true)
                .with_titlebar_transparent(true)
                .with_title_hidden(true)
                .with_fullsize_content_view(true)
                .with_movable_by_window_background(true)
                .with_has_shadow(false)
                .with_inner_size(LogicalSize::new(320.0, 180.0))
                // Short enough for a thin spectrum strip; the plot clamps at small heights.
                .with_min_inner_size(LogicalSize::new(320.0, 60.0)),
            self.window_state.visualizer,
        );
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("window creation failed"),
        );
        let refresh_millihertz = window
            .current_monitor()
            .and_then(|monitor| monitor.refresh_rate_millihertz())
            .unwrap_or(60_000)
            .max(1);
        self.frame_interval = Duration::from_secs_f64(1000.0 / refresh_millihertz as f64);
        self.next_frame = Instant::now();
        let controls_window = Arc::new(
            event_loop
                .create_window(restore(
                    WindowAttributes::default()
                        .with_title(format!("VisualAudio Controls{profile_suffix}"))
                        .with_inner_size(LogicalSize::new(380.0, 760.0))
                        .with_min_inner_size(LogicalSize::new(300.0, 360.0))
                        .with_visible(false),
                    self.window_state.controls,
                ))
                .expect("controls window creation failed"),
        );
        let gpu = match pollster::block_on(GpuState::new(
            Arc::clone(&window),
            Arc::clone(&controls_window),
            self.audio.sample_rate(),
            self.profile.as_deref(),
        )) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("GPU initialization failed: {error:#}");
                event_loop.exit();
                return;
            }
        };
        window.request_redraw();
        self.window = Some(window);
        self.controls_window = Some(controls_window);
        self.gpu = Some(gpu);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let is_visualizer = self
            .window
            .as_ref()
            .is_some_and(|window| window.id() == window_id);
        let is_controls = self
            .controls_window
            .as_ref()
            .is_some_and(|window| window.id() == window_id);
        if !is_visualizer && !is_controls {
            return;
        }
        if matches!(event, WindowEvent::Moved(_) | WindowEvent::Resized(_)) {
            self.schedule_window_state_save();
        }

        if is_controls {
            let controls_window = self.controls_window.as_ref().unwrap();
            let request_redraw = !matches!(
                &event,
                WindowEvent::RedrawRequested | WindowEvent::CloseRequested
            );
            self.gpu
                .as_mut()
                .unwrap()
                .handle_controls_event(controls_window, &event);
            match event {
                WindowEvent::CloseRequested => {
                    controls_window.set_visible(false);
                    self.controls_visible = false;
                }
                WindowEvent::Resized(size) => self.gpu.as_mut().unwrap().resize_controls(size),
                WindowEvent::RedrawRequested => {
                    match self.gpu.as_mut().unwrap().render_controls(controls_window) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            self.gpu
                                .as_mut()
                                .unwrap()
                                .resize_controls(controls_window.inner_size());
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(error) => eprintln!("controls rendering error: {error}"),
                    }
                }
                _ => {}
            }
            if request_redraw && self.controls_visible {
                controls_window.request_redraw();
            }
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.gpu.as_mut().unwrap().resize_visualizer(size);
                self.interaction_until = Some(Instant::now() + Duration::from_millis(60));
            }
            WindowEvent::Moved(_) => {
                self.interaction_until = Some(Instant::now() + Duration::from_millis(60));
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    // The last frame shown may predate the hidden period.
                    if let Some(gpu) = &mut self.gpu {
                        gpu.note_activity();
                    }
                    self.next_frame = Instant::now();
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                self.controls_visible = true;
                let controls_window = self.controls_window.as_ref().unwrap();
                controls_window.set_visible(true);
                controls_window.focus_window();
                controls_window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match physical_key {
                PhysicalKey::Code(KeyCode::Escape) => event_loop.exit(),
                PhysicalKey::Code(KeyCode::ArrowUp | KeyCode::Equal) => {
                    self.gpu.as_mut().unwrap().adjust_gain(1.25)
                }
                PhysicalKey::Code(KeyCode::ArrowDown | KeyCode::Minus) => {
                    self.gpu.as_mut().unwrap().adjust_gain(0.8)
                }
                PhysicalKey::Code(KeyCode::KeyH) => {
                    self.controls_visible = !self.controls_visible;
                    let controls_window = self.controls_window.as_ref().unwrap();
                    controls_window.set_visible(self.controls_visible);
                    if self.controls_visible {
                        controls_window.request_redraw();
                    }
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                if self.occluded
                    || self
                        .interaction_until
                        .is_some_and(|deadline| Instant::now() < deadline)
                {
                    return;
                }
                let output_latency_ms = self.audio.output_latency_ms();
                match self
                    .gpu
                    .as_mut()
                    .unwrap()
                    .render_visualizer(&self.audio.ring, output_latency_ms)
                {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = self.window.as_ref().unwrap().inner_size();
                        self.gpu.as_mut().unwrap().resize_visualizer(size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(error) => eprintln!("rendering error: {error}"),
                }
                // The FIFO drawable wait at the start of the next frame aligns it
                // with the display refresh. Requesting it after half an interval
                // keeps that wait short without missing a refresh; the floor only
                // prevents a busy loop if a drawable is ever returned immediately.
                self.next_frame = Instant::now() + self.frame_interval / 2;
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // The loop wakes at least every 250 ms in every state below, so a
        // pending save runs shortly after moves settle.
        self.save_window_state_if_due(false);
        let Some(window) = &self.window else {
            return;
        };
        let now = Instant::now();
        if self.occluded {
            if now >= self.next_frame {
                if let Some(gpu) = &mut self.gpu {
                    gpu.drain_undrawn_audio(&self.audio.ring);
                }
                self.next_frame = now + HIDDEN_AUDIO_DRAIN_INTERVAL;
            }
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
            return;
        }
        if self.interaction_until.is_none()
            && let Some(gpu) = &mut self.gpu
            && gpu.idle_ready()
        {
            // Silent and settled: the frame on screen is still correct. Check
            // the newest audio without drawing, and resume as soon as it returns.
            if now >= self.next_frame {
                gpu.drain_undrawn_audio(&self.audio.ring);
                self.next_frame = now + IDLE_AUDIO_POLL_INTERVAL;
            }
            if gpu.idle_ready() {
                event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
                return;
            }
            self.next_frame = now;
        }
        if let Some(deadline) = self.interaction_until {
            if now >= deadline {
                self.interaction_until = None;
                self.next_frame = now;
                window.request_redraw();
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
                return;
            }
        } else if now >= self.next_frame {
            window.request_redraw();
            self.next_frame = now + self.frame_interval;
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_frame));
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Quitting right after a move, or before a move was reported, must not lose it.
        self.save_window_state_if_due(true);
    }
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut device_match = None;
    let mut profile = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => {
                device_match = Some(args.next().context("--device needs a name fragment")?)
            }
            "--profile" => {
                let value = args.next().context("--profile needs a name")?;
                profile = Some(validate_profile_name(&value)?);
            }
            "--help" | "-h" => {
                println!(
                    "visual-audio [--device NAME] [--profile NAME]\n\nDefaults to native system audio capture (macOS 14.2+, no virtual driver).\nUse --device NAME to capture a microphone or loopback input instead.\nProfile names use letters, numbers, '-' or '_'.\nKeys: Up/+ gain, Down/- gain, H controls, Esc quit."
                );
                return Ok(());
            }
            _ => anyhow::bail!("unknown argument {arg:?}; use --help"),
        }
    }

    let audio = match device_match {
        Some(name) => AudioInput::open(&name)?,
        None => AudioInput::system()?,
    };
    eprintln!(
        "capturing {} at {} Hz, {} channel(s)",
        audio.device_name,
        audio.sample_rate(),
        audio.channels
    );
    let window_state_path = window_state::window_state_path(profile.as_deref());
    let window_state = window_state_path
        .as_deref()
        .map(WindowState::load)
        .unwrap_or_default();
    let event_loop = EventLoop::new().context("could not create the event loop")?;
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut App {
        audio,
        profile,
        window_state,
        window_state_path,
        window_state_save_at: None,
        window: None,
        controls_window: None,
        controls_visible: false,
        gpu: None,
        frame_interval: Duration::from_micros(16_667),
        next_frame: Instant::now(),
        interaction_until: None,
        occluded: false,
    })?;
    Ok(())
}

fn validate_profile_name(value: &str) -> Result<String> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > 48
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        anyhow::bail!("profile names must be 1–48 letters, numbers, '-' or '_'");
    }
    Ok(value.to_owned())
}
