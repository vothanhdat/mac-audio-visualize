//! A pitch-class overlay with optional short-window audio analysis. No GPU
//! readback is needed; the font atlas is prepared once when enabled.

use wgpu::util::DeviceExt;

use super::{compute_pipeline, storage_entry, uniform_entry};

#[path = "fast_notes.rs"]
mod fast_notes;
use fast_notes::FastNotes;

const NOTE_LABELS: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];
const LABEL_CELL: usize = 96;

pub(super) struct NoteRibbon {
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
    compute_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    appearance_buffer: wgpu::Buffer,
    width_percent: f32,
    midrange_only: bool,
    fast_response: bool,
    fast_notes_buffer: wgpu::Buffer,
    fast_analyzer: Option<FastNotes>,
    smoothing_buffer: wgpu::Buffer,
    smoothing_ms: f32,
    // Retained for GPU validation/readback in tests. Bind groups also retain it.
    _state: wgpu::Buffer,
}

impl NoteRibbon {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        params: &wgpu::Buffer,
        left: &wgpu::Buffer,
        right: &wgpu::Buffer,
    ) -> Self {
        let state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("twelve-note activity state"),
            contents: bytemuck::cast_slice(&[[0.0f32; 4]; 13]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let appearance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("note ribbon appearance and frequency range"),
            contents: bytemuck::cast_slice(&[0.76f32, 200.0, 2000.0, 0.0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let fast_notes_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fresh audio pitch-class power"),
            contents: bytemuck::cast_slice(&[0.0f32; 12]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let smoothing_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("note ribbon visual smoothing"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("note activity compute layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::COMPUTE),
                storage_entry(1, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(2, true, wgpu::ShaderStages::COMPUTE),
                storage_entry(3, false, wgpu::ShaderStages::COMPUTE),
                uniform_entry(4, wgpu::ShaderStages::COMPUTE),
                storage_entry(5, true, wgpu::ShaderStages::COMPUTE),
                uniform_entry(6, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("note activity compute bindings"),
            layout: &compute_layout,
            entries: &[
                buffer_entry(0, params),
                buffer_entry(1, left),
                buffer_entry(2, right),
                buffer_entry(3, &state),
                buffer_entry(4, &appearance_buffer),
                buffer_entry(5, &fast_notes_buffer),
                buffer_entry(6, &smoothing_buffer),
            ],
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("note activity compute pipeline layout"),
                bind_group_layouts: &[&compute_layout],
                push_constant_ranges: &[],
            });
        let compute_shader = device.create_shader_module(wgpu::include_wgsl!("note_activity.wgsl"));
        let compute_pipeline = compute_pipeline(
            device,
            &compute_pipeline_layout,
            &compute_shader,
            "pitch-class activity",
            "analyze_notes",
        );

        let atlas = label_atlas();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("note label font atlas"),
            size: wgpu::Extent3d {
                width: (LABEL_CELL * 12) as u32,
                height: LABEL_CELL as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            texture.as_image_copy(),
            &atlas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((LABEL_CELL * 12) as u32),
                rows_per_image: Some(LABEL_CELL as u32),
            },
            texture.size(),
        );
        let texture_view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("smooth note lettering"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("note ribbon render layout"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                storage_entry(1, true, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(4, wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("note ribbon render bindings"),
            layout: &render_layout,
            entries: &[
                buffer_entry(0, params),
                buffer_entry(1, &state),
                buffer_entry(4, &appearance_buffer),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("note ribbon render pipeline layout"),
                bind_group_layouts: &[&render_layout],
                push_constant_ranges: &[],
            });
        let shader = device.create_shader_module(wgpu::include_wgsl!("note_ribbon.wgsl"));
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("top note activity ribbon"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("ribbon_vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("ribbon_fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        Self {
            compute_pipeline,
            render_pipeline,
            compute_bind_group,
            render_bind_group,
            appearance_buffer,
            width_percent: 76.0,
            midrange_only: true,
            fast_response: false,
            fast_notes_buffer,
            fast_analyzer: None,
            smoothing_buffer,
            smoothing_ms: 0.0,
            _state: state,
        }
    }

    pub(super) fn configure(
        &mut self,
        queue: &wgpu::Queue,
        percent: f32,
        midrange_only: bool,
        fast_response: bool,
    ) {
        if self.width_percent != percent
            || self.midrange_only != midrange_only
            || self.fast_response != fast_response
        {
            let (lower, upper) = if midrange_only {
                (200.0, 2000.0)
            } else {
                (0.0, 1e9)
            };
            queue.write_buffer(
                &self.appearance_buffer,
                0,
                bytemuck::cast_slice(&[percent * 0.01, lower, upper, f32::from(fast_response)]),
            );
            if self.midrange_only != midrange_only || self.fast_response != fast_response {
                // A range change should not retain a highlight from excluded notes.
                queue.write_buffer(&self._state, 0, bytemuck::cast_slice(&[[0.0f32; 4]; 13]));
            }
            self.width_percent = percent;
            self.midrange_only = midrange_only;
            if self.fast_response != fast_response {
                self.fast_analyzer = fast_response.then(FastNotes::default);
            }
            self.fast_response = fast_response;
        }
    }

    pub(super) fn update_audio(
        &mut self,
        queue: &wgpu::Queue,
        samples: [&[f32]; 2],
        new_count: usize,
        sample_rate: f32,
        gain: f32,
        frequency_range: [f32; 2],
        reset: bool,
    ) {
        if let Some(analyzer) = &mut self.fast_analyzer {
            let range = if self.midrange_only {
                [
                    frequency_range[0].max(200.0),
                    frequency_range[1].min(2000.0),
                ]
            } else {
                frequency_range
            };
            let power = analyzer.update(samples, new_count, sample_rate, range, reset);
            queue.write_buffer(
                &self.fast_notes_buffer,
                0,
                bytemuck::cast_slice(&power.map(|value| value * gain * gain)),
            );
        }
    }

    pub(super) fn set_smoothing(&mut self, queue: &wgpu::Queue, milliseconds: f32) {
        if self.smoothing_ms != milliseconds {
            queue.write_buffer(
                &self.smoothing_buffer,
                0,
                bytemuck::cast_slice(&[milliseconds * 0.001, 0.0, 0.0, 0.0]),
            );
            self.smoothing_ms = milliseconds;
        }
    }

    pub(super) fn analyze(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("twelve-note activity"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &self.compute_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub(super) fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("top note ribbon overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.draw(&mut pass);
    }

    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Rasterize the app's existing proportional font at high resolution once.
/// Each atlas cell centers a complete label. ASCII sharps use the app's bundled
/// font reliably, without depending on system font fallback.
fn label_atlas() -> Vec<u8> {
    let mut fonts = egui::epaint::text::Fonts::new(2048, Default::default(), Default::default());
    let galleys: Vec<_> = {
        let mut view = fonts.with_pixels_per_point(1.0);
        NOTE_LABELS
            .iter()
            .map(|text| {
                view.layout_no_wrap(
                    (*text).to_owned(),
                    egui::FontId::proportional(64.0),
                    egui::Color32::WHITE,
                )
            })
            .collect()
    };
    let image = fonts.image();
    let width = LABEL_CELL * 12;
    let mut atlas = vec![0u8; width * LABEL_CELL];
    for (note, galley) in galleys.iter().enumerate() {
        let row = &galley.rows[0];
        let bounds = row.visuals.mesh_bounds;
        let offset = egui::vec2(
            LABEL_CELL as f32 * 0.5 - bounds.center().x,
            LABEL_CELL as f32 * 0.5 - bounds.center().y,
        );
        for glyph in &row.glyphs {
            let uv = glyph.uv_rect;
            let position = glyph.pos + uv.offset;
            let x0 = (position.x.round() + offset.x).round() as i32;
            let y0 = (position.y.round() + offset.y).round() as i32;
            for y in 0..(uv.max[1] - uv.min[1]) as usize {
                for x in 0..(uv.max[0] - uv.min[0]) as usize {
                    let dx = x0 + x as i32;
                    let dy = y0 + y as i32;
                    if (0..LABEL_CELL as i32).contains(&dx) && (0..LABEL_CELL as i32).contains(&dy)
                    {
                        let alpha = image.pixels
                            [(uv.min[1] as usize + y) * image.size[0] + uv.min[0] as usize + x]
                            .a();
                        let pixel =
                            &mut atlas[dy as usize * width + note * LABEL_CELL + dx as usize];
                        *pixel = (*pixel).max(alpha);
                    }
                }
            }
        }
    }
    atlas
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{
        AnalysisParams, BarLayout, VisualizerSettings, analysis_params, display_bin_count,
    };
    use winit::dpi::PhysicalSize;

    #[test]
    fn all_twelve_labels_have_distinct_visible_glyphs() {
        let mut fonts =
            egui::epaint::text::Fonts::new(2048, Default::default(), Default::default());
        let mut view = fonts.with_pixels_per_point(1.0);
        for character in NOTE_LABELS.iter().flat_map(|label| label.chars()) {
            assert!(view.has_glyph(&egui::FontId::proportional(64.0), character));
        }
        let atlas = label_atlas();
        let cells: Vec<Vec<u8>> = (0..12)
            .map(|note| {
                (0..LABEL_CELL)
                    .flat_map(|y| {
                        atlas[y * LABEL_CELL * 12 + note * LABEL_CELL
                            ..y * LABEL_CELL * 12 + (note + 1) * LABEL_CELL]
                            .iter()
                            .copied()
                    })
                    .collect()
            })
            .collect();
        for (note, cell) in cells.iter().enumerate() {
            assert!(
                cell.iter().filter(|&&alpha| alpha > 128).count() > 100,
                "{} is empty",
                NOTE_LABELS[note]
            );
            for other in &cells[..note] {
                assert_ne!(cell, other);
            }
        }
    }

    #[test]
    #[ignore = "requires Metal; checks ribbon pixel equivalence and isolated render cost"]
    fn gpu_note_ribbon_render_optimization_preserves_pixels() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                ..Default::default()
            });
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
            let settings = VisualizerSettings::default();
            let mut params = analysis_params(
                &settings,
                [48000.0, 0.0, 0.016],
                32,
                false,
                ([0; 5], false),
                [0.0; 4],
                PhysicalSize::new(640, 360),
            );
            let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            let spectrum = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[[0.0f32; 4]; 32]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let format = wgpu::TextureFormat::Rgba8UnormSrgb;
            let mut reference =
                NoteRibbon::new(&device, &queue, format, &uniform, &spectrum, &spectrum);
            let mut optimized =
                NoteRibbon::new(&device, &queue, format, &uniform, &spectrum, &spectrum);
            let reference_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("original note ribbon renderer"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("testdata/note_ribbon_reference.wgsl").into(),
                ),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&reference.render_pipeline.get_bind_group_layout(0)],
                push_constant_ranges: &[],
            });
            reference.render_pipeline =
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("original note ribbon renderer"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &reference_shader,
                        entry_point: Some("ribbon_vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &reference_shader,
                        entry_point: Some("ribbon_fs"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview: None,
                    cache: None,
                });
            let mut cases = 0;
            let mut max_delta = 0;
            for (width, height) in [(640, 360), (1920, 1080)] {
                params.resolution = [width as f32, height as f32];
                queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                for percent in [20.0, 26.0, 76.0, 100.0] {
                    reference.configure(&queue, percent, true, true);
                    optimized.configure(&queue, percent, true, true);
                    for pattern in 0..3 {
                        let mut state = [[0.0f32; 4]; 13];
                        for note in 0..12 {
                            state[note][1] = if pattern == 0 {
                                0.0
                            } else {
                                (note as f32 + 1.0) / 12.0
                            };
                            state[note][2] = if note == pattern * 5 { 1.0 } else { 0.0 };
                        }
                        for ribbon in [&reference, &optimized] {
                            queue.write_buffer(&ribbon._state, 0, bytemuck::cast_slice(&state));
                        }
                        for clear in [
                            wgpu::Color::TRANSPARENT,
                            wgpu::Color::BLACK,
                            wgpu::Color {
                                r: 0.12,
                                g: 0.33,
                                b: 0.07,
                                a: 1.0,
                            },
                        ] {
                            let expected =
                                render_and_read(&device, &queue, &reference, width, height, clear);
                            for merged in [false, true] {
                                let actual = render_and_read_mode(
                                    &device, &queue, &optimized, width, height, clear, merged,
                                );
                                let delta = expected
                                    .iter()
                                    .zip(&actual)
                                    .map(|(a, b)| a.abs_diff(*b))
                                    .max()
                                    .unwrap();
                                assert!(
                                    delta <= 1,
                                    "ribbon differs: {width}x{height}, width {percent}%, pattern {pattern}, merged {merged}, max delta {delta}"
                                );
                                max_delta = max_delta.max(delta);
                                cases += 1;
                            }
                        }
                    }
                }
            }
            eprintln!(
                "Note ribbon: {cases} rendered cases; largest channel difference {max_delta}/255."
            );
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 1920,
                    height: 1080,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            for percent in [26.0, 76.0] {
                reference.configure(&queue, percent, true, true);
                optimized.configure(&queue, percent, true, true);
                let mut timings = Vec::new();
                for (ribbon, merged) in
                    [(&reference, false), (&optimized, false), (&optimized, true)]
                {
                    let ms = crate::gpu::gpu_performance_tests::completed_batch(
                        &device,
                        &queue,
                        |encoder| {
                            {
                                let mut pass =
                                    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                        label: None,
                                        color_attachments: &[Some(
                                            wgpu::RenderPassColorAttachment {
                                                view: &view,
                                                resolve_target: None,
                                                depth_slice: None,
                                                ops: wgpu::Operations {
                                                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                                    store: wgpu::StoreOp::Store,
                                                },
                                            },
                                        )],
                                        depth_stencil_attachment: None,
                                        timestamp_writes: None,
                                        occlusion_query_set: None,
                                    });
                                if merged {
                                    ribbon.draw(&mut pass);
                                }
                            }
                            if !merged {
                                ribbon.render(encoder, &view);
                            }
                        },
                    );
                    timings.push(ms);
                }
                eprintln!(
                    "1080p ribbon {percent}% width: original {:.3}ms, conditional atlas {:.3}ms, merged pass {:.3}ms (CPU + GPU completion; ribbon and clear only)",
                    timings[0], timings[1], timings[2]
                );
            }
        });
    }

    /// Synthetic measured spectra let the production Metal shader be checked
    /// independently of microphone permission or the current audio device.
    #[test]
    #[ignore = "requires a Metal GPU"]
    fn gpu_note_activity_tracks_octaves_stereo_release_and_renders() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::METAL,
                ..Default::default()
            });
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let size = PhysicalSize::new(1280, 640);
            for (layout, focus, origin) in [
                (BarLayout::FixedPixels, 50.0, 50.0),
                (BarLayout::FixedPixels, 60.0, 50.0),
                (BarLayout::FixedPixels, 85.0, 30.0),
                (BarLayout::MusicalNotes, 60.0, 50.0),
            ] {
                let settings = VisualizerSettings {
                    bar_layout: layout,
                    frequency_focus_strength_percent: focus,
                    frequency_focus_origin_percent: origin,
                    ..Default::default()
                };
                let count = display_bin_count(&settings, size, 48000.0);
                let mut params = analysis_params(
                    &settings,
                    [48000.0, 0.0, 1.0 / 60.0],
                    count,
                    true,
                    ([0; 5], false),
                    [0.0; 4],
                    size,
                );
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let input = |data: &[[f32; 4]]| {
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: None,
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    })
                };
                let silence = vec![[0.0; 4]; count as usize];
                // Only the RIGHT channel contains A3 and A4, with a quieter E4.
                let left = input(&silence);
                let right = input(&synthetic_spectrum(
                    &params,
                    &[(57.0, -12.0), (69.0, -12.0), (64.0, -24.0)],
                ));
                let mut ribbon = NoteRibbon::new(
                    &device,
                    &queue,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    &uniform,
                    &left,
                    &right,
                );
                let state = analyze_and_read(&device, &queue, &ribbon, 1);
                assert_eq!(
                    state[12][0], 9.0,
                    "A should dominate for {layout:?}, focus {focus}"
                );
                assert!(state[9][1] > 0.99 && state[9][2] > 0.99);
                assert!(state[4][1] > 0.01 && state[4][1] < state[9][1]);

                // Exercise the actual font/gradient pipeline with both opaque
                // and transparent render targets, including narrow windows.
                for (width, height, clear, width_percent) in [
                    (1280, 640, wgpu::Color::BLACK, 76.0),
                    (1280, 640, wgpu::Color::BLACK, 20.0),
                    (1280, 640, wgpu::Color::BLACK, 100.0),
                    (320, 180, wgpu::Color::TRANSPARENT, 20.0),
                    (320, 180, wgpu::Color::TRANSPARENT, 100.0),
                ] {
                    ribbon.configure(&queue, width_percent, true, false);
                    params.resolution = [width as f32, height as f32];
                    queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                    let pixels = render_and_read(&device, &queue, &ribbon, width, height, clear);
                    assert!(
                        pixels.chunks_exact(4).any(|p| p[0] > 100 && p[2] > 100),
                        "ribbon must draw"
                    );
                    let occupied: Vec<_> = pixels
                        .chunks_exact(4)
                        .enumerate()
                        .filter(|(_, pixel)| pixel[..3].iter().any(|&value| value > 20))
                        .map(|(index, _)| (index % width as usize) as f32)
                        .collect();
                    let left_edge = occupied.iter().copied().fold(f32::INFINITY, f32::min);
                    let right_edge = occupied.iter().copied().fold(0.0, f32::max);
                    let expected_span = width as f32 * width_percent * 0.01;
                    assert!(
                        (right_edge - left_edge - expected_span).abs() < 40.0,
                        "visible ribbon span must follow the width slider"
                    );
                    assert!(
                        (left_edge + right_edge - width as f32).abs() < 16.0,
                        "ribbon must remain centered"
                    );
                    let row_bytes = width as usize * 4;
                    assert!(
                        pixels[(height as usize / 2) * row_bytes..]
                            .chunks_exact(4)
                            .all(|p| p[..3] == [0, 0, 0]),
                        "overlay must stay at top"
                    );
                    if width == 1280
                        && width_percent == 76.0
                        && layout == BarLayout::FixedPixels
                        && focus == 60.0
                    {
                        if let Ok(path) = std::env::var("VISUAL_AUDIO_NOTE_PREVIEW") {
                            let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
                            ppm.extend(pixels.chunks_exact(4).flat_map(|p| p[..3].iter().copied()));
                            std::fs::write(path, ppm).unwrap();
                        }
                    }
                }
                params.resolution = [1280.0, 640.0];
                params.perception[1] = 0.0;
                queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                // An equal-energy note change should become visibly dominant
                // within 50 ms at 60 Hz, without relying on a mapping reset.
                queue.write_buffer(
                    &right,
                    0,
                    bytemuck::cast_slice(&synthetic_spectrum(
                        &params,
                        &[(61.0, -12.0), (73.0, -12.0), (64.0, -24.0)],
                    )),
                );
                let quick_change = analyze_and_read(&device, &queue, &ribbon, 3);
                assert_eq!(quick_change[12][0], 1.0, "C# should take over within 50 ms");
                assert!(
                    quick_change[1][2] > 0.6,
                    "new highlight should rise promptly"
                );
                assert!(
                    quick_change[9][2] < 0.7,
                    "old highlight should fade promptly"
                );

                params.perception[1] = 1.0;
                queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                queue.write_buffer(
                    &right,
                    0,
                    bytemuck::cast_slice(&synthetic_spectrum(
                        &params,
                        &[(57.0, -12.0), (69.0, -12.0), (64.0, -24.0)],
                    )),
                );
                analyze_and_read(&device, &queue, &ribbon, 1);
                params.perception[1] = 0.0;
                queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                queue.write_buffer(&right, 0, bytemuck::cast_slice(&silence));
                let first_release = analyze_and_read(&device, &queue, &ribbon, 1);
                assert!(
                    first_release[9][0] > 0.0 && first_release[9][0] < state[9][0],
                    "release must be smooth"
                );
                let released = analyze_and_read(&device, &queue, &ribbon, 300);
                assert!(
                    released[..12]
                        .iter()
                        .all(|note| note[1] < 0.001 && note[2] < 0.001),
                    "silence must dim every note"
                );

                // A fresh C# must replace A, including after a mapping reset.
                params.perception[1] = 1.0;
                queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                queue.write_buffer(
                    &right,
                    0,
                    bytemuck::cast_slice(&synthetic_spectrum(&params, &[(61.0, -12.0)])),
                );
                let changed = analyze_and_read(&device, &queue, &ribbon, 1);
                assert_eq!(changed[12][0], 1.0);
                assert!(changed.iter().flatten().all(|x| x.is_finite()));

                // Loud bass C2 and treble C7 must not mask the quieter A4
                // when midrange focus is enabled. Switching ranges clears
                // stale activity rather than letting it linger through a fade.
                queue.write_buffer(
                    &right,
                    0,
                    bytemuck::cast_slice(&synthetic_spectrum(
                        &params,
                        &[(36.0, -1.0), (96.0, -1.0), (69.0, -18.0)],
                    )),
                );
                ribbon.configure(&queue, 76.0, false, false);
                let full_range = analyze_and_read(&device, &queue, &ribbon, 1);
                assert_eq!(full_range[12][0], 0.0, "C should dominate in full range");
                ribbon.configure(&queue, 76.0, true, false);
                params.perception[1] = 0.0;
                queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                let midrange = analyze_and_read(&device, &queue, &ribbon, 1);
                assert_eq!(midrange[12][0], 9.0, "midrange focus should reveal A");
                assert_eq!(midrange[0][0], 0.0, "excluded C energy must be cleared");
                assert!(
                    midrange[9][2] > 0.5,
                    "highlight should respond in its first frame"
                );

                // The spectrum remains frozen on A while independent audio
                // plays E4, then a quieter G4. This catches accidentally using
                // resonator output or smoothed old-note power in fast mode.
                ribbon.configure(&queue, 76.0, true, true);
                let mut audio: Vec<_> = (0..16384)
                    .map(|n| (std::f32::consts::TAU * 329.62756 * n as f32 / 48000.0).sin() * 0.8)
                    .collect();
                let silent_audio = vec![0.0; audio.len()];
                ribbon.update_audio(
                    &queue,
                    [&silent_audio, &audio],
                    audio.len(),
                    48000.0,
                    1.0,
                    [16.0, 22500.0],
                    false,
                );
                let fast = analyze_and_read(&device, &queue, &ribbon, 1);
                assert_eq!(
                    fast[12][0], 4.0,
                    "fast mode must read E from audio, not A from the spectrum"
                );
                assert!(fast[4][2] > 0.99, "fast highlight must appear immediately");
                for frame in 0..3 {
                    audio.copy_within(800.., 0);
                    let end = audio.len();
                    for (index, sample) in audio[end - 800..].iter_mut().enumerate() {
                        *sample =
                            (std::f32::consts::TAU * 391.99542 * (frame * 800 + index) as f32
                                / 48000.0)
                                .sin()
                                * 0.2;
                    }
                    ribbon.update_audio(
                        &queue,
                        [&silent_audio, &audio],
                        800,
                        48000.0,
                        1.0,
                        [16.0, 22500.0],
                        false,
                    );
                    let change = analyze_and_read(&device, &queue, &ribbon, 1);
                    if frame == 2 {
                        assert_eq!(change[12][0], 7.0, "quieter G must replace E within 50 ms");
                        assert!(change[7][2] > 0.99);
                        assert!(change[4][2] < 0.1, "old highlight must not linger");
                    }
                }
                audio.fill(0.0);
                ribbon.update_audio(
                    &queue,
                    [&silent_audio, &audio],
                    audio.len(),
                    48000.0,
                    1.0,
                    [16.0, 22500.0],
                    false,
                );
                let fast_silence = analyze_and_read(&device, &queue, &ribbon, 1);
                assert!(
                    fast_silence[..12]
                        .iter()
                        .all(|note| note[0] == 0.0 && note[1] == 0.0)
                );

                // Visual smoothing crossfades the label/glow but must leave
                // current note power and winner selection responsive.
                ribbon.set_smoothing(&queue, 100.0);
                for (index, sample) in audio.iter_mut().enumerate() {
                    *sample =
                        (std::f32::consts::TAU * 329.62756 * index as f32 / 48000.0).sin() * 0.8;
                }
                ribbon.update_audio(
                    &queue,
                    [&silent_audio, &audio],
                    audio.len(),
                    48000.0,
                    1.0,
                    [16.0, 22500.0],
                    false,
                );
                let smooth_start = analyze_and_read(&device, &queue, &ribbon, 1);
                assert_eq!(
                    smooth_start[12][0], 4.0,
                    "smoothing must not delay selecting E"
                );
                assert!(
                    (smooth_start[4][0] - fast[4][0]).abs() < 0.0001,
                    "visual smoothing must not change measured power"
                );
                assert!(smooth_start[4][1] > 0.05 && smooth_start[4][1] < 0.5);
                assert!(
                    smooth_start[4][2] > 0.05 && smooth_start[4][2] < 0.5,
                    "100 ms smoothing must soften the first-frame highlight"
                );
                let smooth_progress = analyze_and_read(&device, &queue, &ribbon, 6);
                assert!(smooth_progress[4][2] > smooth_start[4][2] && smooth_progress[4][2] < 0.95);
                ribbon.set_smoothing(&queue, 0.0);
                let immediate = analyze_and_read(&device, &queue, &ribbon, 1);
                assert!(
                    immediate[4][1] > 0.99 && immediate[4][2] > 0.99,
                    "zero smoothing must restore the immediate look without a reset"
                );

                ribbon.set_smoothing(&queue, 250.0);
                audio.fill(0.0);
                ribbon.update_audio(
                    &queue,
                    [&silent_audio, &audio],
                    audio.len(),
                    48000.0,
                    1.0,
                    [16.0, 22500.0],
                    false,
                );
                let smooth_release = analyze_and_read(&device, &queue, &ribbon, 1);
                assert_eq!(smooth_release[4][0], 0.0);
                assert!(
                    smooth_release[4][1] > 0.8 && smooth_release[4][1] < 1.0,
                    "smoothing should fade visible energy after audio stops"
                );
                let smooth_silence = analyze_and_read(&device, &queue, &ribbon, 180);
                assert!(
                    smooth_silence[..12]
                        .iter()
                        .all(|note| note[1] < 0.001 && note[2] < 0.001)
                );
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            assert!(
                device.pop_error_scope().await.is_none(),
                "note pipelines must validate"
            );
        });
    }

    fn synthetic_spectrum(params: &AnalysisParams, tones: &[(f32, f32)]) -> Vec<[f32; 4]> {
        let lower = params.frequency_range[0] as f64;
        let upper = params.frequency_range[1].min(params.sample_rate * 0.48) as f64;
        (0..params.bin_count)
            .map(|bin| {
                let (midi, t) = if params.perception[0] > 0.5 {
                    let first = (69.0 + 12.0 * (lower / 440.0).log2() - 0.0001).ceil();
                    let midi = first + bin as f64;
                    let hz = 440.0 * 2f64.powf((midi - 69.0) / 12.0);
                    (midi, (hz / lower).log2() / (upper / lower).log2())
                } else {
                    let display = bin as f64 / (params.bin_count - 1) as f64;
                    let origin = params.axis_focus[1] as f64;
                    let exponent = 4f64.powf(params.axis_focus[0] as f64 * 2.0 - 1.0);
                    let origin_logit = (origin / (1.0 - origin)).ln();
                    let t = if display == 0.0 || display == 1.0 {
                        display
                    } else {
                        1.0 / (1.0
                            + (-(origin_logit
                                + ((display / (1.0 - display)).ln() - origin_logit) / exponent))
                                .exp())
                    };
                    let hz = lower * (upper / lower).powf(t);
                    (69.0 + 12.0 * (hz / 440.0).log2(), t)
                };
                let power: f64 = tones
                    .iter()
                    .map(|&(note, db)| {
                        10f64.powf(db as f64 / 10.0)
                            * (-((midi - note as f64) / 0.22).powi(2)).exp()
                    })
                    .sum();
                let floor = params.db_scale[0] as f64
                    + (params.db_scale[1] - params.db_scale[0]) as f64 * t * t * (3.0 - 2.0 * t);
                let level = ((10.0 * power.max(1e-20).log10() - floor)
                    / (params.db_scale[2] as f64 - floor))
                    .clamp(0.0, 1.0);
                [level as f32, 0.0, 0.0, 0.0]
            })
            .collect()
    }

    fn read_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
        let (send, receive) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap()
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receive.recv().unwrap().unwrap();
        let bytes = buffer.slice(..).get_mapped_range().to_vec();
        buffer.unmap();
        bytes
    }

    fn analyze_and_read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ribbon: &NoteRibbon,
        frames: usize,
    ) -> Vec<[f32; 4]> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 13 * 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..frames {
            ribbon.analyze(&mut encoder);
        }
        encoder.copy_buffer_to_buffer(&ribbon._state, 0, &staging, 0, 13 * 16);
        queue.submit([encoder.finish()]);
        bytemuck::cast_slice(&read_buffer(device, &staging)).to_vec()
    }

    fn render_and_read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ribbon: &NoteRibbon,
        width: u32,
        height: u32,
        clear: wgpu::Color,
    ) -> Vec<u8> {
        render_and_read_mode(device, queue, ribbon, width, height, clear, false)
    }

    fn render_and_read_mode(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ribbon: &NoteRibbon,
        width: u32,
        height: u32,
        clear: wgpu::Color,
        merged: bool,
    ) -> Vec<u8> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let row_bytes = (width * 4).div_ceil(256) * 256;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row_bytes * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if merged {
                ribbon.draw(&mut clear_pass);
            }
        }
        if !merged {
            ribbon.render(&mut encoder, &view);
        }
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(height),
                },
            },
            texture.size(),
        );
        queue.submit([encoder.finish()]);
        read_buffer(device, &staging)
            .chunks_exact(row_bytes as usize)
            .flat_map(|row| row[..width as usize * 4].iter().copied())
            .collect()
    }
}
