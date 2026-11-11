//! Opt-in diagnostics. These report completed batches, including CPU encoding
//! and GPU execution, rather than treating system GPU percentage as GPU time.
use super::*;

#[test]
#[ignore = "requires Metal; profiles saved default hybrid/flat settings without modifying them"]
fn gpu_bar_width_cost_breakdown() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let saved = VisualizerSettings::load(&settings_path(None).unwrap());
        assert_eq!(
            saved.analysis_method,
            AnalysisMethod::HybridResonatorFft,
            "This diagnostic is for the hybrid pipeline"
        );
        assert_eq!(
            saved.render_mode,
            RenderMode::Flat,
            "This diagnostic measures flat drawing"
        );
        let (compute_layout, render_layout) = analyzer_layouts(&device);
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let resonator = build_resonator_pipeline(&device, &compute_pipeline_layout);
        let fft_map = build_hybrid_fft_pipeline(&device, &compute_pipeline_layout);
        let analysis_shader = device.create_shader_module(wgpu::include_wgsl!("analysis.wgsl"));
        let compose = compute_pipeline(
            &device,
            &compute_pipeline_layout,
            &analysis_shader,
            "compose",
            "compose_display",
        );
        let shader = device.create_shader_module(wgpu::include_wgsl!("visualizer.wgsl"));
        let pipelines = build_bar_pipelines(
            &device,
            &render_pipeline_layout,
            &shader,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        let reference_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("original bar renderer"),
            source: wgpu::ShaderSource::Wgsl(reference_bar_source().into()),
        });
        let reference_pipelines = build_bar_pipelines(
            &device,
            &render_pipeline_layout,
            &reference_shader,
            wgpu::TextureFormat::Bgra8UnormSrgb,
        );
        for size in [PhysicalSize::new(640, 360), PhysicalSize::new(1920, 1080)] {
            for width in [1.0, 10.0] {
                let settings = VisualizerSettings {
                    bar_width_px: width,
                    ..saved.clone()
                };
                let bins = display_bin_count(&settings, size, 48000.0);
                let count = settings.channel_mode.channel_count();
                let params = analysis_params(
                    &settings,
                    [48000.0, 0.0, 1.0 / 60.0],
                    bins,
                    false,
                    ([800, 400, 200, 100, 50], false),
                    [0.0; 4],
                    size,
                );
                let analyzers: Vec<_> = (0..count)
                    .map(|index| {
                        create_analyzer_channel(
                            &device,
                            &compute_layout,
                            &render_layout,
                            &params,
                            HISTORY_SAMPLE_COUNT,
                            index,
                            count as f32,
                            &settings,
                        )
                    })
                    .collect();
                for (index, analyzer) in analyzers.iter().enumerate() {
                    let channel = RenderChannelParams {
                        channel_index: index as f32,
                        channel_count: count as f32,
                        bottom_mirror: f32::from(settings.bottom_mirror),
                        stereo_treble_center: f32::from(settings.stereo_treble_center),
                        mirror_baseline_percent: settings.mirror_baseline_percent,
                        mirror_initial_fade_percent: settings.mirror_initial_fade_percent,
                        mirror_fade_percent: settings.mirror_fade_percent,
                        axis_labels_overlay: f32::from(settings.axis_labels_overlay),
                        axis_label_mode: settings.axis_label_mode.shader_value(),
                        show_frequency_grid: f32::from(settings.show_frequency_grid),
                        palette: shader_palette(&settings),
                        ..bytemuck::Zeroable::zeroed()
                    };
                    queue.write_buffer(
                        &analyzer.render_channel_buffer,
                        0,
                        bytemuck::bytes_of(&channel),
                    );
                    let mut samples = vec![0.0f32; SAMPLE_COUNT];
                    for (i, sample) in samples.iter_mut().enumerate() {
                        *sample = 0.15 * (std::f32::consts::TAU * 440.0 * i as f32 / 48000.0).sin();
                    }
                    queue.write_buffer(&analyzer.sample_buffer, 0, bytemuck::cast_slice(&samples));
                    let mut resampler = ResonatorResampler::default();
                    resampler.process(
                        &samples[SAMPLE_COUNT - 800..],
                        true,
                        settings.resonator_five_stage,
                        settings.resonator_timestamp_alignment,
                    );
                    if settings.resonator_multirate {
                        for (stream, buffer) in resampler.streams().iter().zip([
                            &analyzer.resonator_half_buffer,
                            &analyzer.resonator_quarter_buffer,
                            &analyzer.resonator_eighth_buffer,
                            &analyzer.resonator_sixteenth_buffer,
                        ]) {
                            if !stream.is_empty() {
                                queue.write_buffer(buffer, 0, bytemuck::cast_slice(stream));
                            }
                        }
                    }
                    let mut initial = params;
                    initial.capture[2] = 1;
                    queue.write_buffer(&analyzer.params_buffer, 0, bytemuck::bytes_of(&initial));
                }
                let encode_analysis = |encoder: &mut wgpu::CommandEncoder| {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    for analyzer in &analyzers {
                        pass.set_pipeline(&resonator);
                        pass.set_bind_group(0, &analyzer.compute_bind_group, &[]);
                        pass.dispatch_workgroups(bins.div_ceil(64), 1, 1);
                        pass.set_pipeline(&fft_map);
                        pass.set_bind_group(0, &analyzer.fft_compute_bind_group, &[]);
                        pass.dispatch_workgroups(bins.div_ceil(64), 1, 1);
                        pass.set_pipeline(&compose);
                        pass.dispatch_workgroups(bins.div_ceil(64), 1, 1);
                    }
                };
                let mut warm_up = device.create_command_encoder(&Default::default());
                encode_analysis(&mut warm_up);
                queue.submit([warm_up.finish()]);
                for analyzer in &analyzers {
                    queue.write_buffer(&analyzer.params_buffer, 0, bytemuck::bytes_of(&params));
                }
                let compute_ms = completed_batch(&device, &queue, encode_analysis);
                // Equal displayed energy isolates width/count from changes in
                // the input music. Production shaders and blending are used.
                let level = vec![[0.65f32, 0.0, 0.65, 0.0]; bins as usize];
                let mut seeded_params = params;
                seeded_params.perception[1] = 1.0;
                for analyzer in &analyzers {
                    queue.write_buffer(&analyzer.spectrum_buffer, 0, bytemuck::cast_slice(&level));
                    queue.write_buffer(
                        &analyzer.params_buffer,
                        0,
                        bytemuck::bytes_of(&seeded_params),
                    );
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&compose);
                    for analyzer in &analyzers {
                        pass.set_bind_group(0, &analyzer.compute_bind_group, &[]);
                        pass.dispatch_workgroups(bins.div_ceil(64), 1, 1);
                    }
                }
                queue.submit([encoder.finish()]);
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                let draw_ms = completed_batch(&device, &queue, |encoder| {
                    draw_bar_layers(
                        encoder,
                        &view,
                        wgpu::Color::BLACK,
                        &analyzers,
                        &pipelines,
                        bins,
                        settings.bottom_mirror,
                    );
                });
                let old_draw_ms = completed_batch(&device, &queue, |encoder| {
                    draw_bar_layers(
                        encoder,
                        &view,
                        wgpu::Color::BLACK,
                        &analyzers,
                        &reference_pipelines,
                        bins,
                        settings.bottom_mirror,
                    );
                });
                eprintln!(
                    "{}x{} width={width}px bar render original {old_draw_ms:.3}ms -> optimized {draw_ms:.3}ms (CPU + GPU completion)",
                    size.width, size.height
                );
                if size.width == 1920 && width == 1.0 {
                    for (label, pipeline) in ["bar glow", "bar bodies", "peak glow", "peak markers"]
                        .into_iter()
                        .zip(&pipelines)
                    {
                        let ms = completed_batch(&device, &queue, |encoder| {
                            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: Some(label),
                                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                    view: &view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                        store: wgpu::StoreOp::Store,
                                    },
                                })],
                                depth_stencil_attachment: None,
                                timestamp_writes: None,
                                occlusion_query_set: None,
                            });
                            pass.set_pipeline(pipeline);
                            for analyzer in &analyzers {
                                pass.set_bind_group(0, &analyzer.render_bind_group, &[]);
                                pass.draw(
                                    0..6,
                                    0..bins * if settings.bottom_mirror { 2 } else { 1 },
                                );
                            }
                        });
                        eprintln!(
                            "1080p 1px {label}: {ms:.3} ms (isolated layer, CPU + GPU completion)"
                        );
                    }
                }
                eprintln!(
                    "{}x{} width={width}px channels={count} bins/channel={bins} workgroups/channel={} analysis={compute_ms:.3}ms bar-drawing={draw_ms:.3}ms (CPU encoding + GPU completion; synthetic audio, equal-height bars; excludes grid, labels, note ribbon, CPU FFT, presentation)",
                    size.width,
                    size.height,
                    bins.div_ceil(64)
                );
            }
        }
    });
}

pub(super) fn completed_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encode: impl Fn(&mut wgpu::CommandEncoder),
) -> f64 {
    const FRAMES: usize = 48;
    let mut samples = Vec::new();
    for _ in 0..5 {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..FRAMES {
            encode(&mut encoder);
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        samples.push(start.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64);
    }
    samples.sort_by(f64::total_cmp);
    samples[2]
}

fn build_bar_pipelines(
    device: &wgpu::Device,
    render_pipeline_layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> Vec<wgpu::RenderPipeline> {
    let additive = wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent::OVER,
    };
    let pipelines: Vec<_> = [
        ("glow_vs", "glow_fs", additive),
        ("vs_main", "fs_main", wgpu::BlendState::ALPHA_BLENDING),
        ("peak_glow_vs", "peak_glow_fs", additive),
        ("peak_vs", "peak_fs", wgpu::BlendState::ALPHA_BLENDING),
    ]
    .into_iter()
    .map(|(vs, fs, blend)| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(fs),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(vs),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(fs),
                targets: &[Some(wgpu::ColorTargetState {
                    format: format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        })
    })
    .collect();
    pipelines
}

fn draw_bar_layers(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    clear: wgpu::Color,
    analyzers: &[AnalyzerChannel],
    pipelines: &[wgpu::RenderPipeline],
    bins: u32,
    mirror: bool,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
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
    for analyzer in analyzers {
        pass.set_bind_group(0, &analyzer.render_bind_group, &[]);
        for pipeline in pipelines {
            pass.set_pipeline(pipeline);
            pass.draw(0..6, 0..bins * if mirror { 2 } else { 1 });
        }
    }
}

fn function_bounds(source: &str, name: &str) -> (usize, usize) {
    let start = source.find(&format!("fn {name}(")).unwrap();
    let open = source[start..].find('{').unwrap() + start;
    let mut depth = 1;
    let mut end = open + 1;
    while depth > 0 {
        match source.as_bytes()[end] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        end += 1;
    }
    (start, end)
}

fn reference_bar_source() -> String {
    let reference = include_str!("testdata/bar_render_reference.wgsl");
    let mut source = include_str!("visualizer.wgsl").to_owned();
    for name in ["vs_main", "fs_main", "glow_vs", "glow_fs", "peak_glow_vs"] {
        let (start, end) = function_bounds(&source, name);
        let (old_start, old_end) = function_bounds(reference, name);
        source.replace_range(start..end, &reference[old_start..old_end]);
    }
    source
}

#[test]
#[ignore = "requires Metal; compares optimized bar pixels with the previous renderer"]
fn gpu_bar_render_optimization_preserves_pixels() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let (compute_layout, render_layout) = analyzer_layouts(&device);
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&compute_layout],
                push_constant_ranges: &[],
            });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_layout],
            push_constant_ranges: &[],
        });
        let analysis_shader = device.create_shader_module(wgpu::include_wgsl!("analysis.wgsl"));
        let compose = compute_pipeline(
            &device,
            &compute_pipeline_layout,
            &analysis_shader,
            "display",
            "compose_display",
        );
        let shader = device.create_shader_module(wgpu::include_wgsl!("visualizer.wgsl"));
        let reference_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(reference_bar_source().into()),
        });
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let reference = build_bar_pipelines(&device, &layout, &reference_shader, format);
        let optimized = build_bar_pipelines(&device, &layout, &shader, format);
        let mut cases = 0;
        let mut largest_delta = 0;
        for size in [PhysicalSize::new(640, 360), PhysicalSize::new(1920, 1080)] {
            for channel_mode in [ChannelMode::Mono, ChannelMode::StereoMirror] {
                for mirror in [false, true] {
                    for width in [1.0, 3.0, 10.0] {
                        let settings = VisualizerSettings {
                            channel_mode,
                            bottom_mirror: mirror,
                            bar_width_px: width,
                            bar_top_fade_percent: if width == 3.0 { 0.0 } else { 45.0 },
                            rise_highlight_strength: 1.35,
                            mirror_fade_percent: if width == 10.0 { 2000.0 } else { 100.0 },
                            ..Default::default()
                        };
                        let bins = display_bin_count(&settings, size, 48000.0);
                        let count = channel_mode.channel_count();
                        let mut params = analysis_params(
                            &settings,
                            [48000.0, 0.0, 1.0 / 60.0],
                            bins,
                            true,
                            ([0; 5], false),
                            [0.0; 4],
                            size,
                        );
                        let analyzers: Vec<_> = (0..count)
                            .map(|index| {
                                create_analyzer_channel(
                                    &device,
                                    &compute_layout,
                                    &render_layout,
                                    &params,
                                    HISTORY_SAMPLE_COUNT,
                                    index,
                                    count as f32,
                                    &settings,
                                )
                            })
                            .collect();
                        for (index, analyzer) in analyzers.iter().enumerate() {
                            let channel = RenderChannelParams {
                                channel_index: index as f32,
                                channel_count: count as f32,
                                bottom_mirror: f32::from(mirror),
                                stereo_treble_center: f32::from(width == 1.0),
                                mirror_baseline_percent: settings.mirror_baseline_percent,
                                mirror_initial_fade_percent: settings.mirror_initial_fade_percent,
                                mirror_fade_percent: settings.mirror_fade_percent,
                                palette: shader_palette(&settings),
                                ..bytemuck::Zeroable::zeroed()
                            };
                            queue.write_buffer(
                                &analyzer.render_channel_buffer,
                                0,
                                bytemuck::bytes_of(&channel),
                            );
                        }
                        for pattern in 0..3 {
                            params.perception[1] = 1.0;
                            for analyzer in &analyzers {
                                queue.write_buffer(
                                    &analyzer.spectrum_buffer,
                                    0,
                                    bytemuck::cast_slice(&vec![[0.0f32; 4]; bins as usize]),
                                );
                                queue.write_buffer(
                                    &analyzer.params_buffer,
                                    0,
                                    bytemuck::bytes_of(&params),
                                );
                            }
                            let mut encoder = device.create_command_encoder(&Default::default());
                            compose_for(&mut encoder, &compose, &analyzers, bins);
                            queue.submit([encoder.finish()]);
                            params.perception[1] = f32::from(pattern != 2);
                            for (channel, analyzer) in analyzers.iter().enumerate() {
                                let levels: Vec<[f32; 4]> = (0..bins)
                                    .map(|bin| {
                                        let level = match pattern {
                                            0 => 0.0,
                                            1 => [0.25, 0.45, 0.4500001, 0.65, 0.85, 1.0]
                                                [(bin as usize + channel) % 6],
                                            _ => {
                                                0.1 + 0.9 * ((bin as usize + channel) % 17) as f32
                                                    / 16.0
                                            }
                                        };
                                        [level, 0.0, level, 0.0]
                                    })
                                    .collect();
                                queue.write_buffer(
                                    &analyzer.spectrum_buffer,
                                    0,
                                    bytemuck::cast_slice(&levels),
                                );
                                queue.write_buffer(
                                    &analyzer.params_buffer,
                                    0,
                                    bytemuck::bytes_of(&params),
                                );
                            }
                            let mut encoder = device.create_command_encoder(&Default::default());
                            compose_for(&mut encoder, &compose, &analyzers, bins);
                            queue.submit([encoder.finish()]);
                            for clear in [wgpu::Color::BLACK, wgpu::Color::TRANSPARENT] {
                                let expected = read_bar_render(
                                    &device, &queue, size, clear, &analyzers, &reference, bins,
                                    mirror,
                                );
                                let actual = read_bar_render(
                                    &device, &queue, size, clear, &analyzers, &optimized, bins,
                                    mirror,
                                );
                                let delta = expected
                                    .iter()
                                    .zip(&actual)
                                    .map(|(a, b)| a.abs_diff(*b))
                                    .max()
                                    .unwrap();
                                if delta > 1 {
                                    for layer in 0..4 {
                                        let old = read_bar_render(
                                            &device,
                                            &queue,
                                            size,
                                            clear,
                                            &analyzers,
                                            &reference[layer..layer + 1],
                                            bins,
                                            mirror,
                                        );
                                        let new = read_bar_render(
                                            &device,
                                            &queue,
                                            size,
                                            clear,
                                            &analyzers,
                                            &optimized[layer..layer + 1],
                                            bins,
                                            mirror,
                                        );
                                        let difference = old
                                            .iter()
                                            .zip(new)
                                            .map(|(a, b)| a.abs_diff(b))
                                            .max()
                                            .unwrap();
                                        eprintln!("layer {layer} max difference {difference}");
                                    }
                                }
                                assert!(
                                    delta <= 1,
                                    "bar pixels changed: {size:?}, {channel_mode:?}, mirror {mirror}, width {width}, pattern {pattern}, max delta {delta}"
                                );
                                largest_delta = largest_delta.max(delta);
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        eprintln!("Bar renderer: {cases} cases; largest channel difference {largest_delta}/255.");
    });
}

fn compose_for(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    analyzers: &[AnalyzerChannel],
    bins: u32,
) {
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    for analyzer in analyzers {
        pass.set_bind_group(0, &analyzer.compute_bind_group, &[]);
        pass.dispatch_workgroups(bins.div_ceil(64), 1, 1);
    }
}

fn read_bar_render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: PhysicalSize<u32>,
    clear: wgpu::Color,
    analyzers: &[AnalyzerChannel],
    pipelines: &[wgpu::RenderPipeline],
    bins: u32,
    mirror: bool,
) -> Vec<u8> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let row_bytes = (size.width * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row_bytes * size.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    draw_bar_layers(
        &mut encoder,
        &view,
        clear,
        analyzers,
        pipelines,
        bins,
        mirror,
    );
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(size.height),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    let (send, receive) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).unwrap()
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    receive.recv().unwrap().unwrap();
    let pixels = buffer
        .slice(..)
        .get_mapped_range()
        .chunks_exact(row_bytes as usize)
        .flat_map(|row| row[..size.width as usize * 4].iter().copied())
        .collect();
    buffer.unmap();
    pixels
}
