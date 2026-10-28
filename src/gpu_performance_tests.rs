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
                        format: wgpu::TextureFormat::Bgra8UnormSrgb,
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
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
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
                    for analyzer in &analyzers {
                        pass.set_bind_group(0, &analyzer.render_bind_group, &[]);
                        for pipeline in &pipelines {
                            pass.set_pipeline(pipeline);
                            pass.draw(0..6, 0..bins * if settings.bottom_mirror { 2 } else { 1 });
                        }
                    }
                });
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

fn completed_batch(
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
