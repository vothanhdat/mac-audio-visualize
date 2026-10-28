use super::super::{
    AnalysisMethod, AxisLabelMode, BarLayout, ChannelMode, HISTORY_SAMPLE_COUNT,
    RenderChannelParams, analysis_params, analyzer_layouts, build_axis_overlay_pipeline,
    create_analyzer_channel, display_bin_count,
};
use super::*;

#[test]
#[ignore = "requires a Metal GPU; compares pixels and reports isolated draw timing"]
fn gpu_cached_grid_preserves_pixels_and_reuses_static_work() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        let format = wgpu::TextureFormat::Bgra8UnormSrgb;
        let (compute_layout, render_layout) = analyzer_layouts(&device);
        let shader = device.create_shader_module(wgpu::include_wgsl!("visualizer.wgsl"));
        let source = include_str!("visualizer.wgsl");
        let start = source.find("@fragment\nfn grid_fs").unwrap();
        let end = source.find("@fragment\nfn axis_overlay_fs").unwrap();
        let reference_source = format!(
            "{}{}{}",
            &source[..start],
            include_str!("testdata/grid_reference.wgsl"),
            &source[end..]
        );
        let reference_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("original uncached frequency grid"),
            source: wgpu::ShaderSource::Wgsl(reference_source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_layout],
            push_constant_ranges: &[],
        });
        let reference =
            build_pipeline(&device, &layout, &reference_shader, format, "grid_fs", true);
        let overlay = build_axis_overlay_pipeline(&device, &layout, format);
        let mut cache = StaticGrid::new(&device, &render_layout, &shader, format);
        let settings = VisualizerSettings::default();
        let size = PhysicalSize::new(640, 360);
        let params = analysis_params(
            &settings,
            [48000.0, 0.0, 0.016],
            display_bin_count(&settings, size, 48000.0),
            true,
            ([0; 5], true),
            [0.0; 4],
            size,
        );
        let analyzer = create_analyzer_channel(
            &device,
            &compute_layout,
            &render_layout,
            &params,
            HISTORY_SAMPLE_COUNT,
            0,
            1.0,
            &settings,
        );
        let mut cases = 0;
        let mut changed_bytes = 0;
        let mut largest_delta = 0;
        for size in [
            PhysicalSize::new(640, 360),
            PhysicalSize::new(641, 359),
            PhysicalSize::new(1920, 1080),
        ] {
            for labels in [AxisLabelMode::Hertz, AxisLabelMode::Notes] {
                for bars in [BarLayout::FixedPixels, BarLayout::MusicalNotes] {
                    for channel in [ChannelMode::Mono, ChannelMode::StereoMirror] {
                        for overlaid in [false, true] {
                            for transparent in [false, true] {
                                let settings = VisualizerSettings {
                                    axis_label_mode: labels,
                                    bar_layout: bars,
                                    channel_mode: channel,
                                    axis_labels_overlay: overlaid,
                                    transparent_background: transparent,
                                    stereo_treble_center: cases % 3 == 0,
                                    show_frequency_grid: cases % 5 != 0,
                                    frequency_focus_strength_percent: if cases % 3 == 0 {
                                        50.0
                                    } else {
                                        72.0
                                    },
                                    frequency_focus_origin_percent: 37.0,
                                    min_frequency_hz: if cases % 7 == 0 { 130.8128 } else { 16.0 },
                                    analysis_method: AnalysisMethod::MultirateFft,
                                    ..VisualizerSettings::default()
                                };
                                let sample_rate = if cases % 3 == 0 { 44100.0 } else { 48000.0 };
                                update_params(
                                    &queue,
                                    &analyzer.params_buffer,
                                    &analyzer.render_channel_buffer,
                                    &settings,
                                    size,
                                    sample_rate,
                                );
                                let clear = if transparent {
                                    wgpu::Color::TRANSPARENT
                                } else {
                                    wgpu::Color::BLACK
                                };
                                let expected = render(
                                    &device,
                                    &queue,
                                    size,
                                    clear,
                                    &reference,
                                    &overlay,
                                    &analyzer.render_bind_group,
                                    None,
                                    &settings,
                                );
                                let actual = render(
                                    &device,
                                    &queue,
                                    size,
                                    clear,
                                    &reference,
                                    &overlay,
                                    &analyzer.render_bind_group,
                                    Some(&mut cache),
                                    &settings,
                                );
                                let differences =
                                    expected.iter().zip(&actual).filter(|(a, b)| a != b).count();
                                let max_delta = expected
                                    .iter()
                                    .zip(&actual)
                                    .map(|(a, b)| a.abs_diff(*b))
                                    .max()
                                    .unwrap();
                                // Moving an intermediate to an R32Float target
                                // may change final 8-bit rounding by one step.
                                assert!(
                                    max_delta <= 1,
                                    "case {cases}: {size:?}, labels {labels:?}, bars {bars:?}, channel {channel:?}, overlay {overlaid}, transparent {transparent}: {differences} changed bytes, max delta {max_delta}"
                                );
                                changed_bytes += differences;
                                largest_delta = largest_delta.max(max_delta);
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "Cached grid: {cases} configurations; max channel difference {largest_delta}/255 across {changed_bytes} bytes."
        );

        {
            let size = PhysicalSize::new(1920, 1080);
            for labels in [AxisLabelMode::Hertz, AxisLabelMode::Notes] {
                let settings = VisualizerSettings {
                    axis_label_mode: labels,
                    ..VisualizerSettings::default()
                };
                update_params(
                    &queue,
                    &analyzer.params_buffer,
                    &analyzer.render_channel_buffer,
                    &settings,
                    size,
                    48000.0,
                );
                let mut encoder = device.create_command_encoder(&Default::default());
                assert!(cache.prepare(
                    &device,
                    &mut encoder,
                    &analyzer.render_bind_group,
                    size,
                    &settings
                ));
                queue.submit([encoder.finish()]);
                let before = timed_grid(
                    &device,
                    &queue,
                    size,
                    &reference,
                    &analyzer.render_bind_group,
                    None,
                );
                let after = timed_grid(
                    &device,
                    &queue,
                    size,
                    &reference,
                    &analyzer.render_bind_group,
                    Some(&cache),
                );
                eprintln!(
                    "1080p {labels:?} isolated grid completion time (CPU + GPU): original {before:.3} ms, cached {after:.3} ms ({:.1}x). Grid pass only; excludes analyzer, bars, and cache regeneration.",
                    before / after
                );
            }
        }
    });
}

fn update_params(
    queue: &wgpu::Queue,
    params_buffer: &wgpu::Buffer,
    channel_buffer: &wgpu::Buffer,
    settings: &VisualizerSettings,
    size: PhysicalSize<u32>,
    sample_rate: f32,
) {
    let params = analysis_params(
        settings,
        [sample_rate, 0.0, 0.016],
        display_bin_count(settings, size, sample_rate),
        true,
        ([0; 5], true),
        [0.0; 4],
        size,
    );
    queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&params));
    let channel = RenderChannelParams {
        channel_count: settings.channel_mode.channel_count() as f32,
        stereo_treble_center: f32::from(settings.stereo_treble_center),
        axis_labels_overlay: f32::from(settings.axis_labels_overlay),
        axis_label_mode: settings.axis_label_mode.shader_value(),
        show_frequency_grid: f32::from(settings.show_frequency_grid),
        ..bytemuck::Zeroable::zeroed()
    };
    queue.write_buffer(channel_buffer, 0, bytemuck::bytes_of(&channel));
}

fn target(device: &wgpu::Device, size: PhysicalSize<u32>) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("grid comparison target"),
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
    })
}

fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
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

fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: PhysicalSize<u32>,
    clear: wgpu::Color,
    reference: &wgpu::RenderPipeline,
    overlay: &wgpu::RenderPipeline,
    analyzer: &wgpu::BindGroup,
    cache: Option<&mut StaticGrid>,
    settings: &VisualizerSettings,
) -> Vec<u8> {
    let texture = target(device, size);
    let view = texture.create_view(&Default::default());
    let row_bytes = (size.width * 4).div_ceil(256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row_bytes * size.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let cached = cache.is_some();
    if let Some(cache) = cache {
        assert!(cache.prepare(device, &mut encoder, analyzer, size, settings));
        assert!(!cache.prepare(device, &mut encoder, analyzer, size, settings));
        draw(
            &mut encoder,
            &view,
            clear,
            reference,
            analyzer,
            Some(cache),
            None,
        );
    } else {
        draw(&mut encoder, &view, clear, reference, analyzer, None, None);
    }
    if settings.axis_labels_overlay {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
        if cached {
            let y = size.height.saturating_sub(14);
            pass.set_scissor_rect(0, y, size.width, 10.min(size.height - y));
        }
        pass.set_pipeline(overlay);
        pass.set_bind_group(0, analyzer, &[]);
        pass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(size.height),
            },
        },
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    read(device, &staging)
        .chunks_exact(row_bytes as usize)
        .flat_map(|row| row[..size.width as usize * 4].iter().copied())
        .collect()
}

fn draw(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    clear: wgpu::Color,
    reference: &wgpu::RenderPipeline,
    analyzer: &wgpu::BindGroup,
    cache: Option<&StaticGrid>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("grid comparison pass"),
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
        timestamp_writes,
        occlusion_query_set: None,
    });
    if let Some(cache) = cache {
        cache.draw(&mut pass, analyzer);
    } else {
        pass.set_pipeline(reference);
        pass.set_bind_group(0, analyzer, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn timed_grid(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: PhysicalSize<u32>,
    reference: &wgpu::RenderPipeline,
    analyzer: &wgpu::BindGroup,
    cache: Option<&StaticGrid>,
) -> f64 {
    const FRAMES: u32 = 24;
    let texture = target(device, size);
    let view = texture.create_view(&Default::default());
    // Device timestamps are unreliable on some Metal drivers. Measure a
    // completed batch instead, including CPU submission overhead for both paths.
    let mut timings = Vec::new();
    for _ in 0..3 {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let start = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..FRAMES {
            draw(
                &mut encoder,
                &view,
                wgpu::Color::BLACK,
                reference,
                analyzer,
                cache,
                None,
            );
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        timings.push(start.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64);
    }
    timings.sort_by(f64::total_cmp);
    timings[1]
}
