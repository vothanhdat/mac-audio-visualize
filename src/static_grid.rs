//! Cache static frequency-line calculations in one full-precision pixel row.
//! The animated spectrum and its analysis still run at the original cadence.

use super::VisualizerSettings;
use winit::dpi::PhysicalSize;

pub(super) struct StaticGrid {
    pipeline: wgpu::RenderPipeline,
    cache_pipeline: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    width: u32,
    key: Option<(PhysicalSize<u32>, VisualizerSettings)>,
}

impl StaticGrid {
    pub(super) fn new(
        device: &wgpu::Device,
        analyzer_layout: &wgpu::BindGroupLayout,
        shader: &wgpu::ShaderModule,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("static grid column layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cached frequency grid layout"),
            bind_group_layouts: &[analyzer_layout, &texture_layout],
            push_constant_ranges: &[],
        });
        let pipeline = build_pipeline(device, &layout, shader, format, "grid_fs", true);
        let cache_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("frequency grid generation layout"),
            bind_group_layouts: &[analyzer_layout],
            push_constant_ranges: &[],
        });
        let cache_pipeline = build_pipeline(
            device,
            &cache_layout,
            shader,
            wgpu::TextureFormat::R32Float,
            "grid_cache_fs",
            false,
        );
        let (view, bind_group) = columns(device, &texture_layout, 1);
        Self {
            pipeline,
            cache_pipeline,
            texture_layout,
            view,
            bind_group,
            width: 1,
            key: None,
        }
    }

    /// Returns true only when the cached row was regenerated. Comparing all
    /// settings also makes future axis settings safe to add without stale lines.
    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        analyzer: &wgpu::BindGroup,
        size: PhysicalSize<u32>,
        settings: &VisualizerSettings,
    ) -> bool {
        if self
            .key
            .as_ref()
            .is_some_and(|(old_size, old_settings)| *old_size == size && old_settings == settings)
        {
            return false;
        }
        if self.width != size.width {
            (self.view, self.bind_group) = columns(device, &self.texture_layout, size.width);
            self.width = size.width;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("refresh static grid columns"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
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
            pass.set_pipeline(&self.cache_pipeline);
            pass.set_bind_group(0, analyzer, &[]);
            pass.draw(0..3, 0..1);
        }
        self.key = Some((size, settings.clone()));
        true
    }

    pub(super) fn draw(&self, pass: &mut wgpu::RenderPass<'_>, analyzer: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, analyzer, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn columns(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    width: u32,
) -> (wgpu::TextureView, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("full-precision static grid columns"),
        size: wgpu::Extent3d {
            width,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("static grid columns"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    (view, bind_group)
}

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    entry: &str,
    blend: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("grid_vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(entry),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
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
}

#[cfg(test)]
#[path = "static_grid_tests.rs"]
mod tests;
