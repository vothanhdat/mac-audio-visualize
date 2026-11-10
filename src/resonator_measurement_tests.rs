//! Numerical regression tests against an independent, exponentially weighted
//! least-squares fit. The GPU probe calls the production WGSL measurement helper.
use super::*;

struct Case {
    input: [f32; 12],
    expected: f64,
    check_fit: bool,
    check_tone: bool,
    gain: f64,
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (frequency, rate) in [
        (16.0, 3000.0),
        (16.0, 384000.0),
        (50.0, 48000.0),
        (800.0, 3000.0),
        (10000.0, 48000.0),
        (22500.0, 48000.0),
    ] {
        for q in [0.5, 10.0, 32.0, 200.0, 512.0] {
            for cycles in [0.0, 0.01, 0.25, 0.5, 1.0, 2.0, 8.0, 64.0] {
                let phase = std::f64::consts::TAU * frequency / rate;
                let exponent = phase / (2.0 * q);
                let weight = -(-exponent).exp_m1();
                let damping = 1.0 - weight;
                let count = (cycles * rate / frequency).round() as usize;
                for signal in 0..6 {
                    let (mut re, mut im) = (0.0, 0.0);
                    let (mut cc, mut ss, mut cs) = (0.0, 0.0, 0.0);
                    let (mut last_sample, mut position, mut previous_position) = (0.0, 0.0, 0.0);
                    for index in 0..count {
                        let angle = phase * index as f64;
                        let (s, c) = angle.sin_cos();
                        let sample = match signal {
                            0..=2 => 0.5 * (angle + signal as f64 * 0.7).cos(),
                            3 => 0.5 * (angle * 1.07 + 0.4).cos(),
                            4 => 0.35 * (angle * 0.98).cos() + 0.15 * (angle * 1.1).cos(),
                            _ => {
                                if index >= count / 2 {
                                    0.5 * (angle + 0.7).cos()
                                } else {
                                    0.0
                                }
                            }
                        };
                        previous_position = position;
                        last_sample = sample;
                        re = damping * re + weight * c * sample;
                        im = damping * im - weight * s * sample;
                        position = re * c - im * s;
                        cc = damping * cc + weight * c * c;
                        ss = damping * ss + weight * s * s;
                        cs = damping * cs + weight * c * s;
                    }
                    let gain = cc + ss;
                    let determinant = cc * ss - cs * cs;
                    let conditioned = gain > 0.0 && 4.0 * determinant / (gain * gain) >= 0.201;
                    let expected = if conditioned {
                        let cosine_fit = (ss * re + cs * im) / determinant;
                        let sine_fit = (-cs * re - cc * im) / determinant;
                        gain * cosine_fit.hypot(sine_fit)
                    } else {
                        0.0
                    };
                    let velocity = (position * phase.cos() - previous_position) / phase.sin();
                    let response_den = (1.0 - damping * (2.0 * phase).cos()).powi(2)
                        + (damping * (2.0 * phase).sin()).powi(2);
                    let h_re = weight * (1.0 - damping * (2.0 * phase).cos()) / response_den;
                    let h_im = weight * damping * (2.0 * phase).sin() / response_den;
                    let calibration = (1.0 + h_re).hypot(h_im);
                    let energy_amplitude = 2.0 * position.hypot(velocity) / calibration;
                    let (os, oc) = (phase * count as f64).sin_cos();
                    cases.push(Case {
                        input: [
                            re as f32,
                            im as f32,
                            oc as f32,
                            -os as f32,
                            phase.cos() as f32,
                            -phase.sin() as f32,
                            phase as f32,
                            exponent as f32,
                            weight as f32,
                            count as f32,
                            last_sample as f32,
                            energy_amplitude as f32,
                        ],
                        expected,
                        check_fit: conditioned,
                        check_tone: conditioned
                            && signal < 3
                            && gain > 0.1
                            && exponent * (count as f64) < 16.0,
                        gain,
                    });
                }
            }
        }
    }
    cases
}

#[test]
#[ignore = "requires a Metal GPU"]
fn phase_corrected_magnitude_matches_weighted_tone_fit() {
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
        let cases = cases();
        let inputs: Vec<f32> = cases.iter().flat_map(|case| case.input).collect();
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tone fit fixtures"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tone fit results"),
            size: cases.len() as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: output.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = analysis_params(
            &VisualizerSettings::default(),
            [48000.0, 0.0, 0.0],
            cases.len() as u32,
            false,
            ([0; 5], false),
            [0.0; 4],
            PhysicalSize::new(320, 180),
        );
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (layout, _) = analyzer_layouts(&device);
        let dummy = storage_buffer(&device, "unused probe binding", 16);
        let bindings = create_compute_channel_bind_group(
            &device,
            &layout,
            "probe",
            &input,
            [
                &output, &uniform, &dummy, &dummy, &dummy, &dummy, &dummy, &dummy,
            ],
        );
        // Keep the old formula in the probe to measure the improvement using
        // identical inputs and GPU precision, independently of the CPU fit.
        let source = format!(
            "{}\n{}",
            include_str!("resonator.wgsl"),
            r#"
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.bin_count) { return; }
    let b = i * 12u;
    let z = vec2(samples[b], samples[b + 1u]);
    let o = vec2(samples[b + 2u], samples[b + 3u]);
    let r = vec2(samples[b + 4u], samples[b + 5u]);
    let phase = samples[b + 6u];
    let exponent = samples[b + 7u];
    let weight = samples[b + 8u];
    let count = samples[b + 9u];
    let actual = measured_phasor(z, o, r, phase, exponent, weight, count);
    let d = exp(-exponent);
    let double_rotation = complex_multiply(complex_conjugate(r), complex_conjugate(r));
    let den = vec2(1.0, 0.0) - d * double_rotation;
    let response = (1.0 - d) * complex_conjugate(den) / max(dot(den, den), 0.000001);
    let previous = complex_multiply(o, complex_conjugate(r));
    let image = complex_multiply(response, complex_multiply(previous, previous));
    let old = (z - complex_multiply(image, complex_conjugate(z))) / max(1.0 - dot(image, image), 0.05);
    spectrum[i] = vec4(2.0 * length(actual), 2.0 * length(old), 2.0 * length(z), one_minus_exp_negative(exponent));
}
@compute @workgroup_size(64)
fn energy_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.bin_count) { return; }
    let b = i * 12u;
    let z = vec2(samples[b], samples[b + 1u]);
    let o = vec2(samples[b + 2u], samples[b + 3u]);
    let r = vec2(samples[b + 4u], samples[b + 5u]);
    let energy = position_velocity_magnitude(z, o, r, samples[b + 8u], samples[b + 10u]);
    let previous = complex_multiply(o, complex_conjugate(r));
    let physical = complex_multiply(z, complex_conjugate(previous));
    spectrum[i] = vec4(energy, samples[b + 11u], physical.x, 0.0);
}
"#
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production magnitude probe"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = compute_pipeline(&device, &pipeline_layout, &shader, "probe", "probe");
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range();
        let results: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        let (mut old_error, mut new_error) = (0.0_f64, 0.0_f64);
        let mut checked = 0;
        for (i, (case, result)) in cases.iter().zip(results).enumerate() {
            assert!(result.iter().all(|x| x.is_finite()), "nonfinite case {i}");
            // Early, poorly conditioned fits must remain bounded.
            assert!(result[0] < 2.0, "unbounded startup case {i}: {result:?}");
            assert!(
                (result[3] as f64 - case.input[8] as f64).abs() <= 2e-6 * case.input[8] as f64,
                "input weight precision case {i}"
            );
            if case.check_fit {
                assert!(
                    (result[0] as f64 - case.expected).abs() < 0.002 * case.gain,
                    "fit case {i}: got {}, expected {}",
                    result[0],
                    case.expected
                );
                checked += 1;
            }
            if case.check_tone {
                old_error = old_error.max((result[1] as f64 / (0.5 * case.gain) - 1.0).abs());
                new_error = new_error.max((result[0] as f64 / (0.5 * case.gain) - 1.0).abs());
            }
        }
        eprintln!(
            "{checked} conditioned fits / {} cases; worst startup relative amplitude error: old {old_error:.6}, new {new_error:.6}",
            cases.len()
        );
        assert!(old_error > 0.05);
        assert!(new_error < 0.004);
        drop(mapped);
        readback.unmap();
        let energy_pipeline = compute_pipeline(
            &device,
            &pipeline_layout,
            &shader,
            "energy probe",
            "energy_probe",
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&energy_pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range();
        let results: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
        let mut near_zero_position = 0;
        for (i, result) in results.iter().enumerate() {
            assert!(
                result.iter().all(|x| x.is_finite()),
                "nonfinite position/velocity case {i}"
            );
            assert!(
                (result[0] - result[1]).abs() < 0.001 * result[1].max(0.001),
                "position/velocity case {i}: GPU {} vs independent displacement difference {}",
                result[0],
                result[1]
            );
            if result[0] > 0.01 && result[2].abs() < result[0] * 0.05 {
                near_zero_position += 1;
            }
        }
        assert!(
            near_zero_position > 10,
            "must recover amplitude near displacement zero crossings"
        );
        eprintln!(
            "position/velocity: {} independent reference cases; {near_zero_position} near-zero displacement cases",
            cases.len()
        );
        drop(mapped);
        readback.unmap();
        check_streaming(&device, &queue, &layout);
    });
}

fn check_streaming(device: &wgpu::Device, queue: &wgpu::Queue, layout: &wgpu::BindGroupLayout) {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    let pipeline = build_resonator_pipeline(device, &pipeline_layout);
    let input = storage_buffer(device, "streaming tone", SAMPLE_COUNT as u64);
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let sixteenth = storage_buffer(device, "sixteenth-rate tone", SAMPLE_COUNT as u64);
    let state = storage_buffer(
        device,
        "streaming state",
        MAX_BIN_COUNT as u64 * 16 + MAX_BIN_COUNT as u64 * 16,
    );
    let dummy = storage_buffer(device, "unused streaming binding", 16);
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of::<AnalysisParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bindings = create_compute_channel_bind_group(
        device,
        layout,
        "streaming",
        &input,
        [
            &output, &uniform, &dummy, &dummy, &dummy, &sixteenth, &dummy, &state,
        ],
    );
    for (frequency, rate, q, multirate, phase_offset) in [
        (16.0, 384000.0, 32.0, false, 0.7),
        (50.0, 48000.0, 10.0, false, 0.7),
        (800.0, 3000.0, 32.0, false, 0.7),
        (10000.0, 48000.0, 200.0, false, 0.7),
        (50.0, 48000.0, 10.0, true, 0.0),
        (50.0, 48000.0, 10.0, true, 0.7),
        (50.0, 48000.0, 10.0, true, 1.4),
        (50.0, 48000.0, 10.0, true, 2.1),
        (16.0, 384000.0, 32.0, true, 0.7),
    ] {
        let settings = VisualizerSettings {
            analysis_method: AnalysisMethod::ResonatorBank,
            resonator_multirate: multirate,
            resonator_q_50_hz: q,
            resonator_q_10_khz: q,
            gain: 1.0,
            min_frequency_hz: frequency,
            max_frequency_hz: frequency,
            low_db_floor: -100.0,
            high_db_floor: -100.0,
            resonator_envelope: ResonatorEnvelope::DecayThenPeak,
            ..VisualizerSettings::default()
        };
        let divisor = if multirate { 16 } else { 1 };
        let effective_rate = rate / divisor as f32;
        let mut previous_levels = [0.0_f32; 2];
        let mut elapsed = 0;
        // The final reset also verifies that both bins' age is cleared.
        for (count, reset) in [
            (4096, true),
            (257, false),
            (0, false),
            (8192, false),
            (4096, true),
        ] {
            let count = count / divisor;
            if reset {
                elapsed = 0;
            }
            let mut samples = vec![0.0_f32; SAMPLE_COUNT];
            for i in 0..count {
                samples[SAMPLE_COUNT - count + i] = (0.5
                    * (std::f64::consts::TAU * frequency as f64 / effective_rate as f64
                        * (elapsed + i) as f64
                        + phase_offset)
                        .cos()) as f32;
            }
            elapsed += count;
            let mut params = analysis_params(
                &settings,
                [rate, 0.0, 1.0 / 60.0],
                2,
                reset,
                (
                    [
                        (count * divisor) as u32,
                        0,
                        0,
                        0,
                        if multirate { count as u32 } else { 0 },
                    ],
                    reset,
                ),
                [0.0; 4],
                PhysicalSize::new(320, 180),
            );
            // Let the measurement through without peak retention obscuring it.
            params.resonator_decay = [1e9, 1.0, 0.0, 0.0];
            // Exercise production suffix uploads, including empty updates and
            // resets, against the independent tone-fit reference below.
            upload_audio_samples(queue, &input, &samples, count * divisor);
            if count > 0 {
                queue.write_buffer(
                    &sixteenth,
                    0,
                    bytemuck::cast_slice(&samples[SAMPLE_COUNT - count..]),
                );
            }
            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 32);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = readback.slice(..).get_mapped_range();
            let bins: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            let expected = 0.5
                * -(-std::f64::consts::PI * frequency as f64 / (q as f64 * effective_rate as f64)
                    * elapsed as f64)
                    .exp_m1();
            for (index, bin) in bins.iter().enumerate() {
                if count == 0 {
                    assert!(
                        (bin[0] - previous_levels[index]).abs() < 2e-6,
                        "empty frame changed magnitude: multirate={multirate}, {} vs {}",
                        bin[0],
                        previous_levels[index]
                    );
                }
                previous_levels[index] = bin[0];
                let amplitude = 10.0_f64.powf((bin[0] as f64 * 100.0 - 100.0) / 20.0);
                assert!(
                    (amplitude - expected).abs() < 0.003 * expected,
                    "stream {frequency} Hz at {rate} Hz, {elapsed} samples: {amplitude} vs {expected}"
                );
            }
            drop(mapped);
            readback.unmap();
        }
    }
    // Preserve the responsive frame-end measurement in both envelope modes.
    // Capturing the block maximum made short bursts look too slow in use.
    let frequency = 3500.0_f64;
    let rate = 48000.0_f64;
    let q = 10.0 + 190.0 * ((frequency.sqrt() - 50.0_f64.sqrt()) / (100.0 - 50.0_f64.sqrt()));
    let phase = std::f64::consts::TAU * frequency / rate;
    let damping = (-phase / (2.0 * q)).exp();
    let weight = 1.0 - damping;
    let mut pulse = vec![0.0_f32; SAMPLE_COUNT];
    let mut z = [0.0_f64; 2];
    let mut true_peak = 0.0_f64;
    let image_den = [
        1.0 - damping * (2.0 * phase).cos(),
        -damping * (2.0 * phase).sin(),
    ];
    let den_norm = image_den[0].powi(2) + image_den[1].powi(2);
    let response = [
        weight * image_den[0] / den_norm,
        -weight * image_den[1] / den_norm,
    ];
    let energy_calibration = (1.0 + response[0]).hypot(response[1]);
    let mut energy_history = Vec::new();
    let mut position_history = Vec::new();
    let mut endpoint = 0.0;
    let mut last_position = 0.0;
    let mut previous_position = 0.0;
    for i in 0..800 {
        let sample = if (32..160).contains(&i) {
            0.5 * (phase * (i - 32) as f64).sin()
        } else {
            0.0
        };
        pulse[SAMPLE_COUNT - 800 + i] = sample as f32;
        let angle = phase * (SAMPLE_COUNT + i) as f64;
        z[0] = damping * z[0] + weight * angle.cos() * sample;
        z[1] = damping * z[1] - weight * angle.sin() * sample;
        previous_position = last_position;
        last_position = z[0] * angle.cos() - z[1] * angle.sin();
        position_history.push(last_position);
        let velocity = (last_position * phase.cos() - previous_position) / phase.sin();
        energy_history
            .push(4.0 * (last_position.powi(2) + velocity.powi(2)) / energy_calibration.powi(2));
        let image = [
            response[0] * (2.0 * angle).cos() + response[1] * (2.0 * angle).sin(),
            -response[0] * (2.0 * angle).sin() + response[1] * (2.0 * angle).cos(),
        ];
        let denominator = 1.0 - image[0].powi(2) - image[1].powi(2);
        let re = (z[0] - image[0] * z[0] - image[1] * z[1]) / denominator;
        let im = (z[1] + image[0] * z[1] - image[1] * z[0]) / denominator;
        endpoint = 2.0 * re.hypot(im);
        true_peak = true_peak.max(endpoint);
    }
    assert!(
        true_peak > endpoint * 1.5,
        "burst must expose missed intra-frame peaks"
    );
    let energy_velocity = (last_position * phase.cos() - previous_position) / phase.sin();
    let energy_calibration = (1.0 + response[0]).hypot(response[1]);
    let energy_endpoint = 2.0 * last_position.hypot(energy_velocity) / energy_calibration;
    let delay = rate / (4.0 * frequency);
    let delay_low = delay.floor() as usize;
    let delayed = energy_history[799 - delay_low] * (1.0 - delay.fract())
        + energy_history[798 - delay_low] * delay.fract();
    let paired_endpoint = (0.5 * (energy_history[799] + delayed)).sqrt();
    let delayed_position = position_history[799 - delay_low] * (1.0 - delay.fract())
        + position_history[798 - delay_low] * delay.fract();
    let position_pair_endpoint =
        2.0 * position_history[799].hypot(delayed_position) / energy_calibration;
    for (envelope, measurement) in [
        (
            ResonatorEnvelope::DecayThenPeak,
            ResonatorMeasurement::ImageRejected,
        ),
        (
            ResonatorEnvelope::PeakThenDecay,
            ResonatorMeasurement::ImageRejected,
        ),
        (
            ResonatorEnvelope::DecayThenPeak,
            ResonatorMeasurement::PositionVelocity,
        ),
        (
            ResonatorEnvelope::PeakThenDecay,
            ResonatorMeasurement::PositionVelocity,
        ),
        (
            ResonatorEnvelope::DecayThenPeak,
            ResonatorMeasurement::PositionVelocityPi2,
        ),
        (
            ResonatorEnvelope::PeakThenDecay,
            ResonatorMeasurement::PositionVelocityPi2,
        ),
        (
            ResonatorEnvelope::DecayThenPeak,
            ResonatorMeasurement::PositionPi2,
        ),
        (
            ResonatorEnvelope::PeakThenDecay,
            ResonatorMeasurement::PositionPi2,
        ),
    ] {
        let settings = VisualizerSettings {
            analysis_method: AnalysisMethod::ResonatorBank,
            resonator_multirate: false,
            min_frequency_hz: frequency as f32,
            max_frequency_hz: frequency as f32,
            low_db_floor: -100.0,
            high_db_floor: -100.0,
            gain: 1.0,
            resonator_envelope: envelope,
            resonator_measurement: measurement,
            ..VisualizerSettings::default()
        };
        for step in 0..6 {
            let reset = step == 0;
            let count = if reset { SAMPLE_COUNT } else { 800 };
            let params = analysis_params(
                &settings,
                [rate as f32, 0.0, 1.0 / 60.0],
                2,
                reset,
                ([count as u32, 0, 0, 0, 0], reset),
                [0.0; 4],
                PhysicalSize::new(320, 180),
            );
            let silence = vec![0.0_f32; SAMPLE_COUNT];
            upload_audio_samples(
                queue,
                &input,
                if step == 1 { &pulse } else { &silence },
                count,
            );
            queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 32);
            queue.submit([encoder.finish()]);
            let (tx, rx) = std::sync::mpsc::channel();
            readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let mapped = readback.slice(..).get_mapped_range();
            if !reset {
                let bins: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
                let initial = match measurement {
                    ResonatorMeasurement::PositionVelocity => energy_endpoint,
                    ResonatorMeasurement::PositionVelocityPi2 => paired_endpoint,
                    ResonatorMeasurement::PositionPi2 => position_pair_endpoint,
                    _ => endpoint,
                };
                let release_db = 60.0 / 0.3 * 800.0 / rate * (step - 1) as f64;
                let expected = initial * 10.0_f64.powf(-release_db / 20.0);
                for bin in bins {
                    let amplitude = 10.0_f64.powf((bin[0] as f64 * 100.0 - 100.0) / 20.0);
                    assert!(
                        (amplitude - expected).abs() < 0.005 * expected,
                        "short transient {envelope:?}: {amplitude} vs {expected}"
                    );
                }
            }
            drop(mapped);
            readback.unmap();
        }
    }
    eprintln!(
        "short transient peak {true_peak:.6}, frame-end {endpoint:.6}: preserving the faster frame-end response ({:.2} dB below block peak)",
        20.0 * (true_peak / endpoint).log10()
    );
}
