mod common;
use llsm2::{
    AnalysisOptions, Chunk, HarmonicFrame, PhaseDirection, SynthesisMode, SynthesisOptions,
    analyze, synthesize,
};
use llsm2::{
    GlottalFlowParams, PbpEffect, PulseContext, interpolate_frames, synthesize_with_effect,
};

fn rms(x: &[f64]) -> f64 {
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

fn periodic_tone() -> (f64, Chunk) {
    let fs = 44100.0;
    let x: Vec<_> = (0..22050)
        .map(|i| {
            (std::f64::consts::TAU * 200.0 * i as f64 / fs).cos()
                + 0.3 * (std::f64::consts::TAU * 400.0 * i as f64 / fs + 0.2).cos()
        })
        .collect();
    // Pitch is known exactly; a signal with only two partials should not use
    // the three-harmonic IF refinement intended for natural voiced speech.
    let mut chunk = analyze(
        &AnalysisOptions {
            refine_f0: false,
            ..AnalysisOptions::default()
        },
        &x,
        fs,
        &[200.0; 100],
    )
    .unwrap()
    .chunk;
    chunk.to_layer1(2048).unwrap();
    chunk.phase_sync_rps(true);
    chunk.clear_layer0();
    chunk.phase_propagate(PhaseDirection::Forward);
    (fs, chunk)
}

#[test]
fn pitch_shift_moves_fundamental_and_compensates_energy() {
    // A known flat tract isolates pitch and gain scaling from formant changes.
    // A narrow-band analyzed signal need not retain constant energy when its
    // harmonics move to different positions in that tract envelope.
    let fs = 44100.0;
    let mut config = AnalysisOptions::default().to_config(fs).unwrap();
    config.nspec = Some(1025);
    let mut tone = Chunk::new(config, 100);
    for frame in &mut tone.frames {
        frame.f0 = 200.0;
        frame.rd = Some(1.0);
        frame.layer1 = Some(llsm2::Layer1Data {
            vt_magn: vec![0.0; 1025],
            vs_phse: vec![0.0; 100],
        });
    }
    tone.phase_propagate(PhaseDirection::Forward);
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(fs)
    };
    for f in &mut tone.frames {
        f.pbp_enabled = true;
    }
    let before = synthesize(&options, &tone).unwrap();
    tone.phase_propagate(PhaseDirection::Backward);
    tone.clear_layer0();
    for f in &mut tone.frames {
        f.f0 *= 1.5;
        if let Some(l) = &mut f.layer1 {
            for v in &mut l.vt_magn {
                *v -= 20.0 * 1.5_f64.log10();
            }
        }
    }
    tone.phase_propagate(PhaseDirection::Forward);
    let shifted = synthesize(&options, &tone).unwrap();
    let pitch = common::yin(&shifted.y_sin, fs, 128.0);
    let voiced: Vec<_> = pitch[40..pitch.len() - 40]
        .iter()
        .filter(|&&f| f > 0.0)
        .copied()
        .collect();
    let estimated = voiced.iter().sum::<f64>() / voiced.len() as f64;
    let energy = rms(&shifted.y_sin[5000..15000]) / rms(&before.y_sin[5000..15000]);
    eprintln!("1.5x tone pitch={estimated:.3} Hz, RMS ratio={energy:.4}");
    assert!((estimated - 300.0).abs() < 1.0);
    assert!((0.95..1.05).contains(&energy));

    let (_, fs, mut speech) = speech_model();
    speech.to_layer1(2048).unwrap();
    speech.phase_sync_rps(true);
    speech.clear_layer0();
    speech.phase_propagate(PhaseDirection::Forward);
    for (i, f) in speech.frames.iter_mut().enumerate() {
        f.pbp_enabled = (i / 50) % 2 == 1;
    }
    let base = synthesize(&options, &speech).unwrap();
    speech.phase_propagate(PhaseDirection::Backward);
    for f in &mut speech.frames {
        f.f0 = (f.f0 * 1.5).min(speech.config.fnyq);
        if let Some(l) = &mut f.layer1 {
            for v in &mut l.vt_magn {
                *v -= 20.0 * 1.5_f64.log10();
            }
        }
    }
    speech.phase_propagate(PhaseDirection::Forward);
    let shifted = synthesize(
        &SynthesisOptions {
            mode: SynthesisMode::Layer1,
            ..SynthesisOptions::new(fs)
        },
        &speech,
    )
    .unwrap();
    let ratio = rms(&shifted.y) / rms(&base.y);
    eprintln!("1.5x speech RMS ratio={ratio:.4}");
    assert!(shifted.y.iter().all(|v| v.is_finite()));
    assert!((0.5..1.7).contains(&ratio));
}

#[test]
fn double_time_stretch_resamples_residual_and_preserves_smooth_energy() {
    let (_, fs, mut original) = speech_model();
    original.to_layer1(2048).unwrap();
    let base = synthesize(&SynthesisOptions::new(fs), &original).unwrap();
    original.phase_propagate(PhaseDirection::Backward);
    original.clear_layer0();
    let n = original.num_frames();
    let mut frames = Vec::with_capacity(n * 2);
    let mut random = 91u64;
    for i in 0..n * 2 {
        let time = (i as f64 / 2.0).min((n - 1) as f64);
        let left = time as usize;
        let right = (left + 1).min(n - 1);
        let mut frame = interpolate_frames(
            &original.config,
            &original.frames[left],
            &original.frames[right],
            time - left as f64,
        )
        .unwrap();
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let neighbor =
            (left as isize + (random % 5) as isize - 2).clamp(0, (n - 1) as isize) as usize;
        frame.nm.psd_residual = original.frames[neighbor].nm.psd_residual.clone();
        frames.push(frame);
    }
    let mut stretched = Chunk {
        config: original.config.clone(),
        frames,
    };
    stretched.phase_propagate(PhaseDirection::Forward);
    stretched.to_layer0().unwrap();
    let layer0 = synthesize(&SynthesisOptions::new(fs), &stretched).unwrap();
    let layer1 = synthesize(
        &SynthesisOptions {
            mode: SynthesisMode::Layer1,
            ..SynthesisOptions::new(fs)
        },
        &stretched,
    )
    .unwrap();
    assert!(layer1.y.iter().all(|v| v.is_finite()));
    // Double the frame grid; the fixed one-hop OLA tail is included once.
    assert_eq!(
        layer1.y.len(),
        ((2 * n + 1) as f64 * original.config.thop * fs).round() as usize
    );
    let ratio = rms(&layer1.y) / rms(&base.y);
    eprintln!("2x speech RMS ratio={ratio:.4}, samples={}", layer1.y.len());
    assert!((0.6..1.5).contains(&ratio));
    let difference: Vec<_> = layer0.y.iter().zip(&layer1.y).map(|(a, b)| a - b).collect();
    assert!(rms(&difference) < 1e-12);
    // Frame joins must not acquire impulsive jumps relative to neighboring
    // waveform differences; compare distributions, not a single loud sample.
    let hop = 128;
    let boundaries: Vec<_> = (hop..layer1.y.len() - hop)
        .step_by(hop)
        .map(|i| (layer1.y[i] - layer1.y[i - 1]).powi(2))
        .collect();
    let derivative: Vec<_> = layer1.y.windows(2).map(|w| w[1] - w[0]).collect();
    let join_ratio =
        (boundaries.iter().sum::<f64>() / boundaries.len() as f64).sqrt() / rms(&derivative);
    eprintln!("stretch frame-join derivative RMS ratio={join_ratio:.4}");
    assert!(join_ratio < 1.5);
}

struct Modulation {
    calls: usize,
    last_frame: usize,
}
impl PbpEffect for Modulation {
    fn modify(&mut self, g: &mut GlottalFlowParams, ctx: &PulseContext<'_>) -> f64 {
        assert_eq!(ctx.pulse_index, self.calls);
        assert!(ctx.frame_index >= self.last_frame);
        assert!(ctx.frame.is_voiced());
        self.last_frame = ctx.frame_index;
        self.calls += 1;
        let osc = (self.calls as f64 * std::f64::consts::TAU / 7.0).sin();
        g.fa *= 1.0 + 0.05 * osc;
        g.rk *= 1.0 + 0.02 * osc;
        g.ee *= 1.0 + 0.2 * osc;
        0.005 * g.t0 * osc
    }
}

#[test]
fn pulse_effect_trait_and_closure_are_effective_and_repeatable() {
    let (fs, mut chunk) = periodic_tone();
    for f in &mut chunk.frames {
        f.pbp_enabled = true;
    }
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(fs)
    };
    let ordinary = synthesize(&options, &chunk).unwrap();
    let mut calls = 0;
    let mut identity = |_: &mut GlottalFlowParams, _: &PulseContext<'_>| {
        calls += 1;
        0.0
    };
    let neutral = synthesize_with_effect(&options, &chunk, &mut identity).unwrap();
    assert_eq!(ordinary.y, neutral.y);
    assert!(calls > 80);
    let mut effect = Modulation {
        calls: 0,
        last_frame: 0,
    };
    let changed = synthesize_with_effect(&options, &chunk, &mut effect).unwrap();
    assert!(changed.y.iter().all(|v| v.is_finite()));
    assert!(effect.calls > 80);
    assert_eq!(changed.y_noise, ordinary.y_noise);
    let diff: Vec<_> = changed
        .y_sin
        .iter()
        .zip(&ordinary.y_sin)
        .map(|(a, b)| a - b)
        .collect();
    let amount = rms(&diff) / rms(&ordinary.y_sin);
    eprintln!("effect periodic relative RMS difference={amount:.4}");
    assert!(amount > 0.03 && amount < 0.5);
    let repeat = synthesize_with_effect(
        &options,
        &chunk,
        &mut Modulation {
            calls: 0,
            last_frame: 0,
        },
    )
    .unwrap();
    assert_eq!(changed.y, repeat.y);
}

#[test]
fn fractional_pulse_clock_supports_output_rate_changes() {
    let source_fs = 16000.0;
    let mut config = AnalysisOptions {
        thop: 100.5 / source_fs,
        chan_freqs: vec![2000.0, 4000.0, 6000.0],
        ..AnalysisOptions::default()
    }
    .to_config(source_fs)
    .unwrap();
    config.nspec = Some(33);
    let mut chunk = Chunk::new(config, 100);
    for frame in &mut chunk.frames {
        frame.f0 = 200.0;
        frame.rd = Some(1.0);
        frame.pbp_enabled = true;
        frame.layer1 = Some(llsm2::Layer1Data {
            vt_magn: vec![0.0; 33],
            vs_phse: vec![0.0; 20],
        });
    }
    chunk.phase_propagate(PhaseDirection::Forward);
    for output_fs in [16000.0, 22050.0, 44100.0] {
        let out = synthesize(
            &SynthesisOptions {
                mode: SynthesisMode::Layer1,
                ..SynthesisOptions::new(output_fs)
            },
            &chunk,
        )
        .unwrap();
        assert_eq!(
            out.y.len(),
            (101.0 * chunk.config.thop * output_fs).round() as usize
        );
        assert!(out.y.iter().all(|v| v.is_finite()));
        let f0 = common::yin(&out.y_sin, output_fs, 128.0);
        let voiced: Vec<_> = f0[10..f0.len() - 10]
            .iter()
            .filter(|&&f| f > 0.0)
            .copied()
            .collect();
        assert!(!voiced.is_empty());
        let estimate = voiced.iter().sum::<f64>() / voiced.len() as f64;
        eprintln!("fractional PbP output fs={output_fs}, F0={estimate:.3}");
        assert!((estimate - 200.0).abs() < 1.0);
    }
}

fn speech_model() -> (Vec<f64>, f64, Chunk) {
    let (x, fs) = common::speech();
    let f0 = common::yin(&x, fs, 128.0);
    let options = AnalysisOptions {
        thop: 128.0 / fs,
        ..AnalysisOptions::default()
    };
    let chunk = analyze(&options, &x, fs, &f0).unwrap().chunk;
    (x, fs, chunk)
}

#[test]
fn speech_layer1_mixed_pulses_matches_input_and_layer0() {
    let (x, fs, mut chunk) = speech_model();
    let layer0 = synthesize(&SynthesisOptions::new(fs), &chunk).unwrap();
    chunk.to_layer1(2048).unwrap();
    chunk.validate_layer1().unwrap();
    chunk.phase_sync_rps(true);
    chunk.clear_layer0();
    for (i, f) in chunk.frames.iter_mut().enumerate() {
        f.pbp_enabled = (i / 50) % 2 == 1;
    }
    chunk.phase_propagate(PhaseDirection::Forward);
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(fs)
    };
    let output = synthesize(&options, &chunk).unwrap();
    assert!(chunk.frames.iter().all(|f| f.hm.is_none()));
    let input_metrics = common::metrics(&x, &output.y);
    let layer0_metrics = common::metrics(&layer0.y, &output.y);
    eprintln!("Layer1 mixed vs input: {input_metrics:?}");
    eprintln!("Layer1 mixed vs Layer0: {layer0_metrics:?}");
    assert!(input_metrics.passes(), "{input_metrics:?}");
    assert!(layer0_metrics.correlation > 0.95 && layer0_metrics.spectral.iter().all(|&v| v < 0.05));
    // Changing a cached harmonic representation cannot override Layer 1.
    for f in &mut chunk.frames {
        if f.is_voiced() {
            f.hm = Some(HarmonicFrame {
                ampl: vec![1e6],
                phse: vec![],
            });
        }
    }
    assert_eq!(output.y, synthesize(&options, &chunk).unwrap().y);
}
