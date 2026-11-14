use llsm2::MIN_VOICED_F0;
use llsm2::{
    AnalysisOptions, Chunk, HarmonicFrame, Layer1Data, LlsmError, PhaseDirection, SynthesisMode,
    SynthesisOptions, analyze, synthesize,
};
use std::f64::consts::{LN_10, TAU};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-11, "{a} != {b}");
}

#[test]
fn empty_chunk_is_rejected_at_validation_and_synthesis_boundaries() {
    let config = AnalysisOptions::default().to_config(44100.0).unwrap();
    let chunk = Chunk::new(config, 0);
    assert!(matches!(
        chunk.validate_layer0(),
        Err(LlsmError::InvalidInput {
            field: "frames",
            ..
        })
    ));
    assert!(matches!(
        synthesize(&SynthesisOptions::new(44100.0), &chunk),
        Err(LlsmError::InvalidInput {
            field: "frames",
            ..
        })
    ));
}

#[test]
fn edited_subaudio_pitch_is_rejected() {
    let conf = AnalysisOptions::default().to_config(44100.0).unwrap();
    let mut chunk = Chunk::new(conf, 1);
    chunk.frames[0].hm = Some(HarmonicFrame::new(1));
    for f0 in [1e-30, 1e-9, 9.999] {
        chunk.frames[0].f0 = f0;
        assert!(matches!(
            chunk.validate_layer0(),
            Err(LlsmError::InvalidFrame { index: 0, .. })
        ));
        assert!(matches!(
            synthesize(&SynthesisOptions::new(44100.0), &chunk),
            Err(LlsmError::InvalidFrame { index: 0, .. })
        ));
    }
}

#[test]
fn analysis_rejects_subaudio_pitch_before_window_allocation_and_bounds_refinement() {
    let fs = 8000.0;
    let x: Vec<_> = (0..4800)
        .map(|i| 0.1 * (TAU * 9.5 * i as f64 / fs).cos())
        .collect();
    for refine in [false, true] {
        let options = AnalysisOptions {
            chan_freqs: vec![],
            npsd: 32,
            refine_f0: refine,
            ..AnalysisOptions::default()
        };
        let mut f0 = vec![MIN_VOICED_F0; 120];
        for invalid in [1e-30, 1e-9, MIN_VOICED_F0 - 0.001] {
            f0[60] = invalid;
            assert!(matches!(
                analyze(&options, &x, fs, &f0),
                Err(LlsmError::InvalidInput { field: "f0", .. })
            ));
        }
        f0[60] = MIN_VOICED_F0;
        // The actual tone is just below the supported minimum. IF candidates
        // must respect the model domain even when the input contour is valid.
        let result = analyze(&options, &x, fs, &f0).unwrap();
        assert!(
            result
                .f0
                .iter()
                .all(|&f| f >= MIN_VOICED_F0 && f <= fs / 2.0)
        );
        assert!(f0.iter().all(|&f| f == MIN_VOICED_F0));
        result.chunk.validate_layer0().unwrap();
    }
}

fn noise_power_above(samples: &[f64], fs: f64, frequency: f64) -> f64 {
    use rustfft::{FftPlanner, num_complex::Complex64};
    let n = 4096;
    let start = (samples.len() - n) / 2;
    let mut input: Vec<_> = samples[start..start + n]
        .iter()
        .enumerate()
        .map(|(i, &v)| Complex64::new(v * (0.5 - 0.5 * (TAU * i as f64 / n as f64).cos()), 0.0))
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut input);
    let total: f64 = input[..=n / 2].iter().map(|v| v.norm_sqr()).sum();
    input[..=n / 2]
        .iter()
        .enumerate()
        .filter(|(i, _)| *i as f64 * fs / n as f64 > frequency)
        .map(|(_, v)| v.norm_sqr())
        .sum::<f64>()
        / total
}

#[test]
fn cross_rate_noise_respects_the_source_bandwidth() {
    for (source_fs, output_fs) in [(16000.0, 44100.0), (44100.0, 16000.0)] {
        let options = AnalysisOptions {
            chan_freqs: vec![2000.0, 4000.0, 6000.0],
            ..AnalysisOptions::default()
        };
        let mut chunk = Chunk::new(options.to_config(source_fs).unwrap(), 100);
        for frame in &mut chunk.frames {
            frame.nm.psd.fill(-30.0);
            frame.nm.edc.fill(1.0);
        }
        let out = synthesize(&SynthesisOptions::new(output_fs), &chunk).unwrap();
        assert_eq!(
            out.y.len(),
            (101.0 * options.thop * output_fs).round() as usize
        );
        assert!(out.y.iter().all(|v| v.is_finite()));
        assert!(out.y_sin.iter().all(|&v| v == 0.0));
        assert!(out.y_noise.iter().any(|v| v.abs() > 1e-3));
        if output_fs > source_fs {
            // Leave 1 kHz for the finite STFT/filter transition and measure
            // the unsupported band, not only whether samples remain finite.
            let ratio = noise_power_above(&out.y_noise, output_fs, source_fs / 2.0 + 1000.0);
            eprintln!("upsampled noise power above source band={ratio:.8e}");
            assert!(ratio < 1e-4, "unsupported-band power ratio={ratio}");
        }

        // Exercise the same rate change from analyzed data as well as the
        // deliberately flat model above, which makes endpoint extrapolation
        // observable independently of the input recording's spectral shape.
        let mut state = 91u64;
        let input: Vec<_> = (0..(source_fs * 0.5) as usize)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 12) as f64 / (1u64 << 52) as f64 - 0.5) * 0.1
            })
            .collect();
        let analysis = analyze(&options, &input, source_fs, &[0.0; 100]).unwrap();
        let analyzed_out = synthesize(&SynthesisOptions::new(output_fs), &analysis.chunk).unwrap();
        assert_eq!(analyzed_out.y.len(), out.y.len());
        assert!(analyzed_out.y.iter().all(|v| v.is_finite()));
        assert!(analyzed_out.y_sin.iter().all(|&v| v == 0.0));
        if output_fs > source_fs {
            assert!(
                noise_power_above(&analyzed_out.y_noise, output_fs, source_fs / 2.0 + 1000.0)
                    < 1e-4
            );
        }
    }
    // Downsampling must also discard a source harmonic above output Nyquist.
    let conf = AnalysisOptions::default().to_config(44100.0).unwrap();
    let mut chunk = Chunk::new(conf, 4);
    for f in &mut chunk.frames {
        f.f0 = 9000.0;
        f.hm = Some(HarmonicFrame {
            ampl: vec![1.0],
            phse: vec![0.0],
        });
    }
    let out = synthesize(&SynthesisOptions::new(16000.0), &chunk).unwrap();
    assert!(out.y.iter().all(|&v| v == 0.0));
}

#[test]
fn public_analysis_boundary_rejects_invalid_options_and_signals() {
    let options = AnalysisOptions::default();
    let fs = 44100.0;
    let x = vec![0.0; 882];
    let f0 = vec![0.0; 4];
    for fs in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(analyze(&options, &x, fs, &f0).is_err());
    }
    for thop in [0.0, -0.1, f64::NAN, f64::INFINITY, 1.0 / fs] {
        let options = AnalysisOptions {
            thop,
            ..options.clone()
        };
        assert!(analyze(&options, &x, fs, &f0).is_err());
    }
    for boundaries in [
        vec![-1.0],
        vec![2000.0, 1000.0],
        vec![2000.0, 2000.0],
        vec![fs / 2.0],
        vec![f64::NAN],
    ] {
        let options = AnalysisOptions {
            chan_freqs: boundaries,
            ..options.clone()
        };
        assert!(analyze(&options, &x, fs, &f0).is_err());
    }
    for invalid in [-1.0, fs, f64::NAN, f64::INFINITY] {
        assert!(analyze(&options, &x, fs, &[invalid; 4]).is_err());
    }
    for n in [0, 3, 5] {
        assert!(matches!(
            analyze(&options, &x, fs, &vec![0.0; n]),
            Err(LlsmError::InvalidF0Length { .. })
        ));
    }
    assert!(analyze(&options, &[], fs, &[]).is_err());
    let mut invalid = x;
    invalid[10] = f64::NAN;
    assert!(analyze(&options, &invalid, fs, &f0).is_err());
    for npsd in [0, 1] {
        assert!(
            AnalysisOptions {
                npsd,
                ..options.clone()
            }
            .to_config(fs)
            .is_err()
        );
    }
    assert!(
        AnalysisOptions {
            max_nhar: 0,
            ..options.clone()
        }
        .to_config(fs)
        .is_err()
    );
    assert!(
        AnalysisOptions {
            rel_winsize: 0.0,
            ..options.clone()
        }
        .to_config(fs)
        .is_err()
    );
    assert!(
        AnalysisOptions {
            lip_radius: 0.0,
            ..options.clone()
        }
        .to_config(fs)
        .is_err()
    );
}

#[test]
fn synthesis_checks_caller_edited_frame_shapes_and_values() {
    let conf = AnalysisOptions::default().to_config(44100.0).unwrap();
    let original = Chunk::new(conf, 2);
    let options = SynthesisOptions::new(44100.0);
    let mut variants = Vec::new();
    let mut chunk = original.clone();
    chunk.frames[1].f0 = 200.0;
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].f0 = f64::NAN;
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].nm.edc.clear();
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].nm.eenv[0].phse.clear();
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].nm.psd.pop();
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].nm.psd[0] = f64::NAN;
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].nm.edc[0] = -1.0;
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].nm.psd_residual = Some(vec![0.0; 2]);
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].layer1 = Some(Layer1Data {
        vt_magn: vec![0.0; 10],
        vs_phse: vec![],
    });
    variants.push(chunk);
    let mut chunk = original.clone();
    chunk.frames[1].hm = Some(HarmonicFrame {
        ampl: vec![1.0],
        phse: vec![],
    });
    variants.push(chunk);
    for chunk in variants {
        assert!(matches!(
            synthesize(&options, &chunk),
            Err(LlsmError::InvalidFrame { index: 1, .. })
        ));
    }
    assert!(matches!(
        synthesize(
            &SynthesisOptions {
                mode: SynthesisMode::Layer1,
                ..options.clone()
            },
            &original
        ),
        Err(LlsmError::InvalidConfig { field: "nspec", .. })
    ));
    for fs in [0.0, -1.0, f64::NAN, 1.0] {
        assert!(synthesize(&SynthesisOptions::new(fs), &original).is_err());
    }
    assert!(
        synthesize(
            &SynthesisOptions {
                iczt_threshold: Some((f64::NAN, 1.0)),
                ..options
            },
            &original
        )
        .is_err()
    );
}

#[test]
fn frame_defaults_clone_and_phase_operations_preserve_model_semantics() {
    let config = AnalysisOptions::default().to_config(44100.0).unwrap();
    let mut chunk = Chunk::new(config, 3);
    assert_eq!(chunk.num_frames(), 3);
    assert_eq!(chunk.config.num_channels(), 4);
    assert_eq!(chunk.frames[0].nm.edc, [1e-5; 4]);
    assert!(chunk.frames[0].nm.psd.iter().all(|&p| p == -120.0));
    chunk.frames[0].f0 = 100.0;
    chunk.frames[1].f0 = 0.0;
    chunk.frames[2].f0 = 200.0;
    for frame in &mut chunk.frames {
        frame.hm = Some(HarmonicFrame {
            ampl: vec![1.0, 0.5],
            phse: vec![0.3, -0.2],
        });
        frame.nm.eenv[0] = HarmonicFrame {
            ampl: vec![0.1],
            phse: vec![0.7],
        };
    }
    let original = chunk.clone();
    chunk.phase_propagate(PhaseDirection::Backward);
    chunk.phase_propagate(PhaseDirection::Forward);
    for (a, b) in chunk.frames.iter().zip(&original.frames) {
        for (p, q) in
            a.hm.as_ref()
                .unwrap()
                .phse
                .iter()
                .zip(&b.hm.as_ref().unwrap().phse)
        {
            close(*p, *q);
        }
        close(a.nm.eenv[0].phse[0], b.nm.eenv[0].phse[0]);
    }
    chunk.phase_sync_rps(false);
    for frame in &chunk.frames {
        close(frame.hm.as_ref().unwrap().phse[0], 0.0);
        close(frame.hm.as_ref().unwrap().phse[1], -0.8);
        close(frame.nm.eenv[0].phse[0], 0.4);
    }
    chunk.phase_propagate(PhaseDirection::Forward);
    let contour = chunk.f0_contour();
    let mut integral = 0.0;
    for (frame, &f) in chunk.frames.iter().zip(&contour) {
        integral += f * TAU * chunk.config.thop;
        close((frame.hm.as_ref().unwrap().phse[0] - integral).sin(), 0.0);
    }
    chunk.frames[0].nm.psd_residual = Some(vec![1.0; 256]);
    let mut clone = chunk.clone();
    clone.frames[0].nm.psd_residual.as_mut().unwrap()[0] = 2.0;
    assert_eq!(chunk.frames[0].nm.psd_residual.as_ref().unwrap()[0], 1.0);
    clone.clear_layer0();
    assert!(clone.frames.iter().all(|f| f.hm.is_none()));
    let mut frame = chunk.frames[0].clone();
    frame.layer1 = Some(Layer1Data {
        vt_magn: vec![],
        vs_phse: vec![0.5, 0.8],
    });
    frame.phase_sync_rps(true);
    close(frame.layer1.unwrap().vs_phse[0], 0.0);
}

#[test]
fn silence_psd_retains_euler_correction_and_reference_sample_rate() {
    for fs in [22050.0, 44100.0] {
        let options = AnalysisOptions::default();
        let n = (fs * options.thop * 10.0) as usize;
        let result = analyze(&options, &vec![0.0; n], fs, &[0.0; 10]).unwrap();
        let expected =
            10.0 * (1e-10 * std::f64::consts::EULER_GAMMA.exp() * 44100.0 / fs + 1e-12).log10();
        for f in &result.chunk.frames {
            for &v in &f.nm.psd {
                close(v, expected);
            }
            for &v in f.nm.psd_residual.as_ref().unwrap() {
                close(v, 0.0);
            }
        }
    }
}

#[test]
fn noise_residual_bias_gate_and_seed_have_explicit_effects() {
    let conf = AnalysisOptions::default().to_config(44100.0).unwrap();
    let mut chunk = Chunk::new(conf, 12);
    for f in &mut chunk.frames {
        f.nm.psd.fill(-40.0);
    }
    let options = SynthesisOptions::new(44100.0);
    let base = synthesize(&options, &chunk).unwrap();
    assert_eq!(base.y, synthesize(&options, &chunk).unwrap().y);
    let different = synthesize(
        &SynthesisOptions {
            random_seed: 999,
            ..options.clone()
        },
        &chunk,
    )
    .unwrap();
    assert_ne!(base.y, different.y);
    for f in &mut chunk.frames {
        f.nm.psd_residual = Some(vec![0.375 * 10.0 / LN_10; 256]);
    }
    assert_eq!(base.y, synthesize(&options, &chunk).unwrap().y);
    for f in &mut chunk.frames {
        f.nm.psd_residual.as_mut().unwrap().fill(0.0);
    }
    let biased = synthesize(&options, &chunk).unwrap();
    for (a, b) in base.y.iter().zip(&biased.y) {
        close(a * (-0.375_f64 / 2.0).exp(), *b);
    }
    for f in &mut chunk.frames {
        f.nm.psd.fill(-120.0);
    }
    assert!(
        synthesize(&options, &chunk)
            .unwrap()
            .y_noise
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn high_band_envelope_uses_original_signal_and_refinement_is_read_only() {
    let fs = 44100.0;
    let hop = 224.0;
    let x: Vec<_> = (0..224 * 30)
        .map(|i| (TAU * 9000.0 * i as f64 / fs).cos())
        .collect();
    let f0 = vec![300.0; 30];
    let original = f0.clone();
    let options = AnalysisOptions {
        thop: hop / fs,
        max_nhar: 40,
        refine_f0: false,
        ..AnalysisOptions::default()
    };
    let result = analyze(&options, &x, fs, &f0).unwrap();
    assert_eq!(f0, original);
    assert_eq!(result.f0, original);
    assert!(
        result.chunk.frames[5..25]
            .iter()
            .all(|f| f.nm.edc[3] > 0.01)
    );
    let mean_residual_power = result.aperiodic_residual[1120..5600]
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        / 4480.0;
    assert!(
        mean_residual_power < 1e-4,
        "residual power={mean_residual_power}"
    );
    // A detuned known tone exercises actual refinement without changing input.
    let x: Vec<_> = (0..224 * 30)
        .map(|i| (TAU * 203.0 * i as f64 / fs).cos())
        .collect();
    let input = vec![200.0; 30];
    let options = AnalysisOptions {
        refine_f0: true,
        ..options
    };
    let result = analyze(&options, &x, fs, &input).unwrap();
    assert_eq!(input, [200.0; 30]);
    assert!(result.f0[5..25].iter().all(|f| (f - 203.0).abs() < 0.2));
}

#[test]
fn low_pitch_long_windows_and_near_nyquist_frames_are_finite() {
    for (fs, pitch) in [(44100.0, 20.0), (8000.0, 3900.0), (8000.0, 4000.0)] {
        let x: Vec<_> = (0..(fs * 0.1) as usize)
            .map(|i| 0.1 * (TAU * pitch * i as f64 / fs).cos())
            .collect();
        let f0 = vec![pitch; 20];
        for refine in [false, true] {
            let options = AnalysisOptions {
                chan_freqs: vec![],
                npsd: 32,
                max_nhar_env: 0,
                refine_f0: refine,
                ..AnalysisOptions::default()
            };
            let result = analyze(&options, &x, fs, &f0).unwrap();
            result.chunk.validate_layer0().unwrap();
            let output = synthesize(&SynthesisOptions::new(fs), &result.chunk).unwrap();
            assert!(output.y.iter().all(|v| v.is_finite()));
        }
    }
}
