#[allow(dead_code)]
mod common;
use llsm2::{
    AnalysisOptions, Chunk, Frame, GlottalFlowParams, HarmonicFrame, Layer1Data, LlsmError,
    ModelConfig, NoiseFrame, PbpEffect, PhaseDirection, PulseContext, RtSynthesizer, SynthesisMode,
    SynthesisOptions, analyze, synthesize,
};
use std::f64::consts::TAU;

fn model(fs: f64, hop: f64, f0: f64, count: usize, layer1: bool) -> Chunk {
    let mut config = AnalysisOptions {
        thop: hop / fs,
        chan_freqs: vec![],
        ..AnalysisOptions::default()
    }
    .to_config(fs)
    .unwrap();
    if layer1 {
        config.nspec = Some(129);
    }
    let mut chunk = Chunk::new(config, count);
    for (i, f) in chunk.frames.iter_mut().enumerate() {
        f.f0 = f0;
        f.rd = Some(1.0);
        f.pbp_enabled = layer1;
        f.nm.edc.fill(0.01);
        f.nm.psd.fill(-35.0);
        f.hm = Some(HarmonicFrame {
            ampl: vec![0.1, 0.03],
            phse: vec![
                i as f64 * TAU * f0 * hop / fs,
                2.0 * i as f64 * TAU * f0 * hop / fs,
            ],
        });
        if layer1 {
            f.hm = None;
            f.layer1 = Some(Layer1Data {
                vt_magn: vec![-20.0; 129],
                vs_phse: vec![0.0; 20],
            });
        }
    }
    if layer1 {
        chunk.phase_propagate(PhaseDirection::Forward);
    }
    chunk
}
fn silent(config: &ModelConfig) -> Frame {
    let mut nm = NoiseFrame::new(config.num_channels(), config.max_nhar_env, config.npsd);
    nm.edc.fill(0.0);
    Frame {
        f0: 0.0,
        rd: Some(1.0),
        hm: None,
        layer1: None,
        nm,
        pbp_enabled: false,
    }
}
fn render(
    rt: &mut RtSynthesizer,
    chunk: &Chunk,
    fs: f64,
    mut effect: Option<&mut dyn PbpEffect>,
) -> (Vec<f64>, Vec<f64>) {
    let hop = chunk.config.thop * fs;
    let wanted = ((chunk.num_frames() + 1) as f64 * hop).round() as usize;
    let padding = ((rt.latency() as f64 + 3.0 * hop) / hop).ceil() as usize + 3;
    let silence = silent(&chunk.config);
    let mut p = Vec::new();
    let mut ap = Vec::new();
    for f in chunk
        .frames
        .iter()
        .chain(std::iter::repeat_n(&silence, padding))
    {
        match effect.as_mut() {
            Some(e) => rt.feed_with_effect(f, *e).unwrap(),
            None => rt.feed(f).unwrap(),
        }
        loop {
            let mut a = [0.0; 127];
            let mut b = [0.0; 137];
            let n = rt.read_decomposed(&mut a, &mut b);
            if n == 0 {
                break;
            }
            p.extend_from_slice(&a[..n]);
            ap.extend_from_slice(&b[..n]);
        }
    }
    assert!(
        p[..rt.latency()]
            .iter()
            .chain(&ap[..rt.latency()])
            .all(|&v| v == 0.0)
    );
    let range = rt.latency()..rt.latency() + wanted;
    (p[range.clone()].to_vec(), ap[range].to_vec())
}
fn rms(x: &[f64]) -> f64 {
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

#[test]
fn full_output_is_atomic_and_reads_handle_unequal_and_empty_slices() {
    fn is_send<T: Send>() {}
    is_send::<RtSynthesizer>();
    let fs = 8000.0;
    let chunk = model(fs, 100.5, 200.0, 3, true);
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(fs)
    };
    assert!(RtSynthesizer::new(options.clone(), chunk.config.clone(), 101).is_err());
    let mut rt = RtSynthesizer::new(options.clone(), chunk.config.clone(), 102).unwrap();
    let mut reference = RtSynthesizer::new(options, chunk.config.clone(), 102).unwrap();
    rt.feed(&chunk.frames[0]).unwrap();
    reference.feed(&chunk.frames[0]).unwrap();
    let before = rt.available_samples();
    let mut called = false;
    let mut effect = |_: &mut GlottalFlowParams, _: &PulseContext<'_>| {
        called = true;
        0.0
    };
    assert!(matches!(
        rt.feed_with_effect(&chunk.frames[1], &mut effect),
        Err(LlsmError::BufferFull {
            required: 100,
            available: 1
        })
    ));
    assert!(!called);
    assert_eq!(rt.available_samples(), before);
    assert_eq!(rt.read(&mut []), 0);
    assert_eq!(rt.read_decomposed(&mut [], &mut [0.0; 3]), 0);
    let mut a = [123.0; 7];
    let mut b = [456.0; 3];
    assert_eq!(rt.read_decomposed(&mut a, &mut b), 3);
    assert_eq!(&a[3..], &[123.0; 4]);
    assert_eq!(reference.read(&mut [0.0; 3]), 3);
    let mut tail = [0.0; 200];
    let mut expected = [0.0; 200];
    assert_eq!(rt.read(&mut tail), before - 3);
    reference.read(&mut expected);
    assert_eq!(tail, expected);
    // Continue past the delay so any hidden state advance on rejection would
    // become audible, while the reference has only accepted successful feeds.
    for frame in chunk.frames.iter().cycle().take(100) {
        rt.feed(frame).unwrap();
        reference.feed(frame).unwrap();
        let mut out = [0.0; 128];
        let mut expected = [0.0; 128];
        assert_eq!(rt.read(&mut out), reference.read(&mut expected));
        assert_eq!(out, expected);
    }
}

#[test]
fn fractional_steps_and_mixed_reads_preserve_the_exact_sample_clock() {
    let fs = 16000.0;
    let chunk = model(fs, 100.5, 200.0, 200, false);
    let options = SynthesisOptions::new(fs);
    let mut a = RtSynthesizer::new(options.clone(), chunk.config.clone(), 256).unwrap();
    let mut b = RtSynthesizer::new(options, chunk.config.clone(), 256).unwrap();
    let mut total = 0;
    for frame in &chunk.frames {
        a.feed(frame).unwrap();
        b.feed(frame).unwrap();
        let mut p = [0.0; 128];
        let mut ap = [0.0; 128];
        let mut mixed = [0.0; 128];
        let n = a.read_decomposed(&mut p, &mut ap);
        assert_eq!(b.read(&mut mixed), n);
        for j in 0..n {
            assert_eq!(mixed[j], p[j] + ap[j]);
        }
        total += n;
    }
    assert_eq!(total, 20100);
}

#[test]
fn clear_resets_pending_noise_modulation_pulses_and_effect_contexts() {
    let fs = 16000.0;
    let chunk = model(fs, 80.5, 200.0, 180, true);
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(fs)
    };
    let mut used = RtSynthesizer::new(options.clone(), chunk.config.clone(), 512).unwrap();
    let mut fresh = RtSynthesizer::new(options, chunk.config.clone(), 512).unwrap();
    render(&mut used, &chunk, fs, None);
    used.clear();
    assert_eq!(used.available_samples(), 0);
    let mut contexts = Vec::new();
    let mut identity = |_: &mut GlottalFlowParams, ctx: &PulseContext<'_>| {
        contexts.push((ctx.frame_index, ctx.pulse_index));
        0.0
    };
    let after = render(&mut used, &chunk, fs, Some(&mut identity));
    let expected = render(&mut fresh, &chunk, fs, None);
    assert_eq!(after, expected);
    assert_eq!(contexts[0], (0, 0));
    assert!(
        contexts
            .windows(2)
            .all(|p| p[1].0 >= p[0].0 && p[1].1 == p[0].1 + 1)
    );
    assert!(chunk.frames.iter().all(|f| f.hm.is_none()));
}

#[test]
fn low_rates_large_hops_and_ten_hertz_pulses_have_sufficient_internal_history() {
    for (fs, hop) in [(1000.0, 100.5), (8000.0, 2000.5)] {
        let chunk = model(fs, hop, 10.0, 20, true);
        let options = SynthesisOptions {
            mode: SynthesisMode::Layer1,
            ..SynthesisOptions::new(fs)
        };
        let mut rt =
            RtSynthesizer::new(options, chunk.config.clone(), hop.ceil() as usize + 2).unwrap();
        for offset in [-1.0, 0.0, 1.0] {
            rt.clear();
            let mut effect = |g: &mut GlottalFlowParams, _: &PulseContext<'_>| offset * g.t0;
            let (p, ap) = render(&mut rt, &chunk, fs, Some(&mut effect));
            assert!(p.iter().chain(&ap).all(|v| v.is_finite()));
            assert!(rms(&p) > 1e-5);
        }
    }
}

#[test]
fn real_speech_mixed_pulses_matches_offline_distribution_and_spectrum() {
    let (x, fs) = common::speech();
    let f0 = common::yin(&x, fs, 128.0);
    let mut chunk = analyze(
        &AnalysisOptions {
            thop: 128.0 / fs,
            ..AnalysisOptions::default()
        },
        &x,
        fs,
        &f0,
    )
    .unwrap()
    .chunk;
    chunk.to_layer1(2048).unwrap();
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
    let offline = synthesize(&options, &chunk).unwrap();
    let mut rt = RtSynthesizer::new(options, chunk.config.clone(), 4096).unwrap();
    let (p, ap) = render(&mut rt, &chunk, fs, None);
    let y: Vec<_> = p.iter().zip(&ap).map(|(p, ap)| p + ap).collect();
    let metrics = common::metrics(&offline.y, &y);
    eprintln!("stream/offline: {metrics:?}");
    eprintln!(
        "periodic error={}, aperiodic error={}",
        rms(&p
            .iter()
            .zip(&offline.y_sin)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>()),
        rms(&ap
            .iter()
            .zip(&offline.y_noise)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>())
    );
    assert!(metrics.passes());
}

#[test]
fn growl_effect_on_second_speech_is_finite_localized_and_reproducible() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/audio/are-you-ready.wav");
    let mut wav = hound::WavReader::open(path).unwrap();
    let fs = wav.spec().sample_rate as f64;
    let x: Vec<_> = wav
        .samples::<i16>()
        .map(|v| v.unwrap() as f64 / 32768.0)
        .collect();
    let f0 = common::yin(&x, fs, 128.0);
    let mut chunk = analyze(
        &AnalysisOptions {
            thop: 128.0 / fs,
            ..AnalysisOptions::default()
        },
        &x,
        fs,
        &f0,
    )
    .unwrap()
    .chunk;
    chunk.to_layer1(2048).unwrap();
    chunk.phase_sync_rps(true);
    chunk.clear_layer0();
    for f in &mut chunk.frames {
        f.pbp_enabled = true;
    }
    chunk.phase_propagate(PhaseDirection::Forward);
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(fs)
    };
    let mut rt = RtSynthesizer::new(options, chunk.config.clone(), 4096).unwrap();
    let baseline = render(&mut rt, &chunk, fs, None);
    rt.clear();
    struct Growl {
        hop: f64,
        fs: f64,
        calls: usize,
    }
    impl PbpEffect for Growl {
        fn modify(&mut self, g: &mut GlottalFlowParams, ctx: &PulseContext<'_>) -> f64 {
            self.calls += 1;
            let time = ctx.frame_index as f64 * self.hop / self.fs;
            let envelope =
                ((time - 1.0) / 0.2).clamp(0.0, 1.0) * ((4.4 - time) / 0.2).clamp(0.0, 1.0);
            let lfo = (TAU * ctx.pulse_index as f64 / 6.0
                + (TAU * ctx.pulse_index as f64 / 50.0).sin())
            .sin();
            g.fa *= 1.0 + 0.16 * lfo * envelope;
            g.rk *= 1.0 + 0.04 * lfo * envelope;
            g.ee *= 1.0 - 0.18 * lfo * envelope;
            0.012 * g.t0 * (ctx.pulse_index as f64 * 2.39996).sin() * envelope
        }
    }
    let mut growl = Growl {
        hop: 128.0,
        fs,
        calls: 0,
    };
    let effected = render(&mut rt, &chunk, fs, Some(&mut growl));
    assert!(growl.calls > 100);
    assert!(effected.0.iter().chain(&effected.1).all(|v| v.is_finite()));
    assert_eq!(baseline.1, effected.1);
    let diff: Vec<_> = effected
        .0
        .iter()
        .zip(&baseline.0)
        .map(|(a, b)| a - b)
        .collect();
    let begin = (1.4 * fs) as usize;
    let end = (4.0 * fs) as usize;
    let relative = rms(&diff[begin..end]) / rms(&baseline.0[begin..end]);
    eprintln!("growl relative RMS difference={relative:.4}");
    assert!(relative > 0.02);
    // Leave a pulse/FFT settling margin around the effect envelope.
    assert!(rms(&diff[..(0.8 * fs) as usize]) < 1e-8);
    assert!(rms(&diff[(4.7 * fs) as usize..]) < 1e-8);
    let peak = effected.0.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let base_peak = baseline.0.iter().map(|v| v.abs()).fold(0.0, f64::max);
    assert!(peak < 2.0 * base_peak);
    let max_jump = effected
        .0
        .windows(2)
        .map(|x| (x[1] - x[0]).abs())
        .fold(0.0, f64::max);
    let base_jump = baseline
        .0
        .windows(2)
        .map(|x| (x[1] - x[0]).abs())
        .fold(0.0, f64::max);
    assert!(max_jump < 2.0 * base_jump);
    rt.clear();
    let mut growl = Growl {
        hop: 128.0,
        fs,
        calls: 0,
    };
    assert_eq!(effected, render(&mut rt, &chunk, fs, Some(&mut growl)));
}

fn high_band_fraction(x: &[f64], fs: f64, cutoff: f64) -> f64 {
    use rustfft::{FftPlanner, num_complex::Complex64};
    let n = 1usize << (usize::BITS - 1 - (x.len() / 2).leading_zeros());
    let start = (x.len() - n) / 2;
    let mut spectrum: Vec<_> = (0..n)
        .map(|i| {
            Complex64::new(
                x[start + i] * (0.5 - 0.5 * (TAU * i as f64 / (n - 1) as f64).cos()),
                0.0,
            )
        })
        .collect();
    FftPlanner::new().plan_fft_forward(n).process(&mut spectrum);
    let total: f64 = spectrum[..=n / 2].iter().map(|v| v.norm_sqr()).sum();
    spectrum[..=n / 2]
        .iter()
        .enumerate()
        .filter(|(i, _)| *i as f64 * fs / n as f64 > cutoff)
        .map(|(_, v)| v.norm_sqr())
        .sum::<f64>()
        / total
}

#[test]
fn cross_rate_streaming_respects_source_bandwidth_and_all_unvoiced_has_zero_periodic_output() {
    for (source_fs, fs) in [(8000.0, 44100.0), (44100.0, 8000.0)] {
        for layer1 in [false, true] {
            let chunk = model(source_fs, 128.0, 200.0, 180, layer1);
            let options = SynthesisOptions {
                mode: if layer1 {
                    SynthesisMode::Layer1
                } else {
                    SynthesisMode::Layer0
                },
                ..SynthesisOptions::new(fs)
            };
            let offline = synthesize(&options, &chunk).unwrap();
            let mut rt = RtSynthesizer::new(options, chunk.config.clone(), 4096).unwrap();
            let (p, ap) = render(&mut rt, &chunk, fs, None);
            assert!(p.iter().chain(&ap).all(|v| v.is_finite()));
            assert!(
                p.iter()
                    .zip(&offline.y_sin)
                    .all(|(a, b)| (a - b).abs() < 1e-12)
            );
            if source_fs < fs {
                let fraction = high_band_fraction(&ap, fs, source_fs / 2.0 + 1000.0);
                eprintln!("stream upsampled noise out of band={fraction:.8e}");
                assert!(fraction < 1e-4);
            }
        }
    }
    let mut chunk = model(16000.0, 100.5, 200.0, 150, true);
    for f in &mut chunk.frames {
        f.f0 = 0.0;
        f.layer1 = None;
    }
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(16000.0)
    };
    let mut rt = RtSynthesizer::new(options, chunk.config.clone(), 512).unwrap();
    let (p, ap) = render(&mut rt, &chunk, 16000.0, None);
    assert!(p.iter().all(|&v| v == 0.0));
    assert!(ap.iter().all(|v| v.is_finite()));
}

#[test]
fn rejected_frames_and_effects_do_not_advance_pulse_or_sample_state() {
    let chunk = model(16000.0, 80.0, 200.0, 150, true);
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(16000.0)
    };
    let mut rt = RtSynthesizer::new(options.clone(), chunk.config.clone(), 512).unwrap();
    let mut bad = chunk.frames[0].clone();
    bad.layer1 = None;
    assert!(matches!(
        rt.feed(&bad),
        Err(LlsmError::InvalidFrame { index: 0, .. })
    ));
    let mut bad = chunk.frames[0].clone();
    bad.nm.psd.fill(10000.0);
    assert!(matches!(
        rt.feed(&bad),
        Err(LlsmError::InvalidFrame { index: 0, .. })
    ));
    let mut invalid = |g: &mut GlottalFlowParams, ctx: &PulseContext<'_>| {
        assert_eq!((ctx.frame_index, ctx.pulse_index), (0, 0));
        g.fa = f64::NAN;
        0.0
    };
    assert!(matches!(
        rt.feed_with_effect(&chunk.frames[0], &mut invalid),
        Err(LlsmError::InvalidPulse {
            frame_index: 0,
            pulse_index: 0,
            ..
        })
    ));
    let mut late_invalid = |_: &mut GlottalFlowParams, ctx: &PulseContext<'_>| {
        if ctx.pulse_index == 1 {
            f64::INFINITY
        } else {
            0.0
        }
    };
    assert!(
        rt.feed_with_effect(&chunk.frames[0], &mut late_invalid)
            .is_err()
    );
    assert_eq!(rt.available_samples(), 0);
    let mut fresh = RtSynthesizer::new(options, chunk.config.clone(), 512).unwrap();
    assert_eq!(
        render(&mut rt, &chunk, 16000.0, None),
        render(&mut fresh, &chunk, 16000.0, None)
    );
    let mut layer0 =
        RtSynthesizer::new(SynthesisOptions::new(16000.0), chunk.config.clone(), 512).unwrap();
    assert!(
        layer0
            .feed_with_effect(&chunk.frames[0], &mut invalid)
            .is_err()
    );
}
