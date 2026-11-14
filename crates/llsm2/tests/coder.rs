#[allow(dead_code)]
mod common;
use llsm2::{
    AnalysisOptions, Chunk, Coder, Frame, Layer1Data, LlsmError, PhaseDirection, SynthesisOptions,
    analyze, synthesize,
};

fn model() -> Chunk {
    let mut config = AnalysisOptions::default().to_config(44100.0).unwrap();
    config.nspec = Some(65);
    config.max_nhar = 7;
    let mut chunk = Chunk::new(config, 1);
    let frame = &mut chunk.frames[0];
    frame.f0 = 200.0;
    frame.rd = Some(1.0);
    frame.layer1 = Some(Layer1Data {
        vt_magn: vec![-20.0; 65],
        vs_phse: vec![0.0; 7],
    });
    frame.nm.psd.fill(-50.0);
    chunk
}

fn validate(config: &llsm2::ModelConfig, frame: Frame, layer1: bool) {
    let mut chunk = Chunk {
        config: config.clone(),
        frames: vec![frame],
    };
    if layer1 {
        chunk.validate_layer1().unwrap();
        chunk.to_layer0().unwrap();
    }
    chunk.validate_layer0().unwrap();
}

#[test]
fn configuration_requires_layer1_positive_orders_and_nonempty_bands() {
    let original = model();
    for (spec, bap, field) in [
        (0, 5, "order_spec"),
        (65, 5, "order_spec"),
        (usize::MAX, 5, "order_spec"),
        (40, 0, "order_bap"),
        (40, 65, "order_bap"),
        (40, usize::MAX, "order_bap"),
    ] {
        assert!(matches!(Coder::new(&original.config, spec, bap),
            Err(LlsmError::InvalidConfig { field: actual, .. }) if actual == field));
    }
    let mut config = original.config.clone();
    for nspec in [None, Some(32)] {
        config.nspec = nspec;
        assert!(matches!(
            Coder::new(&config, 1, 1),
            Err(LlsmError::InvalidConfig { field: "nspec", .. })
        ));
    }
    config.nspec = Some(65);
    config.chan_freqs.clear();
    config.thop = 0.1;
    for nyquist in [40.0, 50.0] {
        config.fnyq = nyquist;
        assert!(matches!(
            Coder::new(&config, 1, 1),
            Err(LlsmError::InvalidConfig { field: "fnyq", .. })
        ));
    }
    config.fnyq = 51.0;
    assert!(Coder::new(&config, 1, 1).is_ok());
    assert!(Coder::new(&original.config, 64, 64).is_ok());
}

#[test]
fn encode_into_matches_encode_and_non_power_of_two_orders_work() {
    let mut chunk = model();
    for nspec in [65, 67] {
        chunk.config.nspec = Some(nspec);
        chunk.frames[0]
            .layer1
            .as_mut()
            .unwrap()
            .vt_magn
            .resize(nspec, -20.0);
        for order in [1, 40, 60, nspec - 1] {
            let coder = Coder::new(&chunk.config, order, 5).unwrap();
            assert_eq!(coder.feature_dim(), order + 8);
            for voiced in [true, false] {
                let mut frame = chunk.frames[0].clone();
                if !voiced {
                    frame.f0 = 0.0;
                    frame.layer1 = None;
                }
                let encoded = coder.encode(&frame).unwrap();
                let mut dst = vec![f64::NAN; coder.feature_dim()];
                coder.encode_into(&frame, &mut dst).unwrap();
                assert_eq!(dst, encoded);
                assert_eq!(dst[0], f64::from(voiced));
                assert_eq!(dst[1], frame.f0);
                assert_eq!(dst[2], if voiced { 1.0 } else { 0.0 });
                if !voiced {
                    assert!(dst[3 + order..].iter().all(|&v| v == 1.0));
                }
                validate(&chunk.config, coder.decode_layer0(&dst).unwrap(), false);
                validate(&chunk.config, coder.decode_layer1(&dst).unwrap(), true);
            }
        }
    }
}

#[test]
fn coder_rejects_mismatched_buffers_invalid_frames_and_nonfinite_features() {
    let chunk = model();
    let coder = Coder::new(&chunk.config, 40, 5).unwrap();
    let features = coder.encode(&chunk.frames[0]).unwrap();
    for n in [0, features.len() - 1, features.len() + 1] {
        assert!(matches!(
            coder.encode_into(&chunk.frames[0], &mut vec![0.0; n]),
            Err(LlsmError::InvalidInput { field: "dst", .. })
        ));
        assert!(coder.decode_layer0(&vec![0.0; n]).is_err());
        assert!(coder.decode_layer1(&vec![0.0; n]).is_err());
    }
    for i in 0..features.len() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut features = features.clone();
            features[i] = bad;
            for result in [
                coder.decode_layer0(&features),
                coder.decode_layer1(&features),
            ] {
                assert!(matches!(
                    result,
                    Err(LlsmError::InvalidInput {
                        field: "features",
                        ..
                    })
                ));
            }
        }
    }
    for case in 0..4 {
        let mut frame = chunk.frames[0].clone();
        match case {
            0 => frame.rd = None,
            1 => frame.layer1 = None,
            2 => frame
                .layer1
                .as_mut()
                .unwrap()
                .vt_magn
                .pop()
                .map(|_| ())
                .unwrap(),
            _ => frame.nm.psd[0] = f64::NAN,
        }
        assert!(matches!(
            coder.encode(&frame),
            Err(LlsmError::InvalidFrame { index: 0, .. })
        ));
    }
    let mut overflow = features.clone();
    overflow[3..43].fill(1000.0);
    assert!(coder.decode_layer0(&overflow).is_err());
    assert!(coder.decode_layer1(&overflow).is_err());
    let mut overflow = chunk.frames[0].clone();
    overflow.nm.psd.fill(10000.0);
    assert!(coder.encode(&overflow).is_err());
}

#[test]
fn statistical_predictions_are_clamped_and_both_decoders_return_valid_frames() {
    let chunk = model();
    let coder = Coder::new(&chunk.config, 40, 5).unwrap();
    let mut base = vec![0.3; coder.feature_dim()];
    base[..3].copy_from_slice(&[1.0, 200.0, 1.0]);
    base[3..43].fill(-3.0);
    for (index, values) in [
        (0, vec![-1.0, 0.5, 0.500_001, 2.0]),
        (1, vec![-10.0, 0.0, 10.0, 1e6]),
        (2, vec![-1.0, 0.02, 3.0, 10.0]),
    ] {
        for value in values {
            let mut features = base.clone();
            features[index] = value;
            for layer1 in [false, true] {
                let frame = if layer1 {
                    coder.decode_layer1(&features)
                } else {
                    coder.decode_layer0(&features)
                }
                .unwrap();
                assert_eq!(
                    frame.f0,
                    if features[0] > 0.5 {
                        features[1].clamp(20.0, chunk.config.fnyq)
                    } else {
                        0.0
                    }
                );
                assert_eq!(frame.rd, Some(features[2].clamp(0.02, 3.0)));
                if frame.is_voiced() {
                    let count = if layer1 {
                        frame.layer1.as_ref().unwrap().vs_phse.len()
                    } else {
                        frame.hm.as_ref().unwrap().num_harmonics()
                    };
                    assert_eq!(
                        count,
                        chunk
                            .config
                            .max_nhar
                            .min((chunk.config.fnyq / frame.f0).floor() as usize)
                    );
                }
                validate(&chunk.config, frame, layer1);
            }
        }
    }
    for bap in [-0.05, 0.0, 1.0, 1.05] {
        for voiced in [false, true] {
            let mut features = base.clone();
            features[0] = f64::from(voiced);
            features[43..].fill(bap);
            validate(
                &chunk.config,
                coder.decode_layer0(&features).unwrap(),
                false,
            );
            validate(&chunk.config, coder.decode_layer1(&features).unwrap(), true);
        }
    }
    // Underflow and zero harmonic/noise power have finite logarithmic floors.
    base[3..43].fill(-1000.0);
    validate(&chunk.config, coder.decode_layer0(&base).unwrap(), false);
    validate(&chunk.config, coder.decode_layer1(&base).unwrap(), true);
}

#[test]
fn immutable_coder_is_send_sync_and_concurrent_calls_match_serial_calls() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Coder>();
    let mut chunk = model();
    let coder = Coder::new(&chunk.config, 40, 5).unwrap();
    // The constructor owns its configuration; edits by the caller cannot alter it.
    chunk.config.nspec = None;
    let run = || {
        let features = coder.encode(&chunk.frames[0]).unwrap();
        let layer0 = coder.decode_layer0(&features).unwrap();
        let layer1 = coder.decode_layer1(&features).unwrap();
        (
            features,
            layer0.hm.unwrap().ampl,
            layer1.layer1.unwrap().vt_magn,
        )
    };
    let expected = run();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4).map(|_| scope.spawn(run)).collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
}

#[test]
fn speech_coder_roundtrip_passes_original_waveform_distribution_acceptance() {
    let (x, fs) = common::speech();
    let f0 = common::yin(&x, fs, 128.0);
    let mut original = analyze(
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
    original.to_layer1(2048).unwrap();
    let coder = Coder::new(&original.config, 64, 5).unwrap();
    let features: Vec<_> = original
        .frames
        .iter()
        .map(|f| coder.encode(f).unwrap())
        .collect();
    for layer1 in [false, true] {
        let frames = features
            .iter()
            .map(|f| {
                if layer1 {
                    coder.decode_layer1(f)
                } else {
                    coder.decode_layer0(f)
                }
                .unwrap()
            })
            .collect();
        let mut reconstructed = Chunk {
            config: original.config.clone(),
            frames,
        };
        if layer1 {
            reconstructed.validate_layer1().unwrap();
            reconstructed.to_layer0().unwrap();
        }
        reconstructed.validate_layer0().unwrap();
        reconstructed.phase_propagate(PhaseDirection::Forward);
        let output = synthesize(&SynthesisOptions::new(fs), &reconstructed).unwrap();
        assert!(output.y.iter().all(|v| v.is_finite()));
        let metrics = common::metrics(&x, &output.y);
        eprintln!("Coder Layer {}: {metrics:?}", if layer1 { 1 } else { 0 });
        assert!(metrics.waveform.iter().all(|&v| v.is_finite() && v < 0.05));
    }
}
