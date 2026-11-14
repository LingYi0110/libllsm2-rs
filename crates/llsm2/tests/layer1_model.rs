use llsm2::{
    AnalysisOptions, Chunk, GlottalFlowParams, HarmonicFrame, Layer1Data, LlsmError, PulseContext,
    SynthesisMode, SynthesisOptions, analyze, interpolate_frames, synthesize,
    synthesize_with_effect,
};
use std::f64::consts::{LN_10, PI};
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}

fn model() -> Chunk {
    let mut config = AnalysisOptions::default().to_config(44100.0).unwrap();
    config.nspec = Some(33);
    let mut chunk = Chunk::new(config, 4);
    for f in &mut chunk.frames {
        f.f0 = 200.0;
        f.rd = Some(1.0);
        f.layer1 = Some(Layer1Data {
            vt_magn: vec![-20.0; 33],
            vs_phse: vec![0.0; 100],
        });
    }
    chunk
}

#[test]
fn layer1_conversion_handles_unvoiced_and_unidentifiable_rd_and_fft_boundaries() {
    let config = AnalysisOptions::default().to_config(44100.0).unwrap();
    let mut empty = Chunk::new(config.clone(), 0);
    assert!(empty.to_layer1(64).is_err());
    assert!(empty.to_layer0().is_err());
    let mut chunk = Chunk::new(config, 3);
    for n in [0, 2, 62, 63, 65] {
        assert!(matches!(
            chunk.to_layer1(n),
            Err(LlsmError::InvalidConfig { field: "nfft", .. })
        ));
        assert_eq!(chunk.config.nspec, None);
    }
    chunk.to_layer1(66).unwrap();
    assert_eq!(chunk.config.nspec, Some(34));
    chunk.validate_layer1().unwrap();
    for frame in &chunk.frames {
        assert_eq!(frame.rd, Some(1.0));
        assert!(frame.layer1.is_none());
    }
    chunk.to_layer0().unwrap();
    let out = synthesize(
        &SynthesisOptions {
            mode: SynthesisMode::Layer1,
            ..SynthesisOptions::new(44100.0)
        },
        &chunk,
    )
    .unwrap();
    assert!(out.y_sin.iter().all(|&v| v == 0.0));
    for f in &mut chunk.frames {
        f.f0 = 200.0;
        f.hm = Some(HarmonicFrame::new(1));
    }
    chunk.to_layer1(64).unwrap();
    chunk.validate_layer1().unwrap();
    assert!(chunk.frames.iter().all(|f| f.rd == Some(1.0)));
    // Insufficient fitting information amidst voiced neighbors is interpolated.
    let fs = 44100.0;
    let x: Vec<_> = (0..4410)
        .map(|i| (std::f64::consts::TAU * 200.0 * i as f64 / fs).cos())
        .collect();
    let mut chunk = analyze(
        &AnalysisOptions {
            refine_f0: false,
            ..AnalysisOptions::default()
        },
        &x,
        fs,
        &[200.0; 20],
    )
    .unwrap()
    .chunk;
    chunk.frames[10].hm = Some(HarmonicFrame::new(1));
    chunk.to_layer1(64).unwrap();
    close(chunk.frames[10].rd.unwrap(), chunk.frames[9].rd.unwrap());
}

#[test]
fn reconstruction_caps_harmonics_and_reports_invalid_source_tract_data() {
    let mut chunk = model();
    chunk.config.max_nhar = 7;
    chunk.to_layer0().unwrap();
    chunk.validate_layer0().unwrap();
    assert!(
        chunk
            .frames
            .iter()
            .all(|f| f.hm.as_ref().unwrap().num_harmonics() == 7)
    );
    let config = chunk.config.clone();
    let f = &mut chunk.frames[0];
    f.f0 = config.fnyq / 3.1;
    f.to_layer0(&config).unwrap();
    assert_eq!(f.hm.as_ref().unwrap().num_harmonics(), 3);
    f.layer1.as_mut().unwrap().vs_phse.truncate(2);
    f.to_layer0(&config).unwrap();
    assert_eq!(f.hm.as_ref().unwrap().num_harmonics(), 2);
    f.f0 = config.fnyq;
    f.to_layer0(&config).unwrap();
    assert_eq!(f.hm.as_ref().unwrap().num_harmonics(), 1);
    chunk.validate_layer0().unwrap();
    let original = model();
    for case in 0..6 {
        let mut bad = original.clone();
        let frame = &mut bad.frames[1];
        match case {
            0 => frame.rd = None,
            1 => frame.rd = Some(4.0),
            2 => frame
                .layer1
                .as_mut()
                .unwrap()
                .vt_magn
                .pop()
                .map(|_| ())
                .unwrap(),
            3 => frame.layer1.as_mut().unwrap().vs_phse.clear(),
            _ => frame.layer1.as_mut().unwrap().vs_phse[0] = f64::NAN,
        }
        assert!(matches!(
            bad.to_layer0(),
            Err(LlsmError::InvalidFrame { index: 1, .. })
        ));
        assert!(bad.frames.iter().all(|f| f.hm.is_none()));
    }
    let mut huge = model();
    huge.frames[2]
        .layer1
        .as_mut()
        .unwrap()
        .vt_magn
        .fill(10000.0);
    assert!(matches!(
        huge.to_layer0(),
        Err(LlsmError::InvalidFrame { index: 2, .. })
    ));
    assert!(huge.frames.iter().all(|f| f.hm.is_none()));
}

#[test]
fn frame_interpolation_handles_voicing_phase_wrap_missing_harmonics_and_residuals() {
    let model = model();
    let mut a = model.frames[0].clone();
    let mut b = model.frames[1].clone();
    a.layer1.as_mut().unwrap().vs_phse = vec![PI - 0.1, 0.8];
    b.layer1.as_mut().unwrap().vs_phse = vec![-PI + 0.1, 1.0, 1.3];
    b.f0 = 300.0;
    b.rd = Some(2.0);
    a.nm.eenv[0] = HarmonicFrame {
        ampl: vec![1.0, 0.4],
        phse: vec![PI - 0.1, 0.8],
    };
    b.nm.eenv[0] = HarmonicFrame {
        ampl: vec![2.0, 0.6, 0.2],
        phse: vec![-PI + 0.1, 1.0, 1.3],
    };
    b.nm.psd_residual = Some(vec![0.0; 256]);
    let f = interpolate_frames(&model.config, &a, &b, 0.5).unwrap();
    close(f.f0, 250.0);
    close(f.rd.unwrap(), 1.5);
    assert!(f.hm.is_none());
    close(f.layer1.as_ref().unwrap().vs_phse[0].abs(), PI);
    assert_eq!(f.layer1.as_ref().unwrap().vs_phse.len(), 3);
    close(f.nm.eenv[0].ampl[2], 0.1);
    close(f.nm.eenv[0].phse[2], 1.3);
    close(
        f.nm.psd_residual.as_ref().unwrap()[0],
        0.375 * 10.0 / LN_10 / 2.0,
    );
    let mut unvoiced = a.clone();
    unvoiced.f0 = 0.0;
    unvoiced.layer1 = None;
    let mut fade = interpolate_frames(&model.config, &unvoiced, &a, 0.25).unwrap();
    close(fade.f0, 200.0);
    close(
        fade.layer1.as_ref().unwrap().vt_magn[0],
        -20.0 + 20.0 * 0.25_f64.log10(),
    );
    close(fade.nm.eenv[0].ampl[0], 0.25);
    fade.to_layer0(&model.config).unwrap();
    let floor = interpolate_frames(&model.config, &unvoiced, &a, 1e-8).unwrap();
    assert!(floor.layer1.unwrap().vt_magn.iter().all(|&v| v >= -80.0));
    let first = interpolate_frames(&model.config, &unvoiced, &a, 0.0).unwrap();
    assert!(!first.is_voiced());
    assert!(first.layer1.is_none());
    for ratio in [-0.1, 1.1, f64::NAN] {
        assert!(interpolate_frames(&model.config, &a, &b, ratio).is_err());
    }
    let mut conf = model.config.clone();
    conf.nspec = None;
    a.layer1 = None;
    b.layer1 = None;
    a.hm = Some(HarmonicFrame {
        ampl: vec![1.0],
        phse: vec![PI - 0.1],
    });
    b.hm = Some(HarmonicFrame {
        ampl: vec![2.0, 0.2],
        phse: vec![-PI + 0.1, 1.3],
    });
    let f = interpolate_frames(&conf, &a, &b, 0.5).unwrap();
    close(f.hm.as_ref().unwrap().ampl[1], 0.1);
    close(f.hm.unwrap().phse[0].abs(), PI);
}

#[test]
fn layer1_interpolation_requires_rd_on_unvoiced_inputs_including_endpoints() {
    let model = model();
    let voiced = &model.frames[0];
    let mut unvoiced = voiced.clone();
    unvoiced.f0 = 0.0;
    unvoiced.layer1 = None;
    for ratio in [0.0, 0.5, 1.0] {
        for (a, b) in [
            (&unvoiced, voiced),
            (voiced, &unvoiced),
            (&unvoiced, &unvoiced),
        ] {
            let frame = interpolate_frames(&model.config, a, b, ratio).unwrap();
            let mut result = Chunk {
                config: model.config.clone(),
                frames: vec![frame],
            };
            result.validate_layer1().unwrap();
            result.to_layer0().unwrap();
        }
    }
    unvoiced.rd = None;
    for ratio in [0.0, 0.5, 1.0] {
        for (a, b, index) in [
            (&unvoiced, voiced, 0),
            (voiced, &unvoiced, 1),
            (&unvoiced, &unvoiced, 0),
        ] {
            assert!(matches!(
                interpolate_frames(&model.config, a, b, ratio),
                Err(LlsmError::InvalidFrame { index: actual, .. }) if actual == index
            ));
        }
    }
    // Layer 0 unvoiced frames still do not require Rd.
    let mut config = model.config.clone();
    config.nspec = None;
    assert!(interpolate_frames(&config, &unvoiced, &unvoiced, 0.5).is_ok());
}

#[test]
fn glottal_parameters_validate_constructor_inputs_and_period_independent_shapes() {
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            GlottalFlowParams::from_rd(1.0, value, 1.0),
            Err(LlsmError::InvalidInput { field: "t0", .. })
        ));
        assert!(matches!(
            GlottalFlowParams::from_rd(1.0, 0.005, value),
            Err(LlsmError::InvalidInput { field: "ee", .. })
        ));
    }
    GlottalFlowParams::from_rd(1.0, 1e-300, 1.0).unwrap();
    // tp=.4, te=.65, ta=.3 is a valid LF shape at every positive period.
    for t0 in [0.0001, 1.0 / 800.0, 0.005, 0.1, 0.5, 2.0, 10.0] {
        GlottalFlowParams {
            fa: 1.0 / (0.3 * t0),
            rk: 0.625,
            rg: 1.25,
            t0,
            ee: 1.0,
        }
        .validate()
        .unwrap();
    }
}

#[test]
fn pulse_effects_reject_invalid_parameters_and_time_offsets_with_context() {
    let mut chunk = model();
    for f in &mut chunk.frames {
        f.pbp_enabled = true;
    }
    let options = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(44100.0)
    };
    for case in 0..5 {
        let mut effect = move |g: &mut GlottalFlowParams, _: &PulseContext<'_>| {
            match case {
                0 => g.fa = f64::NAN,
                1 => g.rk = -1.0,
                2 => g.rg = 0.01,
                3 => return f64::INFINITY,
                4 => return 2.0 * g.t0,
                _ => g.fa = f64::MAX,
            }
            0.0
        };
        assert!(matches!(
            synthesize_with_effect(&options, &chunk, &mut effect),
            Err(LlsmError::InvalidPulse {
                frame_index: 0,
                pulse_index: 0,
                ..
            })
        ));
    }
    for rd in [0.02, 0.3, 1.0, 2.7, 3.0] {
        GlottalFlowParams::from_rd(rd, 0.005, 1.0)
            .unwrap()
            .validate()
            .unwrap();
    }
    assert!(GlottalFlowParams::from_rd(4.0, 0.005, 1.0).is_err());
    let mut identity = |_: &mut GlottalFlowParams, _: &PulseContext<'_>| 0.0;
    assert!(
        synthesize_with_effect(&SynthesisOptions::new(44100.0), &chunk, &mut identity).is_err()
    );
}
