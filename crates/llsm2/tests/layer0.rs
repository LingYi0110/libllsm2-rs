mod common;
use llsm2::{
    AnalysisOptions, HarmonicAnalysisMethod, PhaseDirection, SynthesisOptions, analyze, synthesize,
};

#[test]
fn yin_test_driver_recovers_known_pitch_and_silence() {
    let fs = 44100.0;
    for pitch in [100.0, 200.0, 350.0] {
        let x: Vec<_> = (0..8820)
            .map(|i| 0.1 * (std::f64::consts::TAU * pitch * i as f64 / fs).sin())
            .collect();
        let f0 = common::yin(&x, fs, 128.0);
        assert!(f0[8..f0.len() - 8].iter().all(|f| (f - pitch).abs() < 1.0));
    }
    assert!(
        common::yin(&vec![0.0; 8820], fs, 128.0)
            .iter()
            .all(|&f| f == 0.0)
    );
}

fn roundtrip(method: HarmonicAnalysisMethod) {
    let (x, fs) = common::speech();
    let f0 = common::yin(&x, fs, 128.0);
    let voiced = f0.iter().filter(|&&f| f > 0.0).count();
    eprintln!("YIN voiced={voiced}/{}", f0.len());
    assert!(voiced > f0.len() / 4 && voiced < f0.len() * 9 / 10);
    let options = AnalysisOptions {
        thop: 128.0 / fs,
        npsd: 128,
        max_nhar: 400,
        max_nhar_env: 5,
        hm_method: method,
        ..AnalysisOptions::default()
    };
    let mut result = analyze(&options, &x, fs, &f0).unwrap();
    assert_eq!(result.aperiodic_residual.len(), x.len());
    assert!(
        result
            .chunk
            .frames
            .iter()
            .all(|f| f.nm.psd_residual.as_ref().unwrap().len() == 128)
    );
    let synthesis = SynthesisOptions::new(fs);
    let ordinary = synthesize(&synthesis, &result.chunk).unwrap();
    let ordinary_metrics = common::metrics(&x, &ordinary.y);
    eprintln!("{method:?} normal: {ordinary_metrics:?}");
    result.chunk.phase_sync_rps(false);
    result.chunk.phase_propagate(PhaseDirection::Forward);
    let synchronized = synthesize(&synthesis, &result.chunk).unwrap();
    let synchronized_metrics = common::metrics(&x, &synchronized.y);
    eprintln!("{method:?} RPS: {synchronized_metrics:?}");
    assert!(
        ordinary_metrics.passes(),
        "{method:?} normal: {ordinary_metrics:?}"
    );
    assert!(
        synchronized_metrics.passes(),
        "{method:?} RPS: {synchronized_metrics:?}"
    );
}

#[test]
fn peak_picking_speech_roundtrip() {
    roundtrip(HarmonicAnalysisMethod::PeakPicking);
}
#[test]
fn chirpz_speech_roundtrip() {
    roundtrip(HarmonicAnalysisMethod::ChirpZ);
}

#[test]
fn fractional_hop_all_unvoiced_is_finite_and_strictly_aperiodic() {
    let (x, fs) = common::speech();
    let f0 = vec![0.0; (x.len() as f64 / 100.5).floor() as usize];
    let options = AnalysisOptions {
        thop: 100.5 / fs,
        ..AnalysisOptions::default()
    };
    let result = analyze(&options, &x, fs, &f0).unwrap();
    assert_eq!(result.f0, f0);
    assert_eq!(result.aperiodic_residual, x);
    let out = synthesize(&SynthesisOptions::new(fs), &result.chunk).unwrap();
    assert_eq!(
        out.y.len(),
        ((f0.len() + 1) as f64 * 100.5).round() as usize
    );
    assert!(out.y_sin.iter().all(|&v| v == 0.0));
    assert!(out.y.iter().chain(&out.y_noise).all(|v| v.is_finite()));
}
