use super::*;
use fft::{Complex, Transforms};
use noise::{RandomSource, SeededRng};
use spectral::{HarmonicAnalysis, HarmonicMethod};
use std::f64::consts::{PI, TAU};
use window::{Window, window};
mod verify;

#[test]
fn sparse_harmonic_envelopes_are_finite() {
    let mut t = Transforms::default();
    for amplitudes in [&[][..], &[1.0][..], &[1.0, 0.5][..]] {
        let envelope = spectral::harmonic_envelope(&mut t, amplitudes, 0.4, 32);
        assert_eq!(envelope.len(), 17);
        assert!(envelope.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn instantaneous_frequency_short_input_is_undefined() {
    let detector = f0::IFDetector::new(0.01, 0.01);
    assert_eq!(detector.estimate(&[]), 0.0);
    assert_eq!(detector.estimate(&vec![1.0; detector.len() - 1]), 0.0);
}

#[test]
fn lf_fit_reports_missing_information() {
    let grid = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5];
    let cache = lfmodel::CachedGlottalModel::new(&grid, 4);
    for amplitudes in [
        &[][..],
        &[1.0][..],
        &[0.0; 4][..],
        &[0.0, 1.0, 0.5, 0.3][..],
        &[1.0, 0.0, 0.5, 0.3][..],
    ] {
        assert_eq!(cache.fit(amplitudes), None);
    }
    // A nonzero multi-harmonic spectrum has enough information for a fit;
    // parameter accuracy is covered by the 500-spectrum acceptance test.
    let frequencies = [200.0, 400.0, 600.0, 800.0];
    let amplitudes: Vec<_> = lfmodel::LfModel::from_rd(1.0, 0.005, 1.0)
        .spectrum(&frequencies)
        .iter()
        .enumerate()
        .map(|(i, v)| v.norm() / (i + 1) as f64)
        .collect();
    assert!(cache.fit(&amplitudes).is_some());
}

#[test]
fn lipfilter_magnitude_only_matches_physical_harmonic_grid() {
    let radius = 1.5;
    let f0 = 200.0;
    let original = [1.0, 0.5, 0.25];
    let mut amplitudes = original;
    spectral::lipfilter(radius, f0, &mut amplitudes, None, false);
    let r = 128.0 / (9.0 * PI * PI);
    let l = 8.0 * radius / 100.0 / (3.0 * PI * 340.0);
    for (i, &a) in amplitudes.iter().enumerate() {
        let omega_l = TAU * f0 * (i + 1) as f64 * l;
        let gain = omega_l * r / (r * r + omega_l * omega_l).sqrt();
        close(a, original[i] * gain, 1e-14);
    }
    spectral::lipfilter(radius, f0, &mut amplitudes, None, true);
    for (a, b) in amplitudes.into_iter().zip(original) {
        close(a, b, 1e-14);
    }
    // Radiation annihilates DC; it has no finite inverse there. The harmonic
    // interfaces above therefore exclude DC instead of inventing a gain for it.
    assert_eq!(spectral::lip_response(radius, 0.0), Complex::default());
}

#[test]
fn peak_picking_matches_local_search_and_interpolates_across_pi() {
    let monotonic = spectral::Spectrum {
        amplitude: (0..9).map(|i| (i as f64).exp()).collect(),
        phase: vec![0.0; 9],
    };
    let result = spectral::harmonic_peak_picking(&monotonic, 16.0, 4.0, 1);
    // No local peak in bins 3..=5: cig_find_peak retains the lower bin.
    close(result.amplitude[0], 3.0_f64.exp(), 1e-7);

    let mut phase = vec![0.0; 9];
    phase[4] = PI - 0.1;
    phase[5] = -PI + 0.1;
    let spectrum = spectral::Spectrum {
        amplitude: (0..9)
            .map(|i| (-((i as f64) - 4.3).powi(2)).exp())
            .collect(),
        phase,
    };
    let result = spectral::harmonic_peak_picking(&spectrum, 16.0, 4.0, 1);
    close(result.phase[0], PI - 0.04, 1e-8);

    // Smallest supported FFT (4 samples), with QIFFT's vertex near Nyquist.
    let spectrum = spectral::Spectrum {
        amplitude: (0..3)
            .map(|i| (-((i as f64) - 1.75).powi(2)).exp())
            .collect(),
        phase: vec![0.0; 3],
    };
    let result = spectral::harmonic_peak_picking(&spectrum, 4.0, 1.0, 1);
    assert_eq!(result.amplitude.len(), 1);
    assert!(result.amplitude[0].is_finite() && result.phase[0].is_finite());
}

#[test]
fn stft_workspace_matches_direct_transform_with_changing_windows() {
    let x: Vec<_> = (0..160)
        .map(|i| (0.17 * i as f64).cos() + 0.4 * (0.71 * i as f64).sin())
        .collect();
    let centers = [0, 32, 64, 0, 100, 159];
    let sizes = [32, 8, 8, 17, 64, 4];
    let nfft = 64;
    for kind in [Window::Blackman, Window::Hann] {
        let spectra =
            spectral::spectrogram(&mut Transforms::default(), &x, &centers, &sizes, nfft, kind);
        for ((s, &center), &size) in spectra.iter().zip(&centers).zip(&sizes) {
            let frame = window::fetch_frame(&x, center, size);
            let w = window(kind, size);
            let scale = 2.0 / w.iter().sum::<f64>();
            for k in 0..=nfft / 2 {
                let expected: Complex = frame
                    .iter()
                    .zip(&w)
                    .enumerate()
                    .map(|(i, (&v, &w))| {
                        Complex::from_polar(
                            v * w * scale,
                            -TAU * k as f64 * (i as f64 - (size / 2) as f64) / nfft as f64,
                        )
                    })
                    .sum();
                let actual = Complex::from_polar(s.amplitude[k], s.phase[k]);
                assert!((actual - expected).norm() < 1e-12);
            }
        }
    }
}

#[test]
fn agrees_with_checked_in_c_reference_samples() {
    use std::collections::BTreeMap;
    let expected: BTreeMap<_, Vec<f64>> = include_str!("data/reference-dsp.txt")
        .lines()
        .map(|line| {
            let mut parts = line.split_whitespace();
            (
                parts.next().unwrap(),
                parts.map(|s| s.parse().unwrap()).collect(),
            )
        })
        .collect();
    let compare = |name: &str, values: &[f64], stride: usize, tolerance: f64| {
        let actual: Vec<_> = values.iter().step_by(stride).copied().collect();
        assert_eq!(actual.len(), expected[name].len());
        assert!(
            actual.iter().all(|v| v.is_finite()),
            "{name}: nonfinite output"
        );
        let maximum = actual
            .iter()
            .zip(&expected[name])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!("C reference {name}: max error={maximum:.8e}");
        assert!(
            maximum < tolerance,
            "{name}: error {maximum}, tolerance {tolerance}"
        );
    };
    let x: Vec<_> = (0..1024)
        .map(|i| {
            (0.13 * i as f64).cos()
                + 0.3 * (0.71 * i as f64).sin()
                + if i == 400 { 2.0 } else { 0.0 }
        })
        .collect();
    for (name, cutoff, kind) in [
        ("lowpass", 0.12, filter::FilterKind::Lowpass),
        ("highpass", 0.08, filter::FilterKind::Highpass),
    ] {
        let (b, a) = filter::chebyshev(cutoff, kind);
        compare(name, &filter::filtfilt(b, a, &x), 31, 1e-10);
    }
    compare(
        "moving_average",
        &filter::moving_average(&x, 3.0),
        31,
        1e-12,
    );
    compare(
        "moving_average_fractional",
        &filter::moving_average(&x, 3.5),
        31,
        1e-12,
    );
    compare("smoothing", &filter::smoothing_filter(&x, 9), 31, 1e-12);
    let q: Vec<_> = (0..1024)
        .map(|i| 0.02 + 0.005 * (0.02 * i as f64).sin())
        .collect();
    let r: Vec<_> = (0..1024)
        .map(|i| 0.3 + 0.1 * (0.03 * i as f64).cos())
        .collect();
    let k = filter::kalman_filter(&x, &q, &r, x[0]);
    compare("kalman", &k.values, 31, 1e-12);
    compare(
        "rts",
        &filter::rts_smooth(&k.values, &k.variances, &q),
        31,
        1e-12,
    );
    let mut t = Transforms::default();
    compare("psd", &spectral::estimate_psd(&mut t, &x, 2048), 31, 1e-10);
    let amplitude: Vec<_> = (1..=20).map(|i| (-0.1 * i as f64).exp()).collect();
    compare(
        "harmonic_spectrum",
        &spectral::harmonic_spectrum(&amplitude, 0.01, 1024),
        7,
        0.01,
    );
    // C uses approximate float log/trigonometry here; Rust uses f64 std math.
    compare(
        "harmonic_envelope",
        &spectral::harmonic_envelope(&mut t, &amplitude, 0.01, 1024),
        7,
        0.01,
    );
    let frequencies: Vec<_> = (1..=20).map(|i| 200.0 * i as f64).collect();
    for (i, rd) in [0.02, 0.3, 1.0, 2.5, 3.0].into_iter().enumerate() {
        let spectrum = lfmodel::LfModel::from_rd(rd, 0.005, 1.0).spectrum(&frequencies);
        compare(
            &format!("lf_amplitude_{i}"),
            &spectrum.iter().map(|v| v.norm()).collect::<Vec<_>>(),
            1,
            1e-6,
        );
        // C complex exponentials use float approximations and its Brent solver
        // stops at absolute residual 1e-8; analytic integration is tested below.
        compare(
            &format!("lf_phase_{i}"),
            &spectrum.iter().map(|v| v.arg()).collect::<Vec<_>>(),
            1,
            1e-3,
        );
    }
}

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a} != {b} (tol={tolerance})");
}
fn rms(x: &[f64]) -> f64 {
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}
fn mean_std(x: &[f64]) -> (f64, f64) {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    (
        mean,
        (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64).sqrt(),
    )
}

#[test]
fn constants_and_windows_match_conventions() {
    use crate::constants::*;
    close(log_to_db(10f64.ln()), 20.0, 1e-14);
    close(db_to_log(20.0), 10f64.ln(), 1e-14);
    close(log_to_power_db(10f64.ln()), 10.0, 1e-14);
    close(power_db_to_log(10.0), 10f64.ln(), 1e-14);
    close(LOG_CHI2_VARIANCE, PI * PI / 6.0, 0.0);
    close(EULER_GAMMA, std::f64::consts::EULER_GAMMA, 0.0);
    close(LOG_RESIDUAL_BIAS, 0.375, 0.0);
    for kind in [
        Window::Hann,
        Window::Hamming,
        Window::Blackman,
        Window::BlackmanHarris,
        Window::Rectangular,
    ] {
        assert!(window(kind, 0).is_empty());
        assert_eq!(window(kind, 1), [1.0]);
    }
    let hann = window(Window::Hann, 5);
    for (a, b) in hann.iter().zip([0.0, 0.5, 1.0, 0.5, 0.0]) {
        close(*a, b, 1e-15);
    }
    close(
        window(Window::Blackman, 64).iter().sum(),
        64.0 * 0.42,
        1e-13,
    );
    close(
        window(Window::BlackmanHarris, 64).iter().sum(),
        64.0 * 0.35875,
        1e-13,
    );
    assert_eq!(
        window::fetch_frame(&[1.0, 2.0, 3.0], 0, 4),
        [0.0, 0.0, 1.0, 2.0]
    );
}

#[test]
fn transforms_agree_with_direct_sums_and_roundtrip() {
    let mut t = Transforms::default();
    for n in [1, 7, 16, 31, 128] {
        let input: Vec<_> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.33).cos(), (i as f64 * 0.71).sin()))
            .collect();
        let mut actual = input.clone();
        t.fft(&mut actual, false);
        for (k, &v) in actual.iter().enumerate() {
            let expected: Complex = input
                .iter()
                .enumerate()
                .map(|(j, v)| v * Complex::from_polar(1.0, -TAU * j as f64 * k as f64 / n as f64))
                .sum();
            assert!((v - expected).norm() < 1e-10);
        }
        t.fft(&mut actual, true);
        for (a, b) in actual.iter().zip(&input) {
            assert!((a - b).norm() < 1e-12);
        }
        let real: Vec<_> = input.iter().map(|v| v.re).collect();
        let half = t.rfft(&mut real.clone());
        let back = t.irfft(&mut half.clone(), n);
        for (a, b) in back.iter().zip(&real) {
            close(*a, *b, 1e-12);
        }
    }
    for (n, m, omega) in [
        (9, 17, 0.17),
        (33, 4, 0.07),
        (4, 33, -0.5),
        (16, 16, TAU / 16.0),
    ] {
        let x: Vec<_> = (0..n)
            .map(|i| Complex::new((i as f64 * 0.73).sin(), (i as f64 * 0.31).cos()))
            .collect();
        let actual = t.czt(&x, m, omega);
        for (k, &a) in actual.iter().enumerate() {
            let expected: Complex = x
                .iter()
                .enumerate()
                .map(|(j, v)| v * Complex::from_polar(1.0, -omega * j as f64 * k as f64))
                .sum();
            assert!((a - expected).norm() < 1e-10);
        }
    }
    let input: Vec<_> = (0..32)
        .map(|i| Complex::new((i as f64).sin(), 0.0))
        .collect();
    let forward = t.czt(&input, 32, TAU / 32.0);
    let back = t.iczt(&forward, 32, TAU / 32.0);
    for (a, b) in back.iter().zip(input) {
        assert!((a - b).norm() < 1e-12);
    }
}

#[test]
fn dct_scaling_matches_ooura_and_orthonormal_definition() {
    let mut t = Transforms::default();
    for n in [1, 7, 16, 64] {
        let input: Vec<_> = (0..n).map(|i| (i as f64 * 0.7).sin() + 0.5).collect();
        let mut raw = input.clone();
        t.dct2(&mut raw);
        for (k, &value) in raw.iter().enumerate() {
            close(
                value,
                input
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v * (PI * (j as f64 + 0.5) * k as f64 / n as f64).cos())
                    .sum(),
                1e-12,
            );
        }
        raw[0] *= 0.5;
        t.dct3(&mut raw);
        for (a, b) in raw.iter().zip(&input) {
            close(*a * 2.0 / n as f64, *b, 1e-12);
        }
        let mut ortho = input.clone();
        t.dct(&mut ortho);
        close(
            ortho.iter().map(|v| v * v).sum(),
            input.iter().map(|v| v * v).sum(),
            1e-11,
        );
        t.idct(&mut ortho);
        for (a, b) in ortho.iter().zip(input) {
            close(*a, b, 1e-12);
        }
    }
}

#[test]
fn minphase_matches_a_known_causal_filter() {
    let n = 2048;
    let response: Vec<_> = (0..=n / 2)
        .map(|i| {
            Complex::new(1.0, 0.0) - 0.6 * Complex::from_polar(1.0, -TAU * i as f64 / n as f64)
        })
        .collect();
    let logs: Vec<_> = response.iter().map(|v| v.norm().ln()).collect();
    let phase = Transforms::default().minphase(&logs);
    for (a, b) in phase.iter().zip(response) {
        close(*a, b.arg(), 1e-12);
    }
}

#[test]
fn interpolation_handles_query_order_endpoints_and_phase_wrap() {
    let q = [4.0, -1.0, 1.0, 0.5, 3.0];
    assert_eq!(
        interp::interp1(&[0.0, 1.0, 3.0], &[0.0, 2.0, 6.0], &q),
        [6.0, 0.0, 2.0, 1.0, 6.0]
    );
    assert_eq!(
        interp::interp1u(0.0, 4.0, &[0.0, 2.0, 4.0, 6.0], &q),
        [6.0, 0.0, 2.0, 1.0, 6.0]
    );
    assert_eq!(
        interp::fill_blanks(&[0.0, 2.0, 0.0, 0.0, 8.0, 0.0], 0.0),
        [2.0, 2.0, 4.0, 6.0, 8.0, 8.0]
    );
    assert_eq!(
        interp::fill_blanks(&[f64::NAN, 2.0, f64::NAN, 6.0], f64::NAN),
        [2.0, 2.0, 4.0, 6.0]
    );
    assert_eq!(interp::fill_blanks(&[0.0; 3], 0.0), [0.0; 3]);
    assert!(
        interp::fill_blanks(&[f64::NAN; 3], f64::NAN)
            .iter()
            .all(|v| v.is_nan())
    );
    close(interp::circular(PI - 0.1, -PI + 0.1, 0.5), -PI, 1e-15);
    let parabola: Vec<_> = (0..5).map(|i| -((i as f64) - 2.3).powi(2)).collect();
    let (index, value) = interp::qifft(&parabola, 2);
    close(index, 2.3, 1e-14);
    close(value, 0.0, 1e-14);
    assert_eq!(interp::qifft(&[1.0; 3], 1), (1.0, 1.0));
}

#[test]
fn all_chebyshev_rows_are_finite_and_selective() {
    for index in 0..48 {
        let cutoff = (index + 1) as f64 / 100.0;
        for kind in [filter::FilterKind::Lowpass, filter::FilterKind::Highpass] {
            let (b, a) = filter::chebyshev(cutoff, kind);
            assert_eq!(a[0], 1.0);
            let response = |f: f64| {
                let eval = |coef: &[f64]| {
                    coef.iter()
                        .enumerate()
                        .map(|(i, &v)| Complex::from_polar(v, -TAU * f * i as f64))
                        .sum::<Complex>()
                };
                (eval(b) / eval(a)).norm()
            };
            let (pass, stop) = match kind {
                filter::FilterKind::Lowpass => (cutoff * 0.2, cutoff + (0.5 - cutoff) * 0.9),
                filter::FilterKind::Highpass => (cutoff + (0.5 - cutoff) * 0.8, cutoff * 0.1),
            };
            assert!(response(pass) > 0.8, "row {index} {kind:?}");
            assert!(response(stop) < 0.1, "row {index} {kind:?}");
            let mut impulse = vec![0.0; 2048];
            impulse[0] = 1.0;
            assert!(
                filter::filter(b, a, &impulse)
                    .iter()
                    .all(|v| v.is_finite() && v.abs() < 2.0)
            );
        }
    }
}

#[test]
fn filters_preserve_reference_initialization_and_smoothing() {
    let y = filter::filter(&[1.0], &[1.0, -0.5], &[1.0, 0.0, 0.0, 0.0]);
    assert_eq!(y, [1.0, 0.5, 0.25, 0.125]);
    let mut impulse = vec![0.0; 1025];
    impulse[512] = 1.0;
    let (b, a) = filter::chebyshev(0.12, filter::FilterKind::Lowpass);
    let z = filter::filtfilt(b, a, &impulse);
    for i in 1..200 {
        close(z[512 - i], z[512 + i], 1e-12);
    }
    assert_eq!(filter::moving_average(&[2.0; 8], 2.0), [2.0; 8]);
    assert_eq!(
        filter::moving_average(&[0.0, 2.0, 4.0, 6.0], 1.0),
        [1.0, 3.0, 5.0, 6.0]
    );
    let raw = [0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0];
    let smooth = filter::smoothing_filter(&raw, 5);
    assert!(smooth[4] < 4.0);
    assert_eq!(filter::smoothing_filter(&[2.0; 8], 5), [2.0; 8]);
    assert_eq!(filter::smoothing_filter(&[1.0, 2.0], 5), [1.0, 2.0]);
    let q = [0.1; 100];
    let r = [1.0; 100];
    let observations: Vec<_> = (0..100).map(|i| 2.0 + (i as f64 * 1.7).sin()).collect();
    let result = filter::kalman_filter(&observations, &q, &r, 2.0);
    close(result.variances[0], 1.0 / 11.0, 1e-15);
    assert!(result.log_likelihood.is_finite());
    let smoothed = filter::rts_smooth(&result.values, &result.variances, &q);
    close(smoothed[99], result.values[99], 0.0);
    assert!(
        rms(&smoothed.iter().map(|v| v - 2.0).collect::<Vec<_>>())
            < rms(&result.values.iter().map(|v| v - 2.0).collect::<Vec<_>>())
    );
}

#[test]
fn chirp_harmonic_analysis_meets_original_acceptance() {
    let fs = 20000.0;
    let n = 100000;
    let f0: Vec<_> = (0..1000)
        .map(|i| 100.0 + 100.0 * (i * 100) as f64 / n as f64)
        .collect();
    let mut phase = 0.0;
    let x: Vec<_> = (0..n)
        .map(|i| {
            phase = interp::wrap(phase + TAU * (100.0 + 100.0 * i as f64 / n as f64) / fs);
            i as f64 / n as f64 * phase.sin()
                + 0.5 * (2.0 * phase).sin()
                + 0.25 * (3.0 * phase).sin()
        })
        .collect();
    for method in [HarmonicMethod::PeakPicking, HarmonicMethod::ChirpZ] {
        let spectra = spectral::harmonic_analysis(
            &mut Transforms::default(),
            &x,
            &f0,
            &HarmonicAnalysis {
                fs,
                thop: 0.005,
                relative_window: 4.0,
                max_harmonics: 3,
                method,
            },
        );
        for h in 0..3 {
            let errors: Vec<_> = (5..995)
                .map(|i| {
                    spectra[i].as_ref().unwrap().amplitude[h]
                        - if h == 0 {
                            i as f64 / 1000.0
                        } else if h == 1 {
                            0.5
                        } else {
                            0.25
                        }
                })
                .collect();
            let (mean, std) = mean_std(&errors);
            eprintln!("{method:?}, harmonic {h}: amplitude mean={mean:.6}, std={std:.6}");
            assert!(mean.abs() < 0.01 && std < 0.01);
        }
        let errors: Vec<_> = (5..995)
            .map(|i| {
                interp::wrap(
                    spectra[i].as_ref().unwrap().phase[0]
                        - spectra[i - 1].as_ref().unwrap().phase[0]
                        - TAU * f0[i] * 0.005,
                )
            })
            .collect();
        let (mean, std) = mean_std(&errors);
        eprintln!("{method:?}: phase mean={mean:.6}, std={std:.6}");
        assert!(mean.abs() < 0.1 && std < 0.1);
    }
}

#[test]
fn five_hundred_lf_fits_meet_original_acceptance() {
    let grid: Vec<_> = (0..64)
        .map(|i| 0.02 + (3.0 - 0.02) * i as f64 / 63.0)
        .collect();
    let cache = lfmodel::CachedGlottalModel::new(&grid, 20);
    let frequencies: Vec<_> = (1..=20).map(|i| 200.0 * i as f64).collect();
    let mut rng = SeededRng::new(1);
    let mut maximum_error: f64 = 0.0;
    for k in 0..500 {
        let rd = 0.3 + (2.5 - 0.3) * k as f64 / 500.0;
        let scale = 0.01 + rng.uniform() * 5.0;
        let amplitudes: Vec<_> = lfmodel::LfModel::from_rd(rd, 1.0 / 200.0, 1.0)
            .spectrum(&frequencies)
            .iter()
            .enumerate()
            .map(|(j, v)| v.norm() / (j + 1) as f64 * scale)
            .collect();
        let fit = cache.fit(&amplitudes).unwrap();
        maximum_error = maximum_error.max((fit - rd).abs());
        assert!((fit - rd).abs() < 0.02, "Rd={rd}, fit={fit}");
    }
    eprintln!("LF maximum Rd fitting error={maximum_error:.6}");
}

#[test]
fn lf_analytic_spectrum_matches_time_domain_integration() {
    let fs = 2000000.0;
    for rd in [0.3, 1.0, 2.5, 3.0] {
        let lf = lfmodel::LfModel::from_rd(rd, 0.005, 1.0);
        let samples = lf.period(fs, 10000);
        let frequencies = [200.0, 400.0, 1000.0, 2000.0];
        let analytic = lf.spectrum(&frequencies);
        for (&f, &value) in frequencies.iter().zip(&analytic) {
            let integral: Complex = samples
                .iter()
                .enumerate()
                .map(|(i, &v)| Complex::from_polar(v / fs, -TAU * f * i as f64 / fs))
                .sum();
            assert!(
                (integral - value).norm() / value.norm() < 0.003,
                "Rd={rd}, f={f}, expected={value}, integral={integral}"
            );
        }
    }
    // Above 800 Hz the reference solves at 800 Hz and scales the frequency axis.
    let low = lfmodel::LfModel::from_rd(1.0, 1.0 / 800.0, 1.0).spectrum(&[800.0, 1600.0]);
    let high = lfmodel::LfModel::from_rd(1.0, 1.0 / 1200.0, 1.0).spectrum(&[1200.0, 2400.0]);
    close(
        low[1].norm() / low[0].norm(),
        high[1].norm() / high[0].norm(),
        1e-12,
    );
}

#[test]
fn lf_custom_shape_spectrum_matches_integration_across_periods() {
    // This valid slow-return shape used to be rejected solely at long periods.
    let n = 20000;
    for t0 in [0.0001, 1.0 / 800.0, 0.005, 0.1, 0.5, 2.0, 10.0] {
        let lf = lfmodel::LfModel {
            tp: 0.4,
            te: 0.65,
            ta: 0.3,
            t0,
            ee: 1.0,
        };
        let fs = n as f64 / t0;
        let scale = t0.max(1.0 / 800.0) / t0;
        let samples = lf.period(fs, n);
        let frequencies = [0.7 / t0, 1.0 / t0, 3.2 / t0, 10.0 / t0];
        let analytic = lf.spectrum(&frequencies);
        for (&f, &value) in frequencies.iter().zip(&analytic) {
            let integral: Complex = samples
                .iter()
                .enumerate()
                .map(|(i, &v)| Complex::from_polar(v / fs, -TAU * f * i as f64 / fs))
                .sum();
            let expected = value / scale;
            let error = (integral - expected).norm() / expected.norm();
            assert!(error < 0.003, "T0={t0}, f={f}, relative error={error}");
        }
    }
}

#[test]
fn recursive_sinusoids_and_iczt_agree_with_direct_cosines() {
    let mut rng = SeededRng::new(3);
    for (nhar, n, f0) in [
        (100, 1024, 0.01),
        (100, 1023, 0.001),
        (20, 8, 0.01),
        (0, 1, 0.1),
        (1, 0, 0.1),
    ] {
        let ampl: Vec<_> = (0..nhar).map(|_| rng.normal()).collect();
        let phase: Vec<_> = (0..nhar).map(|_| rng.normal() * 100.0).collect();
        let direct: Vec<f64> = (0..n)
            .map(|j| {
                ampl.iter()
                    .zip(&phase)
                    .enumerate()
                    .map(|(i, (&a, &p))| {
                        a * (TAU * f0 * (i + 1) as f64 * (j as f64 - (n / 2) as f64) + p).cos()
                    })
                    .sum()
            })
            .collect();
        let recursive = sinusoid::harmonic_frame(&ampl, &phase, f0, n);
        let iczt = sinusoid::harmonic_frame_iczt(&mut Transforms::default(), &ampl, &phase, f0, n);
        for (a, b) in recursive.iter().zip(&direct) {
            close(*a, *b, 1e-8);
        }
        for (a, b) in iczt.iter().zip(&direct) {
            close(*a, *b, 1e-8);
        }
        for threshold in [None, Some((0.0, -100.0)), Some((0.275, 2.26))] {
            let adaptive =
                sinusoid::synthesize(&mut Transforms::default(), &ampl, &phase, f0, n, threshold);
            for (a, b) in adaptive.iter().zip(&direct) {
                close(*a, *b, 1e-8);
            }
        }
        if n > 1 && nhar > 0 {
            let error: Vec<_> = recursive.iter().zip(&iczt).map(|(a, b)| a - b).collect();
            let snr = 20.0 * (rms(&direct) / rms(&error)).log10();
            eprintln!("nhar={nhar}, n={n}: SNR={snr:.2} dB");
            assert!(snr > 160.0);
        }
    }
}

#[test]
fn psd_normalization_lip_response_and_envelope() {
    let mut t = Transforms::default();
    let n = 512;
    let x: Vec<_> = (0..n)
        .map(|i| (TAU * 20.0 * i as f64 / n as f64).cos())
        .collect();
    let psd = spectral::estimate_psd(&mut t, &x, 1024);
    let energy = (2.0 * psd.iter().sum::<f64>() - psd[0] - psd[512]) / 1024.0;
    close(energy, 0.5, 1e-12);
    let original_ampl = [1.0, 0.5, 0.25];
    let original_phase = [0.1, 0.2, 0.3];
    let mut ampl = original_ampl;
    let mut phase = original_phase;
    let mut complex: Vec<_> = ampl
        .iter()
        .zip(&phase)
        .map(|(&a, &p)| Complex::from_polar(a, p))
        .collect();
    spectral::lipfilter(1.5, 200.0, &mut ampl, Some(&mut phase), false);
    spectral::lipfilter_harmonics_complex(1.5, 200.0, &mut complex, false);
    for (i, v) in complex.iter().enumerate() {
        close(v.norm(), ampl[i], 1e-14);
        close(v.arg(), phase[i], 1e-14);
    }
    spectral::lipfilter(1.5, 200.0, &mut ampl, Some(&mut phase), true);
    spectral::lipfilter_harmonics_complex(1.5, 200.0, &mut complex, true);
    for i in 0..3 {
        close(ampl[i], original_ampl[i], 1e-14);
        close(phase[i], original_phase[i], 1e-14);
        close(complex[i].norm(), ampl[i], 1e-14);
    }
    let ampl: Vec<_> = (1..=40).map(|i| (-0.08 * i as f64).exp()).collect();
    let envelope = spectral::harmonic_envelope(&mut t, &ampl, 0.01, 2048);
    assert!(envelope.iter().all(|v| v.is_finite()));
    // Integer smoothing width gives unity DC gain in the reference convention.
    let flat = spectral::spec2env(&mut t, &vec![2.0; 1025], 0.01171875, 40);
    for v in flat {
        close(v, 2.0_f64.ln(), 1e-11);
    }
    let harmonic_phase = spectral::harmonic_minphase(&mut t, &[1.0; 40]);
    for p in harmonic_phase {
        close(p, 0.0, 1e-14);
    }
    assert_eq!(spectral::compute_dc(&[2.0; 8], &[4], &[4]), [2.0]);
}

#[test]
fn instantaneous_frequency_and_refinement_are_accurate() {
    let fs = 20000.0;
    let f = 203.0;
    let detector = f0::IFDetector::new(200.0 / fs, 200.0 / fs);
    let x: Vec<_> = (0..detector.len())
        .map(|i| (TAU * f / fs * i as f64 + 0.2).cos())
        .collect();
    close(detector.estimate(&x) * fs, f, 0.01);
    assert_eq!(detector.estimate(&vec![0.0; detector.len()]), 0.0);
    let x: Vec<_> = (0..20000)
        .map(|i| (TAU * f / fs * i as f64).cos() + 0.5 * (TAU * 2.0 * f / fs * i as f64).cos())
        .collect();
    let mut track = vec![200.0; 100];
    track[40] = 0.0;
    let refined = f0::refine_f0(&x, fs, &track, 0.005, 10.0);
    assert_eq!(track[20], 200.0);
    assert_eq!(refined[40], 0.0);
    for &value in &refined[10..35] {
        close(value, f, 0.1);
    }
}

#[test]
fn noise_is_repeatable_gaussian_and_bandlimited() {
    let mut a = SeededRng::new(0);
    let mut b = SeededRng::new(0);
    let x = noise::white_noise(&mut a, 60000);
    assert_eq!(x, noise::white_noise(&mut b, 60000));
    assert_eq!(&x[..20000], &x[20000..40000]);
    let (mean, std) = mean_std(&x);
    assert!(mean.abs() < 0.03 && (std - 1.0).abs() < 0.03);
    let colored = noise::bandlimited_noise(&mut a, 60000, 0.05, 0.15);
    let mut t = Transforms::default();
    let psd = spectral::estimate_psd(&mut t, &colored[1000..17384], 16384);
    let power: f64 = psd.iter().sum();
    let in_band: f64 = psd[819..2458].iter().sum();
    // Fourth-order tabulated filters retain a transition band outside cutoffs.
    assert!(
        in_band / power > 0.85,
        "in-band fraction={}",
        in_band / power
    );
    assert!(noise::bandlimited_noise(&mut a, 1, 0.0, 0.5)[0].is_finite());
    assert!(noise::bandlimited_noise(&mut a, 0, 0.0, 0.5).is_empty());
    let white = noise::white_noise(&mut a, 20000);
    let repeated = noise::stretch_stationary(&white, 200000, 128);
    let (_, std) = mean_std(&repeated);
    assert!((std - 1.0).abs() < 0.03);
}

#[test]
fn empirical_kld_self_test_matches_known_distributions() {
    let mut rng = SeededRng::new(1);
    let x: Vec<_> = (0..100000).map(|_| 1.0 + rng.normal()).collect();
    let same: Vec<_> = (0..50000).map(|_| 1.0 + rng.normal()).collect();
    let shifted: Vec<_> = (0..50000).map(|_| rng.normal()).collect();
    let wider: Vec<_> = (0..50000)
        .map(|_| 1.0 + 3.0_f64.sqrt() * rng.normal())
        .collect();
    let uniform: Vec<_> = (0..50000).map(|_| rng.uniform()).collect();
    let d0 = verify::empirical_kld(&x, &same);
    let d1 = verify::empirical_kld(&x, &shifted);
    let d2 = verify::empirical_kld(&x, &wider);
    let d3 = verify::empirical_kld(&x, &uniform);
    eprintln!("KLD self-test: {d0:.5}, {d1:.5}, {d2:.5}, {d3:.5}");
    assert!(d0.abs() < 0.05 && d1 > d0 && d2 > d0 && d3 > d0);
    close(d1, 0.5, 0.05);
    close(d2, 0.5 * 3.0_f64.ln() - 1.0 / 3.0, 0.05);
}

#[test]
fn verify_utils_accept_equivalent_harmonic_renderings() {
    let mut rng = SeededRng::new(10);
    let ampl: Vec<_> = (0..40).map(|i| rng.uniform() / (i + 1) as f64).collect();
    let phase: Vec<_> = (0..40).map(|_| TAU * rng.uniform()).collect();
    // Irrational spacing avoids duplicate sample values in the KLD estimator.
    let x = sinusoid::harmonic_frame(&ampl, &phase, 0.007_071_067_811_865_476, 20000);
    let y = sinusoid::harmonic_frame_iczt(
        &mut Transforms::default(),
        &ampl,
        &phase,
        0.007_071_067_811_865_476,
        20000,
    );
    verify::verify_waveform(&x, &y);
    verify::verify_spectrum(&x, &y);
}
