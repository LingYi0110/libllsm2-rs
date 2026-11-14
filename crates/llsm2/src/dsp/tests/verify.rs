//! Statistical acceptance helpers ported from libllsm2's verify-utils.h.
use super::super::{
    fft::Transforms,
    noise::{RandomSource, SeededRng},
    spectral::spectrogram,
    window::Window,
};

pub(super) fn empirical_kld(x: &[f64], y: &[f64]) -> f64 {
    let mut x = x.to_vec();
    let mut y = y.to_vec();
    x.sort_by(f64::total_cmp);
    y.sort_by(f64::total_cmp);
    y[0] = x[0];
    let last = y.len() - 1;
    y[last] = x[x.len() - 1];
    let mut yi = 0;
    let mut distance = 0.0;
    for i in 0..x.len() {
        while yi < y.len() - 1 && y[yi] < x[i] {
            yi += 1;
        }
        yi = yi.max(1);
        let xi = i.max(1);
        let dx = (x[xi] - x[xi - 1]).max(1e-10);
        let dy = (y[yi] - y[yi - 1]).max(1e-10);
        distance += (y.len() as f64 * dy / (x.len() as f64 * dx)).ln();
    }
    distance / x.len() as f64 - 1.0
}

pub(super) fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let mx = x.iter().sum::<f64>() / x.len() as f64;
    let my = y.iter().sum::<f64>() / y.len() as f64;
    let covariance: f64 = x.iter().zip(y).map(|(x, y)| (x - mx) * (y - my)).sum();
    covariance
        / (x.iter().map(|v| (v - mx).powi(2)).sum::<f64>()
            * y.iter().map(|v| (v - my).powi(2)).sum::<f64>())
        .sqrt()
}

fn dither(x: &[f64], rng: &mut SeededRng) -> Vec<f64> {
    x.iter().map(|&v| v + 1e-4 * rng.normal()).collect()
}

pub(super) fn verify_waveform(x: &[f64], y: &[f64]) {
    let mut rng = SeededRng::new(8123);
    let mut x = x.to_vec();
    let mut y = y.to_vec();
    for order in 0..=2 {
        let kld = empirical_kld(&dither(&x, &mut rng), &dither(&y, &mut rng));
        assert!(kld < 0.05, "waveform order {order}: KLD={kld}");
        x = x.windows(2).map(|v| v[1] - v[0]).collect();
        y = y.windows(2).map(|v| v[1] - v[0]).collect();
    }
}

pub(super) fn verify_spectrum(x: &[f64], y: &[f64]) {
    let mut t = Transforms::default();
    let centers: Vec<_> = (0..x.len().min(y.len()) / 512)
        .map(|i| (i * 512) as isize)
        .collect();
    let sizes = vec![2048; centers.len()];
    let mut x: Vec<_> = spectrogram(&mut t, x, &centers, &sizes, 2048, Window::Blackman)
        .into_iter()
        .map(|v| v.amplitude)
        .collect();
    let mut y: Vec<_> = spectrogram(&mut t, y, &centers, &sizes, 2048, Window::Blackman)
        .into_iter()
        .map(|v| v.amplitude)
        .collect();
    let flat = |x: &[Vec<f64>]| x.iter().flatten().copied().collect::<Vec<_>>();
    assert!(correlation(&flat(&x), &flat(&y)) > 0.95);
    let mut rng = SeededRng::new(5432);
    for order in 0..=1 {
        let kld = empirical_kld(&dither(&flat(&x), &mut rng), &dither(&flat(&y), &mut rng));
        assert!(kld < 0.05, "spectral order {order}: KLD={kld}");
        x = x
            .windows(2)
            .map(|w| w[0].iter().zip(&w[1]).map(|(a, b)| b - a).collect())
            .collect();
        y = y
            .windows(2)
            .map(|w| w[0].iter().zip(&w[1]).map(|(a, b)| b - a).collect())
            .collect();
    }
}
