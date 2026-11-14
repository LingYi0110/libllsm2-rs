use rustfft::{FftPlanner, num_complex::Complex64};
use std::{f64::consts::TAU, path::Path};

pub fn speech() -> (Vec<f64>, f64) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio/arctic_a0001.wav");
    let mut reader = hound::WavReader::open(path).unwrap();
    assert_eq!(reader.spec().channels, 1);
    let fs = reader.spec().sample_rate as f64;
    let samples = reader
        .samples::<i16>()
        .map(|v| v.unwrap() as f64 / 32768.0)
        .collect();
    (samples, fs)
}

/// Small offline YIN test driver, not part of the library's pitch API.
/// Four-sample averaging/decimation limits cost; CMNDF < .2 selects a voiced
/// local trough in 50..500 Hz. There are no hand-edited or C-generated labels.
pub fn yin(x: &[f64], fs: f64, hop: f64) -> Vec<f64> {
    let down: Vec<_> = x
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| v.iter().sum::<f64>() / 4.0)
        .collect();
    let ds = fs / 4.0;
    let minimum = (ds / 500.0).floor() as usize;
    let maximum = (ds / 50.0).ceil() as usize;
    let size = (ds * 0.025).ceil() as usize;
    let sample = |i: isize| {
        if i < 0 || i as usize >= down.len() {
            0.0
        } else {
            down[i as usize]
        }
    };
    let mut contour = Vec::new();
    for frame in 0..(x.len() as f64 / hop).floor() as usize {
        let center = (frame as f64 * hop / 4.0).round() as isize;
        let energy = (0..size)
            .map(|j| sample(center + j as isize - (size / 2) as isize).powi(2))
            .sum::<f64>()
            / size as f64;
        if energy < 1e-6 {
            contour.push(0.0);
            continue;
        }
        let mut difference = vec![0.0; maximum + 1];
        for (tau, difference) in difference.iter_mut().enumerate().skip(1) {
            let start = center - ((size + tau) / 2) as isize;
            *difference = (0..size)
                .map(|j| (sample(start + j as isize) - sample(start + (j + tau) as isize)).powi(2))
                .sum();
        }
        let mut cmndf = vec![1.0; maximum + 1];
        let mut cumulative = 0.0;
        for tau in 1..=maximum {
            cumulative += difference[tau];
            if cumulative > 0.0 {
                cmndf[tau] = difference[tau] * tau as f64 / cumulative;
            }
        }
        let mut tau = minimum;
        while tau < maximum && cmndf[tau] >= 0.2 {
            tau += 1;
        }
        if tau == maximum {
            contour.push(0.0);
            continue;
        }
        while tau + 1 < maximum && cmndf[tau + 1] < cmndf[tau] {
            tau += 1;
        }
        let delta = 0.5 * (cmndf[tau - 1] - cmndf[tau + 1])
            / (cmndf[tau - 1] - 2.0 * cmndf[tau] + cmndf[tau + 1]);
        contour.push(ds / (tau as f64 + delta));
    }
    contour
}

// Perez-Cruz estimator and dither amplitude ported from verify-utils.h.
pub fn empirical_kld(x: &[f64], y: &[f64]) -> f64 {
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

struct TestRng(u64);
impl TestRng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 12) as f64 + 0.5) / (1u64 << 52) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (TAU * self.uniform()).cos()
    }
    fn dither(&mut self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|v| v + 1e-4 * self.normal()).collect()
    }
}

/// Raw-magnitude STFT used by the C acceptance suite (2048 samples, hop 512).
/// This is independent of llsm2's private STFT and its amplitude normalization.
fn spectra(x: &[f64]) -> Vec<Vec<f64>> {
    let size = 2048;
    let plan = FftPlanner::new().plan_fft_forward(size);
    let window: Vec<_> = (0..size)
        .map(|i| {
            0.42 - 0.5 * (TAU * i as f64 / size as f64).cos()
                + 0.08 * (2.0 * TAU * i as f64 / size as f64).cos()
        })
        .collect();
    let mut input = vec![Complex64::default(); size];
    (0..x.len() / 512)
        .map(|frame| {
            for (i, v) in input.iter_mut().enumerate() {
                let source = (frame * 512 + i) as isize - (size / 2) as isize;
                *v = Complex64::new(
                    if source >= 0 && (source as usize) < x.len() {
                        x[source as usize] * window[i]
                    } else {
                        0.0
                    },
                    0.0,
                );
            }
            plan.process(&mut input);
            input[..=size / 2].iter().map(|v| v.norm()).collect()
        })
        .collect()
}

#[derive(Debug)]
pub struct Metrics {
    pub waveform: [f64; 3],
    pub correlation: f64,
    pub spectral: [f64; 2],
}
impl Metrics {
    pub fn passes(&self) -> bool {
        self.waveform
            .iter()
            .chain(&self.spectral)
            .all(|&v| v.is_finite() && v < 0.05)
            && self.correlation > 0.95
    }
}
pub fn metrics(x: &[f64], y: &[f64]) -> Metrics {
    let mut rng = TestRng(77);
    let mut waveform = [0.0; 3];
    let mut dx = x.to_vec();
    let mut dy = y.to_vec();
    for value in &mut waveform {
        *value = empirical_kld(&rng.dither(&dx), &rng.dither(&dy));
        dx = dx.windows(2).map(|v| v[1] - v[0]).collect();
        dy = dy.windows(2).map(|v| v[1] - v[0]).collect();
    }
    let mut sx = spectra(x);
    let mut sy = spectra(y);
    let flat = |x: &[Vec<f64>]| x.iter().flatten().copied().collect::<Vec<_>>();
    let n = sx.len().min(sy.len());
    let a = flat(&sx[..n]);
    let b = flat(&sy[..n]);
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let correlation = a
        .iter()
        .zip(&b)
        .map(|(a, b)| (a - ma) * (b - mb))
        .sum::<f64>()
        / (a.iter().map(|a| (a - ma).powi(2)).sum::<f64>()
            * b.iter().map(|b| (b - mb).powi(2)).sum::<f64>())
        .sqrt();
    let mut spectral = [0.0; 2];
    for value in &mut spectral {
        *value = empirical_kld(&rng.dither(&flat(&sx)), &rng.dither(&flat(&sy)));
        sx = sx
            .windows(2)
            .map(|v| v[0].iter().zip(&v[1]).map(|(a, b)| b - a).collect())
            .collect();
        sy = sy
            .windows(2)
            .map(|v| v[0].iter().zip(&v[1]).map(|(a, b)| b - a).collect())
            .collect();
    }
    Metrics {
        waveform,
        correlation,
        spectral,
    }
}
