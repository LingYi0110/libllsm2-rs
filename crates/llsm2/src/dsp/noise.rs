use super::filter::bandpass;
use std::f64::consts::TAU;

/// Injectable RNG contract: uniform variates strictly between 0 and 1.
pub(crate) trait RandomSource {
    fn uniform(&mut self) -> f64;
    fn normal(&mut self) -> f64 {
        (-2.0 * self.uniform().ln()).sqrt() * (TAU * self.uniform()).cos()
    }
}

/// SplitMix64; every u64 seed, including zero, defines a valid stream.
pub(crate) struct SeededRng {
    state: u64,
}
impl SeededRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}
impl Default for SeededRng {
    fn default() -> Self {
        Self::new(1)
    }
}
impl RandomSource for SeededRng {
    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        // 52 bits + half a unit stays strictly below one after f64 rounding.
        ((z >> 12) as f64 + 0.5) / (1u64 << 52) as f64
    }
}

pub(crate) fn white_noise(rng: &mut impl RandomSource, n: usize) -> Vec<f64> {
    let mut out: Vec<_> = (0..n.min(20000)).map(|_| rng.normal()).collect();
    for i in out.len()..n {
        out.push(out[i % 20000]);
    }
    out
}

/// Repeat a stationary template with equal-power circular crossfades.
/// The template is nonempty and overlap < template.len().
pub(crate) fn stretch_stationary(x: &[f64], n: usize, overlap: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    let initial = x.len().min(n);
    y[..initial].copy_from_slice(&x[..initial]);
    let mut head = x.len();
    while head < n {
        for (i, &sample) in x.iter().enumerate().take(overlap) {
            let r = i as f64 / overlap as f64;
            let index = head - overlap + i;
            y[index] = (y[index] * (1.0 - r) + sample * r) / ((1.0 - r).powi(2) + r * r).sqrt();
        }
        let count = (x.len() - overlap).min(n - head);
        y[head..head + count].copy_from_slice(&x[overlap..overlap + count]);
        head += count;
    }
    y
}

pub(crate) fn bandlimited_noise(
    rng: &mut impl RandomSource,
    n: usize,
    low: f64,
    high: f64,
) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let len = n.min(20000);
    let white = white_noise(rng, len + 128);
    let colored = bandpass(&white, low, high);
    stretch_stationary(&colored[..len], n, 128.min(len - 1))
}
