use super::window::fetch_frame;
use std::f64::consts::TAU;

/// Flanagan instantaneous-frequency detector, frequencies in cycles/sample.
pub(crate) struct IFDetector {
    h: Vec<[f64; 4]>,
}

impl IFDetector {
    pub(crate) fn new(center: f64, resolution: f64) -> Self {
        let n = (4.0 / resolution) as usize;
        let omega = TAU * center;
        let omega_w = TAU / n as f64;
        let h = (0..n)
            .map(|i| {
                let t = i as f64 - (n / 2) as f64;
                let mut w = 0.0;
                let mut wd = 0.0;
                for (k, a) in [0.338946, 0.481973, 0.161054, 0.018027].iter().enumerate() {
                    let kw = k as f64 * omega_w;
                    w += a * (kw * t).cos();
                    wd -= kw * a * (kw * t).sin();
                }
                let (sin, cos) = (omega * t).sin_cos();
                [
                    w * cos,
                    w * sin,
                    wd * cos - omega * w * sin,
                    omega * w * cos + wd * sin,
                ]
            })
            .collect();
        Self { h }
    }
    pub(crate) fn len(&self) -> usize {
        self.h.len()
    }
    /// Returns zero on silence or insufficient samples, where frequency is undefined.
    pub(crate) fn estimate(&self, x: &[f64]) -> f64 {
        if x.len() < self.h.len() {
            return 0.0;
        }
        let start = x.len() / 2 - self.h.len() / 2;
        let mut y = [0.0; 4];
        for (h, &x) in self.h.iter().zip(&x[start..]) {
            for k in 0..4 {
                y[k] += h[k] * x;
            }
        }
        let power = y[0] * y[0] + y[1] * y[1];
        if power == 0.0 {
            0.0
        } else {
            (y[0] * y[3] - y[1] * y[2]) / power / TAU
        }
    }
}

/// Refine a read-only F0 track (Hz); unvoiced frames remain zero. Voiced input
/// and accepted IF candidates lie in [minimum_f0, fs/2], as set by the caller.
pub(crate) fn refine_f0(x: &[f64], fs: f64, f0: &[f64], thop: f64, minimum_f0: f64) -> Vec<f64> {
    f0.iter()
        .enumerate()
        .map(|(i, &f)| {
            if f == 0.0 {
                return 0.0;
            }
            let mut sum = 0.0;
            let mut count = 0;
            for j in 1..=3 {
                let detector = IFDetector::new(f / fs * j as f64, f / fs);
                let frame = fetch_frame(x, (i as f64 * thop * fs).round() as isize, detector.len());
                let estimate = detector.estimate(&frame) / j as f64;
                if estimate >= minimum_f0 / fs
                    && estimate <= 0.5
                    && (estimate - f / fs).abs() < f / fs * 0.1
                {
                    sum += estimate;
                    count += 1;
                }
            }
            if count > 0 {
                sum / count as f64 * fs
            } else {
                f
            }
        })
        .collect()
}
