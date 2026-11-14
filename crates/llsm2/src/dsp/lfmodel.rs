use super::{fft::Complex, interp::qifft};
use std::f64::consts::{PI, TAU};

/// LF opening/closing times as fractions of the fundamental period.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LfModel {
    pub(crate) tp: f64,
    pub(crate) te: f64,
    pub(crate) ta: f64,
    pub(crate) t0: f64,
    pub(crate) ee: f64,
}

/// Brent's bracketed root search. The caller supplies a sign-changing bracket.
fn brent(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let mut fa = f(a);
    let mut fb = f(b);
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = c;
    let mut bisected = true;
    loop {
        let tol = 1e-12 * b.abs();
        if fb == 0.0 || (b - a).abs() <= tol {
            return b;
        }
        let mut s = if fa != fc && fb != fc {
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            b - fb * (b - a) / (fb - fa)
        };
        let bound = (3.0 * a + b) / 4.0;
        if s < bound.min(b)
            || s > bound.max(b)
            || (bisected && (s - b).abs() >= (b - c).abs() / 2.0)
            || (!bisected && (s - b).abs() >= (c - d).abs() / 2.0)
            || (bisected && (b - c).abs() < tol)
            || (!bisected && (c - d).abs() < tol)
        {
            s = (a + b) / 2.0;
            bisected = true;
        } else {
            bisected = false;
        }
        let fs = f(s);
        d = c;
        c = b;
        fc = fb;
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }
}

/// Positive dimensionless root u = epsilon * T0 of
/// 1 - exp(-(1-te)*u) = ta*u, for 0 < ta < 1-te.
fn return_decay(te: f64, ta: f64) -> f64 {
    let duration = 1.0 - te;
    let ratio = ta / duration;
    let gap = (duration - ta) / duration;
    // With x=(1-te)*u, divide out the unwanted zero root. Taylor's bound
    // 1-(1-exp(-x))/x < x/2 gives a positive lower endpoint at x=gap;
    // x=2/ratio gives a negative upper endpoint, independent of T0.
    brent(
        |x| {
            if x < 0.01 {
                // Evaluate the loss from 1 without cancellation near ta=1-te.
                let loss = x
                    * (0.5
                        + x * (-1.0 / 6.0
                            + x * (1.0 / 24.0
                                + x * (-1.0 / 120.0 + x * (1.0 / 720.0 - x / 5040.0)))));
                gap - loss
            } else {
                -(-x).exp_m1() / x - ratio
            }
        },
        gap,
        2.0 / ratio,
    ) / duration
}

struct LfParameters {
    t0: f64,
    te: f64,
    ta: f64,
    wg: f64,
    epsilon: f64,
    alpha: f64,
    e0: f64,
    scale: f64,
}

impl LfModel {
    pub(crate) fn from_rd(rd: f64, t0: f64, ee: f64) -> Self {
        let ra = if rd < 0.21 {
            1e-6
        } else if rd < 2.7 {
            (-1.0 + 4.8 * rd) / 100.0
        } else {
            0.323 / rd
        };
        let (rk, rg) = if rd < 2.7 {
            let rk = (22.4 + 11.8 * rd) / 100.0;
            (rk, 0.25 * rk / ((0.11 * rd) / (0.5 + 1.2 * rk) - ra))
        } else {
            let oq = 1.0 - 1.0 / (2.17 * rd);
            let rg = 0.0093552 + 5.96 / (7.96 - 2.0 * oq);
            (2.0 * rg * oq - 1.0428, rg)
        };
        let tp = 1.0 / (2.0 * rg);
        Self {
            tp,
            te: tp * (rk + 1.0),
            ta: ra,
            t0,
            ee,
        }
    }

    fn parameters(self) -> LfParameters {
        let t0 = self.t0.max(1.0 / 800.0);
        let scale = t0 / self.t0;
        let te = t0 * self.te;
        let ta = t0 * self.ta;
        let wg = PI / (t0 * self.tp);
        let sin = (wg * te).sin();
        let cos = (wg * te).cos();
        let epsilon = return_decay(self.te, self.ta) / t0;
        let decay = (epsilon * (te - t0)).exp();
        let area = (1.0 - decay) / (epsilon * epsilon * ta) + (te - t0) * decay / (epsilon * ta);
        let constant = wg * wg * sin * area - wg * cos;
        let mut alpha = 0.0;
        for _ in 0..8 {
            let value =
                sin * area * alpha * alpha + sin * alpha + wg * (-alpha * te).exp() + constant;
            let derivative = 2.0 * sin * area * alpha + sin - wg * te * (-alpha * te).exp();
            alpha -= value / derivative;
        }
        LfParameters {
            t0,
            te,
            ta,
            wg,
            epsilon,
            alpha,
            e0: -self.ee / ((alpha * te).exp() * sin),
            scale,
        }
    }

    /// Analytic spectrum of differentiated glottal flow at positive frequencies (Hz).
    /// The DC value is zero because the pulse integrates to zero over one period.
    pub(crate) fn spectrum(self, frequencies: &[f64]) -> Vec<Complex> {
        let p = self.parameters();
        let e = p.epsilon;
        let tail = e * (-e * (p.t0 - p.te)).exp();
        let sin = (p.wg * p.te).sin();
        let cos = (p.wg * p.te).cos();
        frequencies
            .iter()
            .map(|&f| {
                if f == 0.0 {
                    return Complex::default();
                }
                let omega = TAU * f / p.scale;
                let z = Complex::new(p.alpha, -omega);
                let p1 = p.e0 / (z * z + p.wg * p.wg);
                let p2 = p.wg + (z * p.te).exp() * (z * sin - p.wg * cos);
                let p3 = self.ee * Complex::from_polar(1.0, -omega * p.te)
                    / (Complex::new(0.0, e * p.ta * omega) * Complex::new(e, omega));
                // Keep the full return term at every period. Compute 1-exp(-iθ)
                // without subtracting nearly equal real parts at small θ.
                let theta = omega * (p.t0 - p.te);
                let p4 = tail * Complex::new(2.0 * (theta / 2.0).sin().powi(2), theta.sin())
                    - Complex::new(0.0, e * p.ta * omega);
                p1 * p2 + p3 * p4
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn period(self, fs: f64, n: usize) -> Vec<f64> {
        let p = self.parameters();
        let end_open = (p.te * fs / p.scale).round() as usize;
        let end_period = (p.t0 * fs / p.scale).round() as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / fs * p.scale;
                if i <= end_open {
                    p.e0 * (p.alpha * t).exp() * (p.wg * t).sin()
                } else if i < end_period {
                    -self.ee / (p.epsilon * p.ta)
                        * ((-p.epsilon * (t - p.te)).exp() - (-p.epsilon * (p.t0 - p.te)).exp())
                } else {
                    0.0
                }
            })
            .collect()
    }
}

pub(crate) struct CachedGlottalModel {
    parameters: Vec<f64>,
    powers: Vec<Vec<f64>>,
}

impl CachedGlottalModel {
    /// Strictly increasing Rd grid and at least two harmonics.
    pub(crate) fn new(parameters: &[f64], nhar: usize) -> Self {
        let frequencies: Vec<_> = (1..=nhar).map(|i| 200.0 * i as f64).collect();
        let powers = parameters
            .iter()
            .map(|&rd| {
                LfModel::from_rd(rd, 1.0 / 200.0, 1.0)
                    .spectrum(&frequencies)
                    .iter()
                    .enumerate()
                    .map(|(j, v)| v.norm_sqr() / ((j + 1) as f64).powi(2))
                    .collect()
            })
            .collect();
        Self {
            parameters: parameters.to_vec(),
            powers,
        }
    }

    /// Scale-independent Itakura–Saito fit of flow amplitudes (derivative / k).
    /// Returns None if fewer than two harmonics or any zero power prevents a
    /// finite, identifiable normalized IS fit. The caller can mark Rd as missing.
    pub(crate) fn fit(&self, amplitudes: &[f64]) -> Option<f64> {
        let n = amplitudes.len().min(self.powers[0].len());
        if n < 2 {
            return None;
        }
        let power: Vec<_> = amplitudes[..n].iter().map(|v| v * v).collect();
        if power.contains(&0.0) {
            return None;
        }
        let distances: Vec<f64> = self
            .powers
            .iter()
            .map(|model| {
                let gain = power[0] / model[0];
                (0..n)
                    .map(|j| {
                        let ratio = power[j] / (model[j] * gain);
                        ratio - ratio.ln() - 1.0
                    })
                    .sum::<f64>()
                    / n as f64
            })
            .collect();
        let valley = (0..distances.len())
            .min_by(|&a, &b| distances[a].total_cmp(&distances[b]))
            .unwrap();
        if valley == 0 || valley + 1 == distances.len() {
            return Some(self.parameters[valley]);
        }
        let (index, _) = qifft(&distances, valley);
        let i = index as usize;
        Some(
            self.parameters[i] + (index - i as f64) * (self.parameters[i + 1] - self.parameters[i]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_decay_has_the_positive_root_independent_of_period() {
        // Independently computed with 70-digit decimal bisection.
        let expected = 0.904_714_258_605_824_5;
        assert!((return_decay(0.65, 0.3) - expected).abs() < 1e-12);
        assert!((return_decay(0.65, 1e-6) / 1e6 - 1.0).abs() < 1e-12);
        // Near the linear-return limit, the positive root tends to zero.
        // Use exactly representable times to avoid decimal input uncertainty.
        for gap in [2.0_f64.powi(-20), 2.0_f64.powi(-52)] {
            let root = return_decay(0.5, 0.5 * (1.0 - gap));
            assert!(root > 0.0);
            assert!((root / (4.0 * gap) - 1.0).abs() < 2.0 * gap + 1e-12);
        }
        for t0 in [0.0001, 1.0 / 800.0, 0.005, 0.1, 0.5, 2.0, 10.0] {
            let lf = LfModel {
                tp: 0.4,
                te: 0.65,
                ta: 0.3,
                t0,
                ee: 1.0,
            };
            let p = lf.parameters();
            assert!((p.epsilon * t0.max(1.0 / 800.0) - expected).abs() < 1e-12);
        }
    }
}
