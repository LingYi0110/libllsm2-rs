use super::fft::{Complex, Transforms};
use std::f64::consts::TAU;

/// Independent oscillators are grouped into SIMD lanes; the recurrence still
/// uses f64 and separate multiply/subtract, without an implicit fused operation.
fn recursive_into(ampl: &[f64], phase: &[f64], frequency: impl Fn(usize) -> f64, out: &mut [f64]) {
    out.fill(0.0);
    let n = out.len();
    #[cfg(not(feature = "simd"))]
    let first = 0;
    #[cfg(feature = "simd")]
    let first = ampl.len() / 4 * 4;
    #[cfg(feature = "simd")]
    {
        use wide::f64x4;
        for base in (0..ampl.len() / 4 * 4).step_by(4) {
            let omega: [f64; 4] = std::array::from_fn(|k| TAU * frequency(base + k));
            let c = f64x4::new(omega.map(|w| 2.0 * w.cos()));
            let mut previous = f64x4::new(std::array::from_fn(|k| {
                ampl[base + k] * (phase[base + k] - omega[k] * ((n / 2) as f64 + 1.0)).cos()
            }));
            let mut current = f64x4::new(std::array::from_fn(|k| {
                ampl[base + k] * (phase[base + k] - omega[k] * (n / 2) as f64).cos()
            }));
            for y in &mut *out {
                *y += current.reduce_add();
                let next = c * current - previous;
                previous = current;
                current = next;
            }
        }
    }
    for k in first..ampl.len() {
        let omega = TAU * frequency(k);
        let c = 2.0 * omega.cos();
        let mut previous = ampl[k] * (phase[k] - omega * ((n / 2) as f64 + 1.0)).cos();
        let mut current = ampl[k] * (phase[k] - omega * (n / 2) as f64).cos();
        for y in &mut *out {
            *y += current;
            let next = c * current - previous;
            previous = current;
            current = next;
        }
    }
}
pub(crate) fn harmonic_frame(ampl: &[f64], phase: &[f64], f0: f64, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    recursive_into(ampl, phase, |k| (k + 1) as f64 * f0, &mut out);
    out
}

/// Per-instance reusable coefficients and Bluestein convolution arrays.
pub(crate) struct SynthesisWorkspace {
    coefficients: Vec<Complex>,
    czt: super::fft::CztWorkspace,
}
impl SynthesisWorkspace {
    pub(crate) fn warm(&mut self, t: &mut Transforms) {
        self.czt.warm(t);
    }
    pub(crate) fn new(max_harmonics: usize, max_samples: usize) -> Self {
        Self {
            coefficients: vec![Complex::default(); max_harmonics + 1],
            czt: super::fft::CztWorkspace::new(max_harmonics + 1, max_samples),
        }
    }
    pub(crate) fn synthesize_into(
        &mut self,
        t: &mut Transforms,
        ampl: &[f64],
        phase: &[f64],
        f0: f64,
        out: &mut [f64],
        threshold: Option<(f64, f64)>,
    ) {
        let n = out.len();
        if n == 0 {
            return;
        }
        if threshold.is_some_and(|(a, b)| (n as f64).ln() * a + b < (ampl.len() as f64).ln()) {
            let coefficients = &mut self.coefficients[..ampl.len() + 1];
            coefficients.fill(Complex::default());
            for (i, (&a, &p)) in ampl.iter().zip(phase).enumerate() {
                coefficients[i + 1] =
                    Complex::from_polar(a, p - TAU * f0 * (i + 1) as f64 * (n / 2) as f64);
            }
            let values = self.czt.process(t, coefficients, n, -TAU * f0);
            for (out, v) in out.iter_mut().zip(values) {
                *out = v.re;
            }
        } else {
            recursive_into(ampl, phase, |k| (k + 1) as f64 * f0, out);
        }
    }
}

pub(crate) fn harmonic_frame_iczt(
    transforms: &mut Transforms,
    ampl: &[f64],
    phase: &[f64],
    f0: f64,
    n: usize,
) -> Vec<f64> {
    let mut coefficients = vec![Complex::default(); ampl.len() + 1];
    for (i, (&a, &p)) in ampl.iter().zip(phase).enumerate() {
        coefficients[i + 1] =
            Complex::from_polar(a, p - TAU * f0 * (i + 1) as f64 * (n / 2) as f64);
    }
    // Positive exponent, unnormalized harmonic sum. Unlike the C nx-limited
    // temporary buffer, this also retains harmonics when nhar >= n.
    transforms
        .czt(&coefficients, n, -TAU * f0)
        .iter()
        .map(|v| v.re)
        .collect()
}

pub(crate) fn synthesize(
    transforms: &mut Transforms,
    ampl: &[f64],
    phase: &[f64],
    f0: f64,
    n: usize,
    threshold: Option<(f64, f64)>,
) -> Vec<f64> {
    if threshold.is_some_and(|(a, b)| (n as f64).ln() * a + b < (ampl.len() as f64).ln()) {
        harmonic_frame_iczt(transforms, ampl, phase, f0, n)
    } else {
        harmonic_frame(ampl, phase, f0, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reusable_recursive_and_czt_paths_match_direct_harmonic_sums() {
        let mut t = Transforms::default();
        let mut workspace = SynthesisWorkspace::new(101, 257);
        for (nhar, n) in [(0, 0), (0, 17), (1, 1), (4, 255), (25, 256), (101, 257)] {
            let ampl: Vec<_> = (0..nhar).map(|i| 0.1 / (i + 1) as f64).collect();
            let phase: Vec<_> = (0..nhar).map(|i| i as f64 * 0.71).collect();
            for threshold in [None, Some((0.0, -100.0))] {
                let mut out = vec![f64::NAN; n];
                workspace.synthesize_into(&mut t, &ampl, &phase, 0.00371, &mut out, threshold);
                for (j, &value) in out.iter().enumerate() {
                    let expected: f64 = ampl
                        .iter()
                        .zip(&phase)
                        .enumerate()
                        .map(|(k, (&a, &p))| {
                            a * (p + TAU * 0.00371 * (k + 1) as f64 * (j as f64 - (n / 2) as f64))
                                .cos()
                        })
                        .sum();
                    assert!(
                        (value - expected).abs() < 1e-10,
                        "nhar={nhar}, n={n}, j={j}"
                    );
                }
            }
        }
    }
}
