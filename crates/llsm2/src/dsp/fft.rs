use realfft::RealFftPlanner;
#[cfg(test)]
use rustdct::DctPlanner;
use rustfft::FftPlanner;
pub(crate) use rustfft::num_complex::Complex64 as Complex;

pub(crate) struct CztWorkspace {
    a: Vec<Complex>,
    b: Vec<Complex>,
}
impl CztWorkspace {
    pub(crate) fn warm(&mut self, t: &mut Transforms) {
        let mut n = 1;
        while n <= self.a.len() {
            t.fft(&mut self.a[..n], false);
            t.fft(&mut self.a[..n], true);
            n *= 2;
        }
    }
    pub(crate) fn new(max_input: usize, max_output: usize) -> Self {
        let n = (max_input + max_output - 1).next_power_of_two();
        Self {
            a: vec![Complex::default(); n],
            b: vec![Complex::default(); n],
        }
    }
    pub(crate) fn process(
        &mut self,
        t: &mut Transforms,
        x: &[Complex],
        m: usize,
        omega: f64,
    ) -> &[Complex] {
        let nfft = (x.len() + m - 1).next_power_of_two();
        let a = &mut self.a[..nfft];
        let b = &mut self.b[..nfft];
        a.fill(Complex::default());
        b.fill(Complex::default());
        let chirp = |j: usize| Complex::from_polar(1.0, -0.5 * omega * (j as f64).powi(2));
        for (j, v) in x.iter().enumerate() {
            a[j] = v * chirp(j);
        }
        for (j, v) in b[..m].iter_mut().enumerate() {
            *v = chirp(j).conj();
        }
        for j in 1..x.len() {
            b[nfft - j] = chirp(j).conj();
        }
        t.fft(a, false);
        t.fft(b, false);
        for (a, b) in a.iter_mut().zip(b) {
            *a *= *b;
        }
        t.fft(a, true);
        for (k, v) in a[..m].iter_mut().enumerate() {
            *v *= chirp(k);
        }
        &self.a[..m]
    }
}

/// Keeps transform plans and complex scratch storage across frames.
pub(crate) struct Transforms {
    fft: FftPlanner<f64>,
    real: RealFftPlanner<f64>,
    #[cfg(test)]
    dct: DctPlanner<f64>,
    scratch: Vec<Complex>,
}

impl Default for Transforms {
    fn default() -> Self {
        Self {
            fft: FftPlanner::new(),
            real: RealFftPlanner::new(),
            #[cfg(test)]
            dct: DctPlanner::new(),
            scratch: Vec::new(),
        }
    }
}

impl Transforms {
    /// Forward is unscaled; inverse is divided by N.
    pub(crate) fn fft(&mut self, x: &mut [Complex], inverse: bool) {
        let plan = self.fft.plan_fft(
            x.len(),
            if inverse {
                rustfft::FftDirection::Inverse
            } else {
                rustfft::FftDirection::Forward
            },
        );
        self.scratch
            .resize(plan.get_inplace_scratch_len(), Complex::default());
        plan.process_with_scratch(x, &mut self.scratch);
        if inverse {
            let n = x.len() as f64;
            for v in x {
                *v /= n;
            }
        }
    }

    pub(crate) fn rfft(&mut self, x: &mut [f64]) -> Vec<Complex> {
        let mut out = vec![Complex::default(); x.len() / 2 + 1];
        self.rfft_into(x, &mut out);
        out
    }

    /// `out` has x.len()/2+1 elements; the caller can reuse it across frames.
    pub(crate) fn rfft_into(&mut self, x: &mut [f64], out: &mut [Complex]) {
        let plan = self.real.plan_fft_forward(x.len());
        self.scratch
            .resize(plan.get_scratch_len(), Complex::default());
        plan.process_with_scratch(x, out, &mut self.scratch)
            .expect("RFFT buffers match plan");
    }

    /// `x` is a Hermitian half spectrum, with real DC and (for even N) Nyquist.
    pub(crate) fn irfft(&mut self, x: &mut [Complex], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n];
        self.irfft_into(x, &mut out);
        out
    }
    pub(crate) fn irfft_into(&mut self, x: &mut [Complex], out: &mut [f64]) {
        let n = out.len();
        let plan = self.real.plan_fft_inverse(n);
        self.scratch
            .resize(plan.get_scratch_len(), Complex::default());
        plan.process_with_scratch(x, out, &mut self.scratch)
            .expect("valid Hermitian half spectrum");
        out.iter_mut().for_each(|v| *v /= n as f64);
    }

    /// Bluestein convolution: sum_j x[j] exp(-i * omega * j * k), k=0..m.
    /// The generalized output length avoids truncating harmonics when m > N.
    pub(crate) fn czt(&mut self, x: &[Complex], m: usize, omega: f64) -> Vec<Complex> {
        if m == 0 || x.is_empty() {
            return vec![Complex::default(); m];
        }
        let nfft = (x.len() + m - 1).next_power_of_two();
        let chirp = |j: usize| Complex::from_polar(1.0, -0.5 * omega * (j as f64).powi(2));
        let mut a = vec![Complex::default(); nfft];
        let mut b = a.clone();
        for (j, v) in x.iter().enumerate() {
            a[j] = v * chirp(j);
        }
        for (j, v) in b[..m].iter_mut().enumerate() {
            *v = chirp(j).conj();
        }
        for j in 1..x.len() {
            b[nfft - j] = chirp(j).conj();
        }
        self.fft(&mut a, false);
        self.fft(&mut b, false);
        for (a, b) in a.iter_mut().zip(b) {
            *a *= b;
        }
        self.fft(&mut a, true);
        a.truncate(m);
        for (k, v) in a.iter_mut().enumerate() {
            *v *= chirp(k);
        }
        a
    }

    /// Sign-reversed CZT divided by input length. It is a true inverse only on
    /// the DFT grid; elsewhere this is the reference library's ICZT convention.
    #[cfg(test)]
    pub(crate) fn iczt(&mut self, x: &[Complex], m: usize, omega: f64) -> Vec<Complex> {
        let mut y = self.czt(x, m, -omega);
        for v in &mut y {
            *v /= x.len() as f64;
        }
        y
    }

    /// Ooura ddct(N,-1): unnormalized DCT-II, sum x[n] cos(pi*k*(n+.5)/N).
    #[cfg(test)]
    pub(crate) fn dct2(&mut self, x: &mut [f64]) {
        self.dct.plan_dct2(x.len()).process_dct2(x);
    }
    /// Ooura ddct(N,+1): DCT-III with full weight on input[0].
    #[cfg(test)]
    pub(crate) fn dct3(&mut self, x: &mut [f64]) {
        x[0] *= 2.0;
        self.dct.plan_dct3(x.len()).process_dct3(x);
    }
    #[cfg(test)]
    pub(crate) fn dct(&mut self, x: &mut [f64]) {
        self.dct2(x);
        let scale = (2.0 / x.len() as f64).sqrt();
        for v in x.iter_mut() {
            *v *= scale;
        }
        x[0] /= 2.0_f64.sqrt();
    }
    #[cfg(test)]
    pub(crate) fn idct(&mut self, x: &mut [f64]) {
        let scale = (2.0 / x.len() as f64).sqrt();
        x[0] *= 2.0_f64.sqrt();
        self.dct.plan_dct3(x.len()).process_dct3(x);
        for v in x {
            *v *= scale;
        }
    }

    /// Minimum-phase response from natural-log amplitudes on an even FFT grid.
    pub(crate) fn minphase(&mut self, log_amplitude: &[f64]) -> Vec<f64> {
        let n = 2 * (log_amplitude.len() - 1);
        let mut c: Vec<_> = (0..n)
            .map(|i| Complex::new(log_amplitude[i.min(n - i)], 0.0))
            .collect();
        self.fft(&mut c, true);
        for v in &mut c[1..n / 2] {
            *v *= 2.0;
        }
        c[n / 2 + 1..].fill(Complex::default());
        self.fft(&mut c, false);
        c[..=n / 2].iter().map(|v| v.im).collect()
    }
}
