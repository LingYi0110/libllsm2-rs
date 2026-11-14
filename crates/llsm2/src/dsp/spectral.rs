use super::{
    fft::{Complex, Transforms},
    filter::{bandpass, moving_average},
    interp::{circular, interp1u, qifft},
    window::{Window, fetch_frame, fill_window, window},
};
use crate::constants::log_to_db;
use std::f64::consts::{PI, TAU};

pub(crate) struct Spectrum {
    pub(crate) amplitude: Vec<f64>,
    pub(crate) phase: Vec<f64>,
}

/// One STFT grid and window kind, with reusable per-frame working buffers.
struct StftWorkspace {
    kind: Window,
    input: Vec<f64>,
    spectrum: Vec<Complex>,
    window: Vec<f64>,
    scale: f64,
}

impl StftWorkspace {
    fn new(nfft: usize, kind: Window) -> Self {
        Self {
            kind,
            input: vec![0.0; nfft],
            spectrum: vec![Complex::default(); nfft / 2 + 1],
            window: Vec::new(),
            scale: 0.0,
        }
    }

    /// The nonempty window fits inside the FFT grid.
    fn analyze(&mut self, t: &mut Transforms, x: &[f64], center: isize, size: usize) -> Spectrum {
        if self.window.len() != size {
            self.window.resize(size, 0.0);
            fill_window(self.kind, &mut self.window);
            self.scale = 2.0 / self.window.iter().sum::<f64>();
        }
        self.input.fill(0.0);
        let nfft = self.input.len();
        for (i, &w) in self.window.iter().enumerate() {
            let source = center + i as isize - (size / 2) as isize;
            if source >= 0 && (source as usize) < x.len() {
                self.input[(i + nfft - size / 2) % nfft] = x[source as usize] * w;
            }
        }
        t.rfft_into(&mut self.input, &mut self.spectrum);
        Spectrum {
            amplitude: self
                .spectrum
                .iter()
                .map(|v| v.norm() * self.scale)
                .collect(),
            phase: self.spectrum.iter().map(|v| v.arg()).collect(),
        }
    }
}

/// Centered, zero-padded STFT; magnitude normalized to sinusoidal amplitude.
pub(crate) fn spectrogram(
    t: &mut Transforms,
    x: &[f64],
    centers: &[isize],
    sizes: &[usize],
    nfft: usize,
    kind: Window,
) -> Vec<Spectrum> {
    let mut workspace = StftWorkspace::new(nfft, kind);
    centers
        .iter()
        .zip(sizes)
        .map(|(&center, &size)| workspace.analyze(t, x, center, size))
        .collect()
}

pub(crate) fn compute_dc(x: &[f64], centers: &[isize], sizes: &[usize]) -> Vec<f64> {
    centers
        .iter()
        .zip(sizes)
        .map(|(&center, &size)| fetch_frame(x, center, size).iter().sum::<f64>() / size as f64)
        .collect()
}

/// Two-sided PSD values on the nonnegative-frequency half grid; no doubling.
/// Unit white noise has mean PSD 1, independent of window/FFT length.
pub(crate) fn fft_to_psd(x: &[Complex], window_power: f64) -> Vec<f64> {
    x.iter().map(|v| v.norm_sqr() / window_power).collect()
}
pub(crate) fn estimate_psd(t: &mut Transforms, x: &[f64], nfft: usize) -> Vec<f64> {
    let w = window(Window::Blackman, x.len());
    let power = w.iter().map(|v| v * v).sum();
    let mut input = vec![0.0; nfft];
    for (dst, (&v, &w)) in input.iter_mut().zip(x.iter().zip(&w)) {
        *dst = v * w;
    }
    fft_to_psd(&t.rfft(&mut input), power)
}

pub(crate) fn subband_energy(x: &[f64], low: f64, high: f64) -> Vec<f64> {
    bandpass(x, low, high).iter().map(|v| v * v).collect()
}

pub(crate) fn harmonic_czt(
    t: &mut Transforms,
    x: &[f64],
    f0: f64,
    fs: f64,
    nhar: usize,
) -> Spectrum {
    let w = window(Window::Blackman, x.len());
    let scale = 2.0 / w.iter().sum::<f64>();
    let input: Vec<_> = x
        .iter()
        .zip(&w)
        .map(|(&x, &w)| Complex::new(x * w, 0.0))
        .collect();
    let spectrum = t.czt(&input, nhar + 1, TAU * f0 / fs);
    let values: Vec<_> = spectrum
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, v)| {
            v * Complex::from_polar(scale, TAU * f0 / fs * i as f64 * (x.len() / 2) as f64)
        })
        .collect();
    Spectrum {
        amplitude: values.iter().map(|v| v.norm()).collect(),
        phase: values.iter().map(|v| v.arg()).collect(),
    }
}

/// Requires matching half-spectrum arrays on an even FFT grid with nfft >= 4.
pub(crate) fn harmonic_peak_picking(
    spectrum: &Spectrum,
    fs: f64,
    f0: f64,
    nhar: usize,
) -> Spectrum {
    let nfft = (spectrum.amplitude.len() - 1) * 2;
    let logs: Vec<_> = spectrum.amplitude.iter().map(|v| (v + 1e-8).ln()).collect();
    let mut out = Spectrum {
        amplitude: Vec::with_capacity(nhar),
        phase: Vec::with_capacity(nhar),
    };
    for i in 1..=nhar {
        let low =
            ((f0 * (i as f64 - 0.3) / fs * nfft as f64).round() as usize).clamp(1, nfft / 2 - 1);
        let high =
            ((f0 * (i as f64 + 0.3) / fs * nfft as f64).round() as usize).clamp(low, nfft / 2 - 1);
        // Match cig_find_peak: strongest local peak exceeding the lower-bin
        // value, or the lower bin if there is no such local peak.
        let mut peak = low;
        for k in low..=high {
            if logs[k] > logs[k - 1] && logs[k] > logs[k + 1] && logs[k] > logs[peak] {
                peak = k;
            }
        }
        let (index, magnitude) = qifft(&logs, peak);
        let k = index as usize;
        out.amplitude.push(magnitude.exp());
        // Circular interpolation intentionally fixes C's +/-pi branch-cut
        // error. QIFFT bounds the offset to (-1,1), keeping k+1 in this grid.
        out.phase.push(circular(
            spectrum.phase[k],
            spectrum.phase[k + 1],
            index - k as f64,
        ));
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum HarmonicMethod {
    PeakPicking,
    ChirpZ,
}

pub(crate) struct HarmonicAnalysis {
    pub(crate) fs: f64,
    pub(crate) thop: f64,
    pub(crate) relative_window: f64,
    pub(crate) max_harmonics: usize,
    pub(crate) method: HarmonicMethod,
}

pub(crate) fn harmonic_analysis(
    t: &mut Transforms,
    x: &[f64],
    f0: &[f64],
    options: &HarmonicAnalysis,
) -> Vec<Option<Spectrum>> {
    let min_f0 = f0
        .iter()
        .copied()
        .filter(|&f| f > 0.0)
        .fold(1000.0, f64::min);
    let nfft = ((options.fs / min_f0 * options.relative_window / 2.0).round() as usize * 2)
        .next_power_of_two()
        .max(4); // QIFFT needs three half-spectrum bins.
    let mut stft = None;
    f0.iter()
        .enumerate()
        .map(|(i, &f)| {
            if f == 0.0 {
                return None;
            }
            let size = (options.fs / f * options.relative_window / 2.0).round() as usize * 2;
            let center = (i as f64 * options.thop * options.fs).round() as isize;
            let nhar = options
                .max_harmonics
                .min((options.fs / f / 2.0).floor() as usize);
            Some(match options.method {
                HarmonicMethod::ChirpZ => {
                    harmonic_czt(t, &fetch_frame(x, center, size), f, options.fs, nhar)
                }
                HarmonicMethod::PeakPicking => {
                    let spectrum = stft
                        .get_or_insert_with(|| StftWorkspace::new(nfft, Window::Blackman))
                        .analyze(t, x, center, size);
                    harmonic_peak_picking(&spectrum, options.fs, f, nhar)
                }
            })
        })
        .collect()
}

fn aliased_sinc(n: f64, omega: f64) -> f64 {
    let denominator = (omega / 2.0).sin();
    if denominator.abs() < 1e-12 {
        n
    } else {
        (n * omega / 2.0).sin() / denominator
    }
}

/// Three-period Hann-windowed zero-phase harmonic spectrum, f0 cycles/sample.
pub(crate) fn harmonic_spectrum(amplitude: &[f64], f0: f64, nfft: usize) -> Vec<f64> {
    let mut out: Vec<f64> = vec![0.0; nfft / 2 + 1];
    let period = (3.0 / f0).floor();
    let width = (f0 * nfft as f64 * 1.5).ceil() as usize;
    for (i, &ampl) in amplitude.iter().enumerate() {
        let freq = f0 * (i + 1) as f64;
        let center = (freq * nfft as f64).round() as usize;
        let start = center.saturating_sub(width).min(out.len());
        let end = (center + width + 1).min(out.len());
        for (j, value) in out.iter_mut().enumerate().take(end).skip(start) {
            let omega = (j as f64 / nfft as f64 - freq) * TAU;
            let response = 0.5 * aliased_sinc(period, omega)
                + 0.25 * aliased_sinc(period, omega - TAU / period)
                + 0.25 * aliased_sinc(period, omega + TAU / period);
            *value = value.max(response * ampl);
        }
    }
    for v in &mut out {
        *v *= f0;
    }
    out
}

/// CheapTrick smoothing and cepstral liftering. Returns natural-log amplitude.
/// Spectrum has nfft/2+1 bins and 0 < f0 <= 0.5. For fewer than three
/// harmonics the upper tail is held from the first smoothed bin.
pub(crate) fn spec2env(t: &mut Transforms, spectrum: &[f64], f0: f64, nhar: usize) -> Vec<f64> {
    let nfft = 2 * (spectrum.len() - 1);
    let mut v = spectrum.to_vec();
    let kf0 = (f0 * nfft as f64).ceil() as usize;
    for i in 0..kf0 / 2 {
        v[i] = v[kf0 - i];
    }
    let mut smoothed = moving_average(&v, nfft as f64 * f0 / 3.0);
    let top = (f0 * nhar.saturating_sub(2) as f64 * nfft as f64).floor() as usize;
    let top = top.clamp(1, smoothed.len());
    let edge = smoothed[top - 1];
    smoothed[top..].fill(edge);
    let mut c: Vec<_> = (0..nfft)
        .map(|i| Complex::new((smoothed[i.min(nfft - i)] + 1e-15).ln(), 0.0))
        .collect();
    t.fft(&mut c, true);
    for (i, value) in c.iter_mut().enumerate().take(nfft / 2 + 1).skip(1) {
        let q = i as f64 * f0 * PI;
        *value *= q.sin() / q * (1.18 - 0.18 * (2.0 * q).cos());
    }
    for i in nfft / 2 + 1..nfft {
        c[i] = c[nfft - i];
    }
    t.fft(&mut c, false);
    c[..=nfft / 2].iter().map(|v| v.re).collect()
}

pub(crate) fn harmonic_envelope(
    t: &mut Transforms,
    amplitude: &[f64],
    f0: f64,
    nfft: usize,
) -> Vec<f64> {
    let peak = amplitude
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(1e-12)
        .ln();
    let compressed: Vec<_> = amplitude
        .iter()
        .map(|v| {
            let log = v.max(1e-12).ln() - peak;
            (if log > -10.0 {
                log
            } else {
                (log + 10.0) / 2.0 - 10.0
            })
            .exp()
        })
        .collect();
    let spec = harmonic_spectrum(&compressed, f0, nfft);
    spec2env(t, &spec, f0, amplitude.len())
        .iter()
        .map(|&v| {
            log_to_db(
                (if v > -10.0 {
                    v
                } else {
                    (v + 10.0) * 2.0 - 10.0
                }) + peak,
            )
        })
        .collect()
}

pub(crate) fn harmonic_minphase(t: &mut Transforms, amplitude: &[f64]) -> Vec<f64> {
    if amplitude.is_empty() {
        return Vec::new();
    }
    let nfft = (amplitude.len() * 4).next_power_of_two().max(64);
    let mut logs: Vec<_> = amplitude.iter().map(|v| (v + 1e-10).ln()).collect();
    logs.insert(0, logs[0]);
    let axis: Vec<_> = (0..=nfft / 2).map(|i| i as f64).collect();
    let spectrum = interp1u(0.0, nfft as f64 / 2.0, &logs, &axis);
    let phase = t.minphase(&spectrum);
    let harmonics: Vec<_> = (1..=amplitude.len())
        .map(|i| i as f64 / (amplitude.len() + 1) as f64 * nfft as f64 / 2.0)
        .collect();
    interp1u(0.0, phase.len() as f64, &phase, &harmonics)
}

/// Lip radius in cm; harmonic frequencies in Hz.
pub(crate) fn lip_response(radius: f64, frequency: f64) -> Complex {
    let r = 128.0 / (9.0 * PI * PI);
    let l = 8.0 * radius / 100.0 / (3.0 * PI * 340.0);
    Complex::new(0.0, TAU * frequency * l * r) / Complex::new(r, TAU * frequency * l)
}
/// Harmonic grid: bin i is (i+1)*f0 Hz, with f0 > 0 and no DC bin.
/// Pass None for magnitude-only filtering, or a matching phase slice.
pub(crate) fn lipfilter(
    radius: f64,
    f0: f64,
    amplitude: &mut [f64],
    mut phase: Option<&mut [f64]>,
    inverse: bool,
) {
    for (i, a) in amplitude.iter_mut().enumerate() {
        let response = lip_response(radius, f0 * (i + 1) as f64);
        if inverse {
            *a /= response.norm();
        } else {
            *a *= response.norm();
        }
        if let Some(phase) = phase.as_deref_mut() {
            if inverse {
                phase[i] -= response.arg();
            } else {
                phase[i] += response.arg();
            }
        }
    }
}
/// Complex harmonic grid, identical to lipfilter: f0 > 0 and no DC bin.
/// An FFT spectrum starts at DC and must not be passed here unchanged;
/// radiation removes DC, so inverse radiation cannot recover its original value.
pub(crate) fn lipfilter_harmonics_complex(
    radius: f64,
    f0: f64,
    spectrum: &mut [Complex],
    inverse: bool,
) {
    for (i, v) in spectrum.iter_mut().enumerate() {
        let response = lip_response(radius, f0 * (i + 1) as f64);
        if inverse {
            *v /= response;
        } else {
            *v *= response;
        }
    }
}
