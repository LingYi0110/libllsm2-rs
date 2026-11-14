//! Layer 0 harmonic/residual analysis and offline overlap-add synthesis.
use crate::config::{MIN_VOICED_F0, valid_f0};
use crate::constants::{
    EULER_GAMMA, LOG_CHI2_VARIANCE, LOG_RESIDUAL_BIAS, db_to_log, log_to_power_db,
};
use crate::dsp::{
    f0::refine_f0,
    fft::{Complex, Transforms},
    filter::{kalman_filter, rts_smooth},
    interp::interp1u,
    noise::{SeededRng, bandlimited_noise},
    sinusoid,
    spectral::{self, HarmonicAnalysis, HarmonicMethod},
    window::{Window, fetch_frame, window},
};
use crate::{
    AnalysisOptions, Chunk, HarmonicAnalysisMethod, HarmonicFrame, LlsmError, PbpEffect,
    SynthesisMode, SynthesisOptions,
};
use std::f64::consts::TAU;

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub chunk: Chunk,
    /// Refined F0 (Hz); the caller's input contour is never modified.
    pub f0: Vec<f64>,
    /// Input minus the center-aligned harmonic reconstruction, in input samples.
    pub aperiodic_residual: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SynthesisOutput {
    pub fs: f64,
    pub y: Vec<f64>,
    pub y_sin: Vec<f64>,
    pub y_noise: Vec<f64>,
}

/// Analyze mono samples with supplied F0: 0 (unvoiced), or a finite voiced
/// frequency in [MIN_VOICED_F0, input Nyquist], in Hz.
/// The contour contains floor(N/hop) or ceil(N/hop) frames, covering the signal
/// on a grid beginning at sample zero. A partial final hop is optional.
pub fn analyze(
    options: &AnalysisOptions,
    x: &[f64],
    fs: f64,
    f0: &[f64],
) -> Result<AnalysisResult, LlsmError> {
    let config = options.to_config(fs)?;
    if x.is_empty() || x.iter().any(|v| !v.is_finite()) {
        return Err(LlsmError::InvalidInput {
            field: "x",
            reason: "must be nonempty and finite",
        });
    }
    let frames = x.len() as f64 / (fs * options.thop);
    let minimum = (frames.floor() as usize).max(1);
    let maximum = (frames.ceil() as usize).max(1);
    if f0.len() < minimum || f0.len() > maximum {
        return Err(LlsmError::InvalidF0Length {
            minimum,
            maximum,
            actual: f0.len(),
        });
    }
    if f0.iter().any(|&v| !valid_f0(v, fs / 2.0)) {
        return Err(LlsmError::InvalidInput {
            field: "f0",
            reason: "must be zero or finite in [10 Hz, Nyquist]",
        });
    }
    let f0 = if options.refine_f0 {
        refine_f0(x, fs, f0, options.thop, MIN_VOICED_F0)
    } else {
        f0.to_vec()
    };
    let mut chunk = Chunk::new(config, f0.len());
    let mut transforms = Transforms::default();
    let harmonic_options = harmonic_options(options, fs, options.max_nhar);
    let harmonics = spectral::harmonic_analysis(&mut transforms, x, &f0, &harmonic_options);
    for ((frame, &f0), hm) in chunk.frames.iter_mut().zip(&f0).zip(harmonics) {
        frame.f0 = f0;
        frame.hm = hm.map(|h| HarmonicFrame {
            ampl: h.amplitude,
            phse: h.phase,
        });
    }
    let mut synthesis = SynthesisOptions::new(fs);
    synthesis.iczt_threshold = None;
    let sinusoid = synthesize_harmonics(&mut transforms, &synthesis, &chunk, x.len());
    let residual: Vec<_> = x.iter().zip(sinusoid).map(|(x, y)| x - y).collect();
    analyze_noise_psd(&mut transforms, options, x, &residual, fs, &mut chunk);
    analyze_noise_envelopes(&mut transforms, options, x, &residual, fs, &f0, &mut chunk);
    Ok(AnalysisResult {
        chunk,
        f0,
        aperiodic_residual: residual,
    })
}

fn harmonic_options(options: &AnalysisOptions, fs: f64, maximum: usize) -> HarmonicAnalysis {
    HarmonicAnalysis {
        fs,
        thop: options.thop,
        relative_window: options.rel_winsize,
        max_harmonics: maximum,
        method: match options.hm_method {
            HarmonicAnalysisMethod::PeakPicking => HarmonicMethod::PeakPicking,
            HarmonicAnalysisMethod::ChirpZ => HarmonicMethod::ChirpZ,
        },
    }
}

/// Offline synthesis in the selected layer. Output length is round((frames+1)*thop*fs),
/// including the final overlap-add tail, as in libllsm2 v2.1.
pub fn synthesize(options: &SynthesisOptions, chunk: &Chunk) -> Result<SynthesisOutput, LlsmError> {
    synthesize_impl(options, chunk, None)
}

/// Layer 1 synthesis with a stateful effect on PbP pulses. Only frames with
/// pbp_enabled (and their outgoing crossfades) invoke the callback.
///
/// ```
/// use llsm2::{analyze, synthesize_with_effect, AnalysisOptions, SynthesisOptions,
///     SynthesisMode, PhaseDirection, GlottalFlowParams, PulseContext, LlsmError};
/// # let fs = 44100.0;
/// # let x: Vec<_> = (0..4410).map(|i| 0.1*(std::f64::consts::TAU*200.0*i as f64/fs).cos()).collect();
/// # let a = AnalysisOptions { refine_f0: false, ..AnalysisOptions::default() };
/// # let mut chunk = analyze(&a, &x, fs, &[200.0;20])?.chunk;
/// chunk.to_layer1(2048)?;
/// chunk.phase_sync_rps(true);
/// chunk.clear_layer0();
/// for frame in &mut chunk.frames { frame.pbp_enabled = true; }
/// chunk.phase_propagate(PhaseDirection::Forward);
/// let options = SynthesisOptions { mode: SynthesisMode::Layer1, ..SynthesisOptions::new(fs) };
/// let mut effect = |g: &mut GlottalFlowParams, _: &PulseContext<'_>| {
///     g.ee *= 0.9;
///     0.0 // pulse time offset, seconds
/// };
/// let output = synthesize_with_effect(&options, &chunk, &mut effect)?;
/// # assert!(output.y.iter().all(|v| v.is_finite()));
/// # Ok::<(), LlsmError>(())
/// ```
pub fn synthesize_with_effect(
    options: &SynthesisOptions,
    chunk: &Chunk,
    effect: &mut dyn PbpEffect,
) -> Result<SynthesisOutput, LlsmError> {
    if options.mode != SynthesisMode::Layer1 {
        return Err(LlsmError::InvalidInput {
            field: "mode",
            reason: "pulse effects require Layer 1 synthesis",
        });
    }
    synthesize_impl(options, chunk, Some(effect))
}

fn synthesize_impl(
    options: &SynthesisOptions,
    chunk: &Chunk,
    effect: Option<&mut dyn PbpEffect>,
) -> Result<SynthesisOutput, LlsmError> {
    options.validate(chunk.config.thop)?;
    match options.mode {
        SynthesisMode::Layer0 => chunk.validate_layer0()?,
        SynthesisMode::Layer1 => chunk.validate_layer1()?,
    };
    let length =
        ((chunk.frames.len() + 1) as f64 * chunk.config.thop * options.fs).round() as usize;
    let mut transforms = Transforms::default();
    let y_sin = match options.mode {
        SynthesisMode::Layer0 => synthesize_harmonics(&mut transforms, options, chunk, length),
        SynthesisMode::Layer1 => {
            crate::layer1::synthesize_periodic(&mut transforms, options, chunk, length, effect)?
        }
    };
    let excitation = synthesize_noise_excitation(&mut transforms, options, chunk, length);
    let y_noise = filter_noise(&mut transforms, chunk, options.fs, &excitation);
    let y = y_sin.iter().zip(&y_noise).map(|(h, n)| h + n).collect();
    Ok(SynthesisOutput {
        fs: options.fs,
        y,
        y_sin,
        y_noise,
    })
}

pub(crate) fn overlap_add(destination: &mut [f64], signal: &[f64], start: isize) {
    for (i, &v) in signal.iter().enumerate() {
        let index = start + i as isize;
        if index >= 0 && (index as usize) < destination.len() {
            destination[index as usize] += v;
        }
    }
}

fn synthesize_harmonics(
    t: &mut Transforms,
    options: &SynthesisOptions,
    chunk: &Chunk,
    length: usize,
) -> Vec<f64> {
    let hop = chunk.config.thop * options.fs;
    let nwin = hop.round() as usize * 2;
    let w = window(Window::Hann, nwin);
    let mut y = vec![0.0; length];
    for (i, frame) in chunk.frames.iter().enumerate() {
        if !frame.is_voiced() {
            continue;
        }
        let hm = frame
            .hm
            .as_ref()
            .expect("voiced Layer 0 frames were validated");
        let nhar = hm
            .num_harmonics()
            .min((options.fs / (2.0 * frame.f0)).floor() as usize);
        let raw = i as f64 * hop;
        let center = raw.round() as isize;
        let correction = (raw - center as f64) * TAU / options.fs * frame.f0;
        let phase: Vec<_> = hm.phse[..nhar]
            .iter()
            .enumerate()
            .map(|(k, p)| p - correction * (k + 1) as f64)
            .collect();
        let mut signal = sinusoid::synthesize(
            t,
            &hm.ampl[..nhar],
            &phase,
            frame.f0 / options.fs,
            nwin,
            options.iczt_threshold,
        );
        for (v, w) in signal.iter_mut().zip(&w) {
            *v *= w;
        }
        overlap_add(&mut y, &signal, center - (nwin / 2) as isize);
    }
    y
}

fn analyze_noise_psd(
    t: &mut Transforms,
    options: &AnalysisOptions,
    x: &[f64],
    residual: &[f64],
    fs: f64,
    chunk: &mut Chunk,
) {
    let nframes = chunk.num_frames();
    let nwin = (options.thop * 4.0 * fs).round() as usize;
    let nfft = nwin.next_power_of_two();
    let nspec = nfft / 2 + 1;
    let centers: Vec<_> = (0..nframes)
        .map(|i| (i as f64 * options.thop * fs).round() as isize)
        .collect();
    let sizes: Vec<_> = chunk
        .frames
        .iter()
        .map(|f| {
            if f.is_voiced() {
                (fs / f.f0 * 3.0) as usize
            } else {
                nwin
            }
        })
        .collect();
    // Low F0 can require more than the C implementation's fixed 30 ms FFT.
    let envelope_fft = ((0.03 * fs) as usize)
        .max(*sizes.iter().max().unwrap())
        .next_power_of_two();
    let spectra = spectral::spectrogram(t, x, &centers, &sizes, envelope_fft, Window::Hann);
    let envelopes: Vec<Vec<f64>> = spectra
        .iter()
        .zip(&chunk.frames)
        .map(|(s, f)| {
            let f0 = if f.is_voiced() {
                f.f0
            } else {
                200.0_f64.min(fs / 2.0)
            } / fs;
            // A full-spectrum envelope keeps the upper tail (no harmonic cutoff).
            let nhar = (0.5 / f0).ceil() as usize + 3;
            let env = spectral::spec2env(t, &s.amplitude, f0, nhar);
            (0..nspec)
                .map(|j| env[j * envelope_fft / nfft] * 2.0)
                .collect()
        })
        .collect();
    let mut psd: Vec<Vec<f64>> = centers
        .iter()
        .map(|&center| {
            spectral::estimate_psd(t, &fetch_frame(residual, center, nwin), nfft)
                .iter()
                .map(|v| v.max(1e-10).ln())
                .collect()
        })
        .collect();
    let mut psd_residual = vec![vec![0.0; nspec]; nframes];
    let r = vec![LOG_CHI2_VARIANCE; nframes];
    for j in 0..nspec {
        let q: Vec<_> = (0..nframes)
            .map(|i| {
                let local = [
                    envelopes[i.saturating_sub(1)][j],
                    envelopes[i][j],
                    envelopes[(i + 1).min(nframes - 1)][j],
                ];
                let mean = local.iter().sum::<f64>() / 3.0;
                (local.iter().map(|v| v * v).sum::<f64>() / 3.0 - mean * mean).max(1e-8)
            })
            .collect();
        let observations: Vec<_> = psd.iter().map(|row| row[j]).collect();
        let forward = kalman_filter(&observations, &q, &r, observations[0]);
        let stationary = rts_smooth(&forward.values, &forward.variances, &q);
        for i in 0..nframes {
            psd_residual[i][j] = psd[i][j] - stationary[i];
            psd[i][j] = stationary[i] + EULER_GAMMA;
        }
    }
    let axis: Vec<_> = (0..options.npsd)
        .map(|i| i as f64 / (options.npsd - 1) as f64 * fs / 2.0)
        .collect();
    for (i, frame) in chunk.frames.iter_mut().enumerate() {
        let stationary = interp1u(0.0, fs / 2.0, &psd[i], &axis);
        let res = interp1u(0.0, fs / 2.0, &psd_residual[i], &axis);
        frame.nm.psd = stationary
            .iter()
            .map(|v| 10.0 * (v.exp() * 44100.0 / fs + 1e-12).log10())
            .collect();
        frame.nm.psd_residual = Some(res.iter().map(|&v| log_to_power_db(v)).collect());
    }
}

fn analyze_noise_envelopes(
    t: &mut Transforms,
    options: &AnalysisOptions,
    x: &[f64],
    residual: &[f64],
    fs: f64,
    f0: &[f64],
    chunk: &mut Chunk,
) {
    let centers: Vec<_> = (0..f0.len())
        .map(|i| (i as f64 * options.thop * fs).round() as isize)
        .collect();
    let sizes: Vec<_> = f0
        .iter()
        .map(|&f| {
            ((if f == 0.0 {
                options.thop * 2.0
            } else {
                2.0 / f
            }) * fs)
                .round() as usize
        })
        .collect();
    let harmonic_options = harmonic_options(options, fs, options.max_nhar_env);
    for channel in 0..chunk.config.num_channels() {
        let low = if channel == 0 {
            0.0
        } else {
            options.chan_freqs[channel - 1]
        };
        let high = options.chan_freqs.get(channel).copied().unwrap_or(fs / 2.0);
        let energy =
            spectral::subband_energy(if low > 6000.0 { x } else { residual }, low / fs, high / fs);
        let harmonics = spectral::harmonic_analysis(t, &energy, f0, &harmonic_options);
        let dc = spectral::compute_dc(&energy, &centers, &sizes);
        for ((frame, hm), dc) in chunk.frames.iter_mut().zip(harmonics).zip(dc) {
            frame.nm.edc[channel] = dc;
            frame.nm.eenv[channel] = hm
                .map(|h| HarmonicFrame {
                    ampl: h.amplitude,
                    phse: h.phase,
                })
                .unwrap_or_default();
        }
    }
}

fn synthesize_noise_excitation(
    t: &mut Transforms,
    options: &SynthesisOptions,
    chunk: &Chunk,
    length: usize,
) -> Vec<f64> {
    let mut y = vec![0.0; length];
    let hop = chunk.config.thop * options.fs;
    let nwin = (2.0 * hop).round() as usize;
    let w = window(Window::Hann, nwin);
    let mut rng = SeededRng::new(options.random_seed);
    let nyquist = chunk.config.fnyq.min(options.fs / 2.0);
    for channel in 0..chunk.config.num_channels() {
        let low = if channel == 0 {
            0.0
        } else {
            chunk.config.chan_freqs[channel - 1]
        };
        if low >= nyquist {
            break;
        }
        let high = chunk
            .config
            .chan_freqs
            .get(channel)
            .copied()
            .unwrap_or(nyquist)
            .min(nyquist);
        let noise = bandlimited_noise(&mut rng, length, low / options.fs, high / options.fs);
        let mut envelope = vec![0.0; length];
        for (i, frame) in chunk.frames.iter().enumerate() {
            let hm = &frame.nm.eenv[channel];
            let nhar = if frame.is_voiced() {
                hm.num_harmonics()
                    .min((options.fs / (2.0 * frame.f0)).floor() as usize)
            } else {
                0
            };
            let mut signal = sinusoid::synthesize(
                t,
                &hm.ampl[..nhar],
                &hm.phse[..nhar],
                frame.f0 / options.fs,
                nwin,
                options.iczt_threshold,
            );
            for (v, w) in signal.iter_mut().zip(&w) {
                *v = (*v + frame.nm.edc[channel]).max(1e-8) * w;
            }
            overlap_add(
                &mut envelope,
                &signal,
                ((i as f64 - 1.0) * hop).round() as isize,
            );
        }
        for ((y, n), e) in y.iter_mut().zip(noise).zip(envelope) {
            *y += n * e.sqrt();
        }
    }
    y
}

/// Shared STFT equalizer, with fixed frequency grids and reusable FFT input.
pub(crate) struct NoiseFilter {
    pub(crate) nfft: usize,
    pub(crate) window: Vec<f64>,
    window_power: f64,
    fs: f64,
    source_nyquist: f64,
    src_axis: Vec<f64>,
    dst_axis: Vec<f64>,
    supported_bins: usize,
    input: Vec<f64>,
    spectrum: Vec<Complex>,
    filtered: Vec<f64>,
    envelope: Vec<f64>,
    prefix: Vec<f64>,
    target: Vec<f64>,
    gains: Vec<f64>,
}
impl NoiseFilter {
    pub(crate) fn new(conf: &crate::ModelConfig, fs: f64) -> Self {
        let nwin = (2.0 * conf.thop * fs).round() as usize;
        let window = window(Window::Hann, nwin);
        let window_power = window.iter().map(|v| v * v).sum();
        let nfft = ((nwin as f64 * 1.2 + 32.0).ceil() as usize).next_power_of_two();
        let src_axis = (0..conf.npsd)
            .map(|i| i as f64 / (conf.npsd - 1) as f64 * conf.fnyq)
            .collect();
        let dst_axis: Vec<_> = (0..nfft / 2).map(|i| i as f64 / nfft as f64 * fs).collect();
        let supported_bins = dst_axis.partition_point(|&f| f <= conf.fnyq);
        Self {
            nfft,
            window,
            window_power,
            fs,
            source_nyquist: conf.fnyq,
            src_axis,
            dst_axis,
            supported_bins,
            input: vec![0.0; nfft],
            spectrum: vec![Complex::default(); nfft / 2 + 1],
            filtered: vec![0.0; nfft],
            envelope: vec![0.0; nfft / 2 + 1],
            prefix: vec![0.0; nfft / 2 + 7],
            target: vec![0.0; conf.npsd],
            gains: vec![0.0; supported_bins],
        }
    }
    pub(crate) fn process(
        &mut self,
        t: &mut Transforms,
        psd: &[f64],
        residual: Option<&[f64]>,
        signal: &[f64],
    ) -> Option<&[f64]> {
        if psd.iter().copied().fold(f64::NEG_INFINITY, f64::max) < -100.0 {
            return None;
        }
        self.input.fill(0.0);
        let nwin = self.window.len();
        for (j, (&v, &w)) in signal.iter().zip(&self.window).enumerate() {
            self.input[j + self.nfft / 2 - nwin / 2] = v * w;
        }
        t.rfft_into(&mut self.input, &mut self.spectrum);
        let spectrum = &mut self.spectrum;
        for (p, v) in self.envelope.iter_mut().zip(&*spectrum) {
            *p = v.norm_sqr() / self.window_power;
        }
        // Integer half-width 3 of ciglet's cumulative moving average: six
        // samples from i-2 through i+3, with held endpoints.
        let mut sum = 0.0;
        let last = self.envelope.len() - 1;
        for (j, p) in self.prefix.iter_mut().enumerate() {
            sum += self.envelope[j.saturating_sub(3).min(last)];
            *p = sum;
        }
        for (j, p) in self.envelope.iter_mut().enumerate() {
            *p = (self.prefix[j + 6] - self.prefix[j]) / 6.0;
        }
        for (j, target) in self.target.iter_mut().enumerate() {
            *target = psd[j] + residual.map_or(0.0, |r| r[j] - log_to_power_db(LOG_RESIDUAL_BIAS));
        }
        crate::dsp::interp::interp1_into(
            &self.src_axis,
            &self.target,
            &self.dst_axis[..self.supported_bins],
            &mut self.gains,
        );
        for ((value, &gain), &envelope) in spectrum.iter_mut().zip(&self.gains).zip(&self.envelope)
        {
            *value *= db_to_log(gain).exp() / (envelope * 44100.0 / self.fs + 1e-8).sqrt();
        }
        spectrum[self.supported_bins..self.nfft / 2].fill(Complex::default());
        spectrum[0].im = 0.0;
        spectrum[self.nfft / 2] = if self.fs / 2.0 > self.source_nyquist {
            Complex::default()
        } else {
            Complex::new(spectrum[self.nfft / 2 - 1].re, 0.0)
        };
        t.irfft_into(spectrum, &mut self.filtered);
        let filtered = &mut self.filtered;
        for j in 0..16 {
            let gain = j as f64 / 16.0;
            filtered[j] *= gain;
            filtered[self.nfft - 1 - j] *= gain;
        }
        Some(&self.filtered)
    }
    pub(crate) fn warm(&mut self, t: &mut Transforms) {
        t.rfft_into(&mut self.input, &mut self.spectrum);
        t.irfft_into(&mut self.spectrum, &mut self.filtered);
    }
}

fn filter_noise(t: &mut Transforms, chunk: &Chunk, fs: f64, excitation: &[f64]) -> Vec<f64> {
    let mut filter = NoiseFilter::new(&chunk.config, fs);
    let hop = chunk.config.thop * fs;
    let mut y = vec![0.0; excitation.len()];
    for (i, frame) in chunk.frames.iter().enumerate() {
        let center = (i as f64 * hop).round() as isize;
        let signal = fetch_frame(excitation, center, filter.window.len());
        let nfft = filter.nfft;
        if let Some(filtered) =
            filter.process(t, &frame.nm.psd, frame.nm.psd_residual.as_deref(), &signal)
        {
            overlap_add(&mut y, filtered, center - (nfft / 2) as isize);
        }
    }
    y
}
