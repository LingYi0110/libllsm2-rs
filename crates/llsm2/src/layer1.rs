//! Source/tract decomposition, harmonic reconstruction and pulse-by-pulse synthesis.
use crate::constants::db_to_log;
use crate::dsp::{
    fft::{Complex, Transforms},
    filter::smoothing_filter,
    interp::{fill_blanks, interp1, wrap},
    lfmodel::{CachedGlottalModel, LfModel},
    spectral::{harmonic_envelope, harmonic_minphase, lipfilter, lipfilter_harmonics_complex},
};
use crate::{Chunk, Frame, HarmonicFrame, Layer1Data, LlsmError, ModelConfig, SynthesisOptions};
use std::f64::consts::{PI, TAU};

/// Intuitive LF parameters. Fa is Hz, T0 seconds, Rk/Rg dimensionless, Ee slope.
#[derive(Debug, Clone, Copy)]
pub struct GlottalFlowParams {
    pub fa: f64,
    pub rk: f64,
    pub rg: f64,
    pub t0: f64,
    pub ee: f64,
}
impl GlottalFlowParams {
    pub fn from_rd(rd: f64, t0: f64, ee: f64) -> Result<Self, LlsmError> {
        if !rd.is_finite() || !(0.02..=3.0).contains(&rd) {
            return Err(LlsmError::InvalidInput {
                field: "rd",
                reason: "must be in [0.02, 3.0]",
            });
        }
        for (field, value) in [("t0", t0), ("ee", ee)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(LlsmError::InvalidInput {
                    field,
                    reason: "must be finite and positive",
                });
            }
        }
        let params = Self::from_lf(LfModel::from_rd(rd, t0, ee));
        params.validate()?;
        Ok(params)
    }
    /// Require a positive LF opening/closing interval and a resolvable return
    /// phase; invalid effect parameters are reported rather than clamped.
    pub fn validate(&self) -> Result<(), LlsmError> {
        self.check().map_err(|reason| LlsmError::InvalidInput {
            field: "glottal_flow",
            reason,
        })
    }
    fn check(&self) -> Result<(), &'static str> {
        if [self.fa, self.rk, self.rg, self.t0, self.ee]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("LF parameters must be finite and positive");
        }
        let model = self.lf();
        // Near-instantaneous opening/return intervals make LF coefficients
        // numerically degenerate. Native Rd models have ta
        // >= 1e-6; these much smaller fractions are outside the supported shape.
        if model.tp < 1e-9 || model.ta < 1e-9 {
            return Err("LF opening and return times must span at least 1e-9 of a period");
        }
        if !(model.tp > 0.0
            && model.tp < model.te
            && model.te < 1.0
            && model.ta > 0.0
            && model.ta < 1.0 - model.te)
        {
            return Err("LF opening/return intervals must fit inside the period");
        }
        Ok(())
    }
    fn from_lf(lf: LfModel) -> Self {
        Self {
            fa: 1.0 / (lf.ta * lf.t0),
            rk: (lf.te - lf.tp) / lf.tp,
            rg: 0.5 / lf.tp,
            t0: lf.t0,
            ee: lf.ee,
        }
    }
    fn lf(self) -> LfModel {
        let tp = 0.5 / self.rg;
        LfModel {
            tp,
            te: tp * (1.0 + self.rk),
            ta: 1.0 / (self.fa * self.t0),
            t0: self.t0,
            ee: self.ee,
        }
    }
}

/// Context for an emitted pulse. pulse_index is global to this synthesis call,
/// starts at zero and includes pulses in crossfades and voiced-section preroll.
pub struct PulseContext<'a> {
    pub frame_index: usize,
    pub pulse_index: usize,
    pub frame: &'a Frame,
}
pub trait PbpEffect {
    /// Modify one LF pulse and return its time offset in seconds. The offset
    /// must be finite and no larger than one frame period in magnitude.
    fn modify(&mut self, gfm: &mut GlottalFlowParams, ctx: &PulseContext<'_>) -> f64;
}
impl<F> PbpEffect for F
where
    F: FnMut(&mut GlottalFlowParams, &PulseContext<'_>) -> f64,
{
    fn modify(&mut self, gfm: &mut GlottalFlowParams, ctx: &PulseContext<'_>) -> f64 {
        self(gfm, ctx)
    }
}

pub(crate) fn validate_config(conf: &ModelConfig) -> Result<(), LlsmError> {
    conf.validate()?;
    crate::error::config(
        conf.nspec.is_some_and(|n| n >= 33),
        "nspec",
        "Layer 1 requires an even FFT of at least 64 samples",
    )
}

pub(crate) fn decompose(chunk: &mut Chunk, nfft: usize) -> Result<(), LlsmError> {
    crate::error::config(
        nfft >= 64 && nfft.is_multiple_of(2),
        "nfft",
        "must be even and at least 64",
    )?;
    chunk.validate_layer0()?;
    for (index, f) in chunk.frames.iter().enumerate() {
        if f.is_voiced() && f.hm.as_ref().unwrap().ampl.is_empty() {
            return Err(LlsmError::InvalidFrame {
                index,
                reason: "Layer 1 decomposition needs at least one voiced harmonic",
            });
        }
    }
    let grid: Vec<_> = (0..64)
        .map(|i| 0.02 + (3.0 - 0.02) * i as f64 / 63.0)
        .collect();
    let cache = CachedGlottalModel::new(&grid, 80);
    let track: Vec<_> = chunk
        .frames
        .iter()
        .map(|f| {
            if !f.is_voiced() {
                return f64::NAN;
            }
            let h = f.hm.as_ref().unwrap();
            let n = h.num_harmonics().min((8000.0 / f.f0).round() as usize);
            let mut ampl = h.ampl[..n].to_vec();
            lipfilter(chunk.config.lip_radius, f.f0, &mut ampl, None, true);
            cache.fit(&ampl).unwrap_or(f64::NAN)
        })
        .collect();
    let filled = if track.iter().all(|v| v.is_nan()) {
        vec![1.0; track.len()]
    } else {
        fill_blanks(&track, f64::NAN)
    };
    let rd = smoothing_filter(
        &filled,
        (0.02 / chunk.config.thop).round().max(1.0) as usize,
    );
    let mut t = Transforms::default();
    let mut data = Vec::with_capacity(chunk.num_frames());
    for (index, (f, &rd)) in chunk.frames.iter().zip(&rd).enumerate() {
        if !f.is_voiced() {
            data.push(None);
            continue;
        }
        let source = source_amplitudes(rd, f.f0, f.hm.as_ref().unwrap().num_harmonics());
        let mut tract = f.hm.as_ref().unwrap().clone();
        lipfilter(
            chunk.config.lip_radius,
            f.f0,
            &mut tract.ampl,
            Some(&mut tract.phse),
            true,
        );
        for (v, s) in tract.ampl.iter_mut().zip(source) {
            *v /= s;
        }
        let phase = harmonic_minphase(&mut t, &tract.ampl);
        let vs_phse = tract
            .phse
            .iter()
            .zip(phase)
            .map(|(p, v)| wrap(p - v))
            .collect();
        let vt_magn =
            harmonic_envelope(&mut t, &tract.ampl, f.f0 / (2.0 * chunk.config.fnyq), nfft);
        let layer1 = Layer1Data { vt_magn, vs_phse };
        if layer1
            .vt_magn
            .iter()
            .chain(&layer1.vs_phse)
            .any(|v| !v.is_finite())
        {
            return Err(LlsmError::InvalidFrame {
                index,
                reason: "source/tract decomposition produced a nonfinite spectrum",
            });
        }
        data.push(Some(layer1));
    }
    chunk.config.nspec = Some(nfft / 2 + 1);
    for ((f, rd), data) in chunk.frames.iter_mut().zip(rd).zip(data) {
        f.rd = Some(rd);
        f.layer1 = data;
    }
    Ok(())
}

fn source_amplitudes(rd: f64, f0: f64, n: usize) -> Vec<f64> {
    let freq: Vec<_> = (1..=n).map(|i| i as f64 * f0).collect();
    let spectrum = LfModel::from_rd(rd, 1.0 / f0, 1.0).spectrum(&freq);
    let fundamental = spectrum[0].norm();
    spectrum
        .iter()
        .enumerate()
        .map(|(i, v)| v.norm() / ((i + 1) as f64 * fundamental))
        .collect()
}

fn tract_amplitudes(frame: &Frame, conf: &ModelConfig, n: usize) -> Vec<f64> {
    let l = frame.layer1.as_ref().unwrap();
    let axis: Vec<_> = (0..l.vt_magn.len())
        .map(|i| i as f64 / (l.vt_magn.len() - 1) as f64 * conf.fnyq)
        .collect();
    let freq: Vec<_> = (1..=n).map(|i| i as f64 * frame.f0).collect();
    interp1(&axis, &l.vt_magn, &freq)
        .iter()
        .map(|&v| db_to_log(v).exp())
        .collect()
}

pub(crate) fn reconstruct_harmonics(
    t: &mut Transforms,
    frame: &Frame,
    conf: &ModelConfig,
) -> Result<Option<HarmonicFrame>, &'static str> {
    if !frame.is_voiced() {
        return Ok(None);
    }
    let l = frame.layer1.as_ref().unwrap();
    let n = l
        .vs_phse
        .len()
        .min(conf.max_nhar)
        .min((conf.fnyq / frame.f0).floor() as usize);
    let mut ampl = tract_amplitudes(frame, conf, n);
    let mut phase = harmonic_minphase(t, &ampl);
    let source = source_amplitudes(frame.rd.unwrap(), frame.f0, n);
    for i in 0..n {
        ampl[i] *= source[i];
        phase[i] += l.vs_phse[i];
    }
    lipfilter(
        conf.lip_radius,
        frame.f0,
        &mut ampl,
        Some(&mut phase),
        false,
    );
    if ampl.iter().chain(&phase).any(|v| !v.is_finite()) {
        return Err("Layer 1 reconstruction produced a nonfinite harmonic model");
    }
    Ok(Some(HarmonicFrame { ampl, phse: phase }))
}

pub(crate) fn make_filtered_pulse(
    t: &mut Transforms,
    frame: &Frame,
    conf: &ModelConfig,
    fs: f64,
    pulses: &[(LfModel, f64)],
    size: usize,
    pre_rotate: usize,
) -> Result<Vec<f64>, &'static str> {
    let l = frame.layer1.as_ref().unwrap();
    let n = l
        .vs_phse
        .len()
        .min(conf.max_nhar)
        .min((conf.fnyq / frame.f0).floor() as usize);
    let harmonic_freq: Vec<_> = (0..=n).map(|i| i as f64 * frame.f0).collect();
    let original = LfModel::from_rd(frame.rd.unwrap(), 1.0 / frame.f0, 1.0);
    let source_harmonic = original.spectrum(&harmonic_freq[1..]);
    let normalizer = source_harmonic[0].norm();
    let source_shift = l.vs_phse[0] - (source_harmonic[0].arg() - PI / 2.0);
    let tract = tract_amplitudes(frame, conf, n);
    let tract_phase = harmonic_minphase(t, &tract);
    let mut correction = vec![0.0; n + 1];
    for i in 0..n {
        correction[i + 1] = wrap(
            l.vs_phse[i] - (source_harmonic[i].arg() - PI / 2.0) - source_shift * (i + 1) as f64,
        ) + tract_phase[i];
    }
    let real: Vec<_> = correction.iter().map(|p| p.cos()).collect();
    let imag: Vec<_> = correction.iter().map(|p| p.sin()).collect();
    let freq: Vec<_> = (0..=size / 2)
        .map(|i| i as f64 * fs / size as f64)
        .collect();
    let real = interp1(&harmonic_freq, &real, &freq);
    let imag = interp1(&harmonic_freq, &imag, &freq);
    let phase: Vec<_> = real.iter().zip(imag).map(|(r, i)| i.atan2(*r)).collect();
    let tract_axis: Vec<_> = (0..l.vt_magn.len())
        .map(|i| i as f64 / (l.vt_magn.len() - 1) as f64 * conf.fnyq)
        .collect();
    let tract_gain = interp1(&tract_axis, &l.vt_magn, &freq);
    let mut spectrum = vec![Complex::default(); freq.len()];
    for &(source, offset) in pulses {
        let pulse = source.spectrum(&freq);
        for k in 1..freq.len() {
            if freq[k] > conf.fnyq {
                continue;
            }
            // Integrate differentiated flow, normalize at the original H1,
            // and scale an isolated pulse's FFT for an fs/f0 sample pulse train.
            let gain = fs / (2.0 * freq[k]) / normalizer;
            let shift = -offset - pre_rotate as f64;
            spectrum[k] += pulse[k]
                * Complex::from_polar(
                    gain,
                    shift * TAU * k as f64 / size as f64 + phase[k] - PI / 2.0,
                );
        }
    }
    // FFT bin zero is DC; the existing harmonic API begins at one bin spacing.
    lipfilter_harmonics_complex(conf.lip_radius, fs / size as f64, &mut spectrum[1..], false);
    for (v, &gain) in spectrum.iter_mut().zip(&tract_gain) {
        *v *= db_to_log(gain).exp();
    }
    spectrum[0] = Complex::default();
    spectrum[size / 2].im = 0.0;
    if spectrum
        .iter()
        .any(|v| !v.re.is_finite() || !v.im.is_finite())
    {
        return Err("filtered LF pulse spectrum is nonfinite");
    }
    let mut signal = t.irfft(&mut spectrum, size);
    let fadein = 256.min(pre_rotate);
    let fadeout = 256.min(size);
    for (i, v) in signal.iter_mut().enumerate().take(fadein) {
        *v *= i as f64 / fadein as f64;
    }
    for (i, v) in signal.iter_mut().enumerate().skip(size - fadeout) {
        *v *= (size - i) as f64 / fadeout as f64;
    }
    Ok(signal)
}

/// Shared offline/streaming pulse tracker and three-period crossfade state.
#[derive(Clone)]
pub(crate) struct PeriodicState {
    frame_index: usize,
    previous_pulse: Option<f64>,
    buffered_periods: usize,
    pulse_index: usize,
    mix_state: f64,
    last_center: usize,
    rate: f64,
}

pub(crate) struct PeriodicFrame {
    pub(crate) harmonic: Vec<f64>,
    pub(crate) pulse: Option<(isize, Vec<f64>)>,
    /// Mix values for the interval from the previous frame center to this one.
    pub(crate) mix: Vec<f64>,
}

impl PeriodicState {
    pub(crate) fn new(hop: f64) -> Self {
        Self {
            frame_index: 0,
            previous_pulse: None,
            buffered_periods: 0,
            pulse_index: 0,
            mix_state: 0.0,
            last_center: 0,
            rate: 1.0 / hop,
        }
    }

    fn mix_to(&mut self, center: usize, target: f64, rate: f64) -> Vec<f64> {
        let mix = (self.last_center..center)
            .map(|_| {
                self.mix_state += (target - self.mix_state).clamp(-rate, rate);
                self.mix_state
            })
            .collect();
        self.last_center = center;
        mix
    }

    pub(crate) fn render_frame(
        &mut self,
        t: &mut Transforms,
        options: &SynthesisOptions,
        conf: &ModelConfig,
        frame: &Frame,
        w: &[f64],
        mut effect: Option<&mut dyn PbpEffect>,
    ) -> Result<PeriodicFrame, LlsmError> {
        let fs = options.fs;
        let index = self.frame_index;
        let hop = conf.thop * fs;
        let nwin = w.len();
        let raw = index as f64 * hop;
        let center = raw.round() as usize;
        if !frame.is_voiced() || frame.f0 > fs / 2.0 {
            let mix = self.mix_to(center, 0.0, self.rate);
            self.previous_pulse = None;
            self.buffered_periods = 0;
            self.frame_index += 1;
            return Ok(PeriodicFrame {
                harmonic: Vec::new(),
                pulse: None,
                mix,
            });
        }
        let source = LfModel::from_rd(frame.rd.unwrap(), 1.0 / frame.f0, 1.0);
        let source_phase = source.spectrum(&[frame.f0])[0].arg() - PI / 2.0;
        let nominal_period = fs / frame.f0;
        let p0 = frame.layer1.as_ref().unwrap().vs_phse[0];
        let projected = raw + (source_phase - p0).rem_euclid(TAU) / TAU * nominal_period;
        let reset = nominal_period.max(hop) * 2.0;
        let previous = self
            .previous_pulse
            .unwrap_or(projected - reset)
            .max(projected - reset);
        let count = ((projected - previous) / nominal_period).round().max(0.0) as usize;
        let period = if count > 0 {
            (projected - previous) / count as f64
        } else {
            nominal_period
        };
        let mut pulse = None;
        if (frame.pbp_enabled || self.buffered_periods > 0) && count > 0 {
            let mut sources = Vec::with_capacity(count);
            let base = previous.floor();
            for j in 0..count {
                let mut gfm = GlottalFlowParams::from_lf(source);
                let mut delta = 0.0;
                if let Some(effect) = effect.as_deref_mut() {
                    delta = effect.modify(
                        &mut gfm,
                        &PulseContext {
                            frame_index: index,
                            pulse_index: self.pulse_index,
                            frame,
                        },
                    );
                    gfm.check().map_err(|reason| LlsmError::InvalidPulse {
                        frame_index: index,
                        pulse_index: self.pulse_index,
                        reason,
                    })?;
                    if !delta.is_finite() || delta.abs() > source.t0 {
                        return Err(LlsmError::InvalidPulse {
                            frame_index: index,
                            pulse_index: self.pulse_index,
                            reason: "pulse offset must be finite and within one frame period",
                        });
                    }
                }
                sources.push((gfm.lf(), previous + j as f64 * period - base + delta * fs));
                self.pulse_index += 1;
            }
            // A legal negative effect offset can precede the nominal
            // prerotation by a fractional sample (or during a pitch change).
            // Keep every shifted pulse inside the FFT instead of wrapping it.
            let earliest = sources
                .iter()
                .map(|(_, offset)| *offset)
                .fold(0.0, f64::min);
            let pre_rotate = (period as usize).max((-earliest).ceil() as usize);
            // A hop can contain several pulses. Include the entire batch,
            // its preroll and a full period of response before the 256-sample
            // fadeout; a fixed two-period FFT would wrap later pulses and
            // modulate their amplitudes at the frame rate.
            let last_offset = sources
                .iter()
                .map(|(_, offset)| *offset)
                .fold(f64::NEG_INFINITY, f64::max);
            let span = pre_rotate as f64 + last_offset + period + 256.0;
            let size = (span.max(conf.nspec.unwrap() as f64).ceil() as usize).next_power_of_two();
            let signal = make_filtered_pulse(t, frame, conf, fs, &sources, size, pre_rotate)
                .map_err(|reason| LlsmError::InvalidPulse {
                    frame_index: index,
                    pulse_index: self.pulse_index - count,
                    reason,
                })?;
            pulse = Some((base as isize - pre_rotate as isize, signal));
            self.buffered_periods = if frame.pbp_enabled {
                (self.buffered_periods + count).min(3)
            } else {
                self.buffered_periods.saturating_sub(count)
            };
        }
        self.previous_pulse = Some(projected);
        self.rate = 1.0 / period.min(hop);
        let target = if frame.pbp_enabled && self.buffered_periods == 3 {
            1.0
        } else if !frame.pbp_enabled && self.buffered_periods == 0 {
            0.0
        } else {
            self.mix_state
        };
        let mix = self.mix_to(center, target, self.rate);
        // Layer 1 is always the source of truth, even if cached hm is present.
        let hm = reconstruct_harmonics(t, frame, conf)
            .map_err(|reason| LlsmError::InvalidFrame { index, reason })?
            .unwrap();
        let n = hm
            .num_harmonics()
            .min((fs / (2.0 * frame.f0)).floor() as usize);
        let correction = (raw - center as f64) * TAU / fs * frame.f0;
        let phase: Vec<_> = hm.phse[..n]
            .iter()
            .enumerate()
            .map(|(k, p)| p - correction * (k + 1) as f64)
            .collect();
        let mut signal = crate::dsp::sinusoid::synthesize(
            t,
            &hm.ampl[..n],
            &phase,
            frame.f0 / fs,
            nwin,
            options.iczt_threshold,
        );
        for (v, w) in signal.iter_mut().zip(w) {
            *v *= w;
        }
        self.frame_index += 1;
        Ok(PeriodicFrame {
            harmonic: signal,
            pulse,
            mix,
        })
    }
}

pub(crate) fn synthesize_periodic(
    t: &mut Transforms,
    options: &SynthesisOptions,
    chunk: &Chunk,
    length: usize,
    mut effect: Option<&mut dyn PbpEffect>,
) -> Result<Vec<f64>, LlsmError> {
    let hop = chunk.config.thop * options.fs;
    let w = crate::dsp::window::window(crate::dsp::window::Window::Hann, hop.round() as usize * 2);
    let mut harmonic = vec![0.0; length];
    let mut pulses = vec![0.0; length];
    let mut mix = vec![0.0; length];
    let mut state = PeriodicState::new(hop);
    for frame in &chunk.frames {
        let start = state.last_center;
        let rendered = match effect.as_mut() {
            Some(effect) => {
                state.render_frame(t, options, &chunk.config, frame, &w, Some(*effect))?
            }
            None => state.render_frame(t, options, &chunk.config, frame, &w, None)?,
        };
        mix[start..state.last_center].copy_from_slice(&rendered.mix);
        crate::layer0::overlap_add(
            &mut harmonic,
            &rendered.harmonic,
            state.last_center as isize - (w.len() / 2) as isize,
        );
        if let Some((start, signal)) = rendered.pulse {
            crate::layer0::overlap_add(&mut pulses, &signal, start);
        }
    }
    let start = state.last_center;
    mix[start..].copy_from_slice(&state.mix_to(length, 0.0, state.rate));
    Ok(harmonic
        .iter()
        .zip(&pulses)
        .zip(mix)
        .map(|((h, p), mix)| h * (1.0 - mix) + p * mix)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn intuitive_lf_parameters_roundtrip_native_rd_models() {
        for rd in [0.02, 0.3, 1.0, 2.7, 3.0] {
            for t0 in [0.1, 0.005, 1.0 / 1200.0] {
                let original = LfModel::from_rd(rd, t0, 1.2);
                let params = GlottalFlowParams::from_lf(original);
                params.validate().unwrap();
                let restored = params.lf();
                for (a, b) in [
                    (restored.tp, original.tp),
                    (restored.te, original.te),
                    (restored.ta, original.ta),
                    (restored.t0, original.t0),
                    (restored.ee, original.ee),
                ] {
                    assert!((a - b).abs() < 1e-14);
                }
            }
        }
    }
}
