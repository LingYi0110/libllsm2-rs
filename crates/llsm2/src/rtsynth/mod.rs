//! Bounded, single-thread streaming synthesis without locks or background work.
mod buffer;
use crate::constants::{LOG_RESIDUAL_BIAS, db_to_log, log_to_power_db};
use crate::dsp::{
    fft::Transforms,
    noise::{SeededRng, bandlimited_noise},
    sinusoid,
    window::{Window, window},
};
use crate::layer0::NoiseFilter;
use crate::layer1::PeriodicState;
use crate::{
    Frame, LlsmError, MIN_VOICED_F0, ModelConfig, PbpEffect, SynthesisMode, SynthesisOptions,
};
use buffer::{DualBuffer, RingBuffer};
use std::f64::consts::TAU;

/// A deterministic, `Send` synthesizer driven by one caller. Each successful
/// feed publishes one rounded frame step, including initial latency samples.
/// Read regularly: a full output buffer returns `BufferFull` before any state
/// or effect callback advances. Read calls never block and never allocate.
///
/// The fixed delay covers the maximum pulse lookback at MIN_VOICED_F0 as well
/// as the centered noise STFT. It is typically about 0.4 seconds. At the end of
/// a stream, feed silent frames to clock out the delay and OLA tail.
/// Layer 0 successful feeds reuse buffers/plans prepared by new(), including
/// recursive and CZT synthesis. Layer 1 source/tract and pulse DSP still allocate.
/// No scheduling deadline is guaranteed.
///
/// ```
/// use llsm2::{AnalysisOptions, Chunk, RtSynthesizer, SynthesisOptions, LlsmError};
/// let config = AnalysisOptions::default().to_config(44100.0)?;
/// let silence = Chunk::new(config.clone(), 1);
/// let mut rt = RtSynthesizer::new(SynthesisOptions::new(44100.0), config, 4096)?;
/// rt.feed(&silence.frames[0])?;
/// let mut samples = [0.0; 256];
/// let count = rt.read(&mut samples); // initial delay is included
/// assert!(count > 0);
/// rt.clear();
/// assert_eq!(rt.available_samples(), 0);
/// # Ok::<(), LlsmError>(())
/// ```
pub struct RtSynthesizer {
    options: SynthesisOptions,
    config: ModelConfig,
    hop: f64,
    latency: usize,
    frame_index: usize,
    center: isize,
    available: usize,
    output_p: RingBuffer,
    output_ap: RingBuffer,
    harmonic: DualBuffer,
    pulse: DualBuffer,
    mix: DualBuffer,
    noise: DualBuffer,
    modulation: Vec<DualBuffer>,
    excitation: RingBuffer,
    templates: Vec<Vec<f64>>,
    pending_noise: [PendingNoise; 3],
    pending_head: usize,
    pending_len: usize,
    scratch: FrameScratch,
    synthesis_workspace: sinusoid::SynthesisWorkspace,
    periodic: PeriodicState,
    harmonic_window: Vec<f64>,
    filter: NoiseFilter,
    transforms: Transforms,
}

struct PendingNoise {
    center: isize,
    psd: Vec<f64>,
    residual: Vec<f64>,
    has_residual: bool,
}
struct FrameScratch {
    harmonic: Vec<f64>,
    phase: Vec<f64>,
    modulation: Vec<Vec<f64>>,
    excitation: Vec<f64>,
    filter_signal: Vec<f64>,
    output_p: Vec<f64>,
    output_ap: Vec<f64>,
}

impl RtSynthesizer {
    /// The output capacity must be at least `ceil(thop * fs) + 1` samples.
    /// Configurations and synthesis mode are checked once and owned thereafter.
    pub fn new(
        options: SynthesisOptions,
        config: ModelConfig,
        capacity_samples: usize,
    ) -> Result<Self, LlsmError> {
        match options.mode {
            SynthesisMode::Layer0 => config.validate()?,
            SynthesisMode::Layer1 => crate::layer1::validate_config(&config)?,
        }
        options.validate(config.thop)?;
        let hop = config.thop * options.fs;
        let max_step = hop.ceil() as usize;
        crate::error::config(
            capacity_samples > max_step,
            "capacity_samples",
            "must be at least ceil(thop * fs) + 1",
        )?;
        let mut filter = NoiseFilter::new(&config, options.fs);
        let longest = (options.fs / MIN_VOICED_F0).max(hop);
        // A pulse batch spans at most two max(period, hop) intervals. Its
        // prerotation can add two more intervals of lookback; an effect may
        // shift the final pulse by one original period. Include FFT padding.
        let pulse_size = ((5.0 * longest + 258.0)
            .max(config.nspec.unwrap_or(0) as f64)
            .ceil() as usize)
            .next_power_of_two();
        let latency = (4.0 * longest + 2.0)
            .max(3.0 * hop + filter.nfft as f64 / 2.0 + 3.0)
            .ceil() as usize;
        let internal = ((0.2 * options.fs).ceil() as usize)
            .max(filter.nfft + 4 * max_step)
            .max(pulse_size + longest.ceil() as usize + 2)
            .max(latency + max_step + 2);
        let mut rng = SeededRng::new(options.random_seed);
        let nyquist = config.fnyq.min(options.fs / 2.0);
        // Same 20,000-sample stationary templates and 128-sample circular
        // crossfades as offline bandlimited_noise, independent of stream length.
        let templates = (0..config.num_channels())
            .map(|c| {
                let low = if c == 0 {
                    0.0
                } else {
                    config.chan_freqs[c - 1]
                };
                let high = config
                    .chan_freqs
                    .get(c)
                    .copied()
                    .unwrap_or(nyquist)
                    .min(nyquist);
                if low >= nyquist {
                    Vec::new()
                } else {
                    bandlimited_noise(&mut rng, 20000, low / options.fs, high / options.fs)
                }
            })
            .collect();
        let modulation = (0..config.num_channels())
            .map(|_| DualBuffer::new(internal))
            .collect();
        let pending_noise = std::array::from_fn(|_| PendingNoise {
            center: 0,
            psd: vec![0.0; config.npsd],
            residual: vec![0.0; config.npsd],
            has_residual: false,
        });
        let harmonic_len = hop.round() as usize * 2;
        let scratch = FrameScratch {
            harmonic: vec![0.0; harmonic_len],
            phase: vec![0.0; config.max_nhar],
            modulation: vec![vec![0.0; filter.window.len()]; config.num_channels()],
            excitation: vec![0.0; max_step],
            filter_signal: vec![0.0; filter.window.len()],
            output_p: vec![0.0; max_step],
            output_ap: vec![0.0; max_step],
        };
        let mut transforms = Transforms::default();
        let mut synthesis_workspace = sinusoid::SynthesisWorkspace::new(
            config.max_nhar.max(config.max_nhar_env),
            harmonic_len.max(filter.window.len()),
        );
        synthesis_workspace.warm(&mut transforms);
        filter.warm(&mut transforms);
        Ok(Self {
            harmonic_window: window(Window::Hann, hop.round() as usize * 2),
            periodic: PeriodicState::new(hop),
            options,
            config,
            hop,
            latency,
            frame_index: 0,
            center: 0,
            available: 0,
            output_p: RingBuffer::new(capacity_samples),
            output_ap: RingBuffer::new(capacity_samples),
            harmonic: DualBuffer::new(internal),
            pulse: DualBuffer::new(internal),
            mix: DualBuffer::new(internal),
            noise: DualBuffer::new(internal),
            modulation,
            excitation: RingBuffer::new(internal),
            templates,
            pending_noise,
            pending_head: 0,
            pending_len: 0,
            scratch,
            synthesis_workspace,
            filter,
            transforms,
        })
    }
    pub fn latency(&self) -> usize {
        self.latency
    }
    pub fn available_samples(&self) -> usize {
        self.available
    }
    pub fn feed(&mut self, frame: &Frame) -> Result<(), LlsmError> {
        self.feed_impl(frame, None)
    }
    pub fn feed_with_effect(
        &mut self,
        frame: &Frame,
        effect: &mut dyn PbpEffect,
    ) -> Result<(), LlsmError> {
        if self.options.mode != SynthesisMode::Layer1 {
            return Err(LlsmError::InvalidInput {
                field: "mode",
                reason: "pulse effects require Layer 1 synthesis",
            });
        }
        self.feed_impl(frame, Some(effect))
    }
    fn feed_impl(
        &mut self,
        frame: &Frame,
        effect: Option<&mut dyn PbpEffect>,
    ) -> Result<(), LlsmError> {
        let center = (self.frame_index as f64 * self.hop).round() as isize;
        let next_center = ((self.frame_index + 1) as f64 * self.hop).round() as isize;
        let step = (next_center - center) as usize;
        let free = self.output_p.capacity() - self.available;
        if free < step {
            return Err(LlsmError::BufferFull {
                required: step,
                available: free,
            });
        }
        let validity = match self.options.mode {
            SynthesisMode::Layer0 => frame.validate_layer0(&self.config),
            SynthesisMode::Layer1 => frame.validate_layer1(&self.config),
        };
        validity.map_err(|reason| LlsmError::InvalidFrame {
            index: self.frame_index,
            reason,
        })?;
        if frame.nm.psd.iter().enumerate().any(|(j, &psd)| {
            let target = psd
                + frame
                    .nm
                    .psd_residual
                    .as_ref()
                    .map_or(0.0, |r| r[j] - log_to_power_db(LOG_RESIDUAL_BIAS));
            !target.is_finite() || !db_to_log(target).exp().is_finite()
        }) {
            return Err(LlsmError::InvalidFrame {
                index: self.frame_index,
                reason: "noise gain exceeds representable linear amplitude",
            });
        }
        // Stage fallible periodic/effect work before changing sample buffers.
        let mut periodic = self.periodic.clone();
        let rendered = if self.options.mode == SynthesisMode::Layer1 {
            Some(periodic.render_frame(
                &mut self.transforms,
                &self.options,
                &self.config,
                frame,
                &self.harmonic_window,
                effect,
            )?)
        } else {
            None
        };
        let harmonic = if let Some(rendered) = &rendered {
            rendered.harmonic.as_slice()
        } else if frame.is_voiced() {
            let hm = frame.hm.as_ref().unwrap();
            let n = hm
                .num_harmonics()
                .min((self.options.fs / (2.0 * frame.f0)).floor() as usize);
            let correction = (self.frame_index as f64 * self.hop - center as f64) * TAU
                / self.options.fs
                * frame.f0;
            for (k, p) in self.scratch.phase[..n].iter_mut().enumerate() {
                *p = hm.phse[k] - correction * (k + 1) as f64;
            }
            self.synthesis_workspace.synthesize_into(
                &mut self.transforms,
                &hm.ampl[..n],
                &self.scratch.phase[..n],
                frame.f0 / self.options.fs,
                &mut self.scratch.harmonic,
                self.options.iczt_threshold,
            );
            for (v, w) in self.scratch.harmonic.iter_mut().zip(&self.harmonic_window) {
                *v *= w;
            }
            self.scratch.harmonic.as_slice()
        } else {
            &[]
        };
        for ((hm, &dc), signal) in frame
            .nm
            .eenv
            .iter()
            .zip(&frame.nm.edc)
            .zip(&mut self.scratch.modulation)
        {
            let n = if frame.is_voiced() {
                hm.num_harmonics()
                    .min((self.options.fs / (2.0 * frame.f0)).floor() as usize)
            } else {
                0
            };
            self.synthesis_workspace.synthesize_into(
                &mut self.transforms,
                &hm.ampl[..n],
                &hm.phse[..n],
                frame.f0 / self.options.fs,
                signal,
                self.options.iczt_threshold,
            );
            for (v, w) in signal.iter_mut().zip(&self.filter.window) {
                *v = (*v + dc).max(1e-8) * w;
            }
        }
        if harmonic
            .iter()
            .chain(self.scratch.modulation.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(LlsmError::InvalidFrame {
                index: self.frame_index,
                reason: "frame synthesis produced nonfinite samples",
            });
        }
        let advance = (center - self.center) as usize;
        self.harmonic.forward(advance);
        self.pulse.forward(advance);
        self.mix.forward(advance);
        self.noise.forward(advance);
        for buffer in &mut self.modulation {
            buffer.forward(advance);
        }
        self.harmonic
            .add(-(self.harmonic_window.len() as isize / 2), harmonic);
        if let Some(rendered) = rendered {
            self.mix.add(-(rendered.mix.len() as isize), &rendered.mix);
            if let Some((start, signal)) = rendered.pulse {
                self.pulse.add(start - center, &signal);
            }
        }
        let mod_start = ((self.frame_index as f64 - 1.0) * self.hop).round() as isize;
        for (buffer, signal) in self.modulation.iter_mut().zip(&self.scratch.modulation) {
            buffer.add(mod_start - center, signal);
        }
        // Future modulation windows begin at this center, so the preceding
        // interval is now complete. Keep only this bounded excitation history.
        let excitation = &mut self.scratch.excitation[..advance];
        excitation.fill(0.0);
        for (c, template) in self.templates.iter().enumerate() {
            if template.is_empty() {
                continue;
            }
            for (j, v) in excitation.iter_mut().enumerate() {
                let time = self.center + j as isize;
                let envelope = self.modulation[c].read(time - center);
                *v += envelope.sqrt() * stationary_sample(template, time as usize);
            }
        }
        self.excitation.append(excitation);
        let slot = &mut self.pending_noise[(self.pending_head + self.pending_len) % 3];
        slot.center = center;
        slot.psd.copy_from_slice(&frame.nm.psd);
        slot.has_residual = frame.nm.psd_residual.is_some();
        if let Some(r) = &frame.nm.psd_residual {
            slot.residual.copy_from_slice(r);
        }
        self.pending_len += 1;
        // Two-frame lookahead also covers fractional-hop rounding at window ends.
        if self.pending_len == 3 {
            let nm = &self.pending_noise[self.pending_head];
            let at = nm.center;
            self.excitation.read_into(
                at - center - self.scratch.filter_signal.len() as isize / 2,
                &mut self.scratch.filter_signal,
            );
            let nfft = self.filter.nfft;
            if let Some(filtered) = self.filter.process(
                &mut self.transforms,
                &nm.psd,
                nm.has_residual.then_some(nm.residual.as_slice()),
                &self.scratch.filter_signal,
            ) {
                self.noise.add(at - center - nfft as isize / 2, filtered);
            }
            self.pending_head = (self.pending_head + 1) % 3;
            self.pending_len -= 1;
        }
        let start = center - self.latency as isize;
        let output_p = &mut self.scratch.output_p[..step];
        let output_ap = &mut self.scratch.output_ap[..step];
        output_p.fill(0.0);
        output_ap.fill(0.0);
        for j in 0..step {
            let time = start + j as isize;
            if time < 0 {
                continue;
            }
            let at = time - center;
            let mix = self.mix.read(at);
            output_p[j] = self.harmonic.read(at) * (1.0 - mix) + self.pulse.read(at) * mix;
            output_ap[j] = self.noise.read(at);
        }
        self.output_p.append(output_p);
        self.output_ap.append(output_ap);
        self.available += step;
        self.periodic = periodic;
        self.frame_index += 1;
        self.center = center;
        Ok(())
    }
    /// Read mixed output. Only the returned prefix is written.
    pub fn read(&mut self, out: &mut [f64]) -> usize {
        let n = out.len().min(self.available);
        let (p, pt) = self.output_p.slices(-(self.available as isize), n);
        let (ap, apt) = self.output_ap.slices(-(self.available as isize), n);
        let (dst, tail) = out[..n].split_at_mut(p.len());
        crate::dsp::vector::sum(dst, p, ap);
        crate::dsp::vector::sum(tail, pt, apt);
        self.available -= n;
        n
    }
    /// The returned count is min(available, both slice lengths).
    pub fn read_decomposed(
        &mut self,
        out_periodic: &mut [f64],
        out_aperiodic: &mut [f64],
    ) -> usize {
        let n = self
            .available
            .min(out_periodic.len())
            .min(out_aperiodic.len());
        self.output_p
            .read_into(-(self.available as isize), &mut out_periodic[..n]);
        self.output_ap
            .read_into(-(self.available as isize), &mut out_aperiodic[..n]);
        self.available -= n;
        n
    }
    /// Reset all histories, counters, pending noise frames and template phase.
    /// Buffers and transform plans remain allocated. External effects are owned
    /// by the caller and must be reset separately if they contain state.
    pub fn clear(&mut self) {
        self.output_p.clear();
        self.output_ap.clear();
        self.excitation.clear();
        self.harmonic.clear();
        self.pulse.clear();
        self.mix.clear();
        self.noise.clear();
        for buffer in &mut self.modulation {
            buffer.clear();
        }
        for nm in &mut self.pending_noise {
            nm.center = 0;
            nm.psd.fill(0.0);
            nm.residual.fill(0.0);
            nm.has_residual = false;
        }
        self.pending_head = 0;
        self.pending_len = 0;
        self.scratch.harmonic.fill(0.0);
        self.scratch.phase.fill(0.0);
        for m in &mut self.scratch.modulation {
            m.fill(0.0);
        }
        self.scratch.excitation.fill(0.0);
        self.scratch.filter_signal.fill(0.0);
        self.scratch.output_p.fill(0.0);
        self.scratch.output_ap.fill(0.0);
        self.periodic = PeriodicState::new(self.hop);
        self.frame_index = 0;
        self.center = 0;
        self.available = 0;
    }
}

fn stationary_sample(template: &[f64], time: usize) -> f64 {
    let overlap = 128;
    let period = template.len() - overlap;
    if time < period {
        return template[time];
    }
    let phase = (time - period) % period;
    if phase >= overlap {
        return template[phase];
    }
    let r = phase as f64 / overlap as f64;
    (template[period + phase] * (1.0 - r) + template[phase] * r)
        / ((1.0 - r).powi(2) + r * r).sqrt()
}
