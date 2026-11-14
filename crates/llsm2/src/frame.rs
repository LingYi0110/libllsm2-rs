//! Owned model frames. Mutations are checked when crossing the synthesis boundary.
use crate::config::valid_f0;
use crate::dsp::interp::wrap;
use crate::{LlsmError, ModelConfig};

/// Harmonic amplitudes (linear) and center phases (radians), of equal length.
#[derive(Debug, Clone, Default)]
pub struct HarmonicFrame {
    pub ampl: Vec<f64>,
    pub phse: Vec<f64>,
}
impl HarmonicFrame {
    pub fn new(nhar: usize) -> Self {
        Self {
            ampl: vec![0.0; nhar],
            phse: vec![0.0; nhar],
        }
    }
    pub fn num_harmonics(&self) -> usize {
        self.ampl.len()
    }
    /// Shift the fundamental by a finite angle and higher harmonics by k*theta.
    pub fn phase_shift(&mut self, theta: f64) {
        for (i, p) in self.phse.iter_mut().enumerate() {
            *p = wrap(*p + theta * (i + 1) as f64);
        }
    }
    fn valid(&self, maximum: usize) -> bool {
        self.ampl.len() == self.phse.len()
            && self.ampl.len() <= maximum
            && self.ampl.iter().all(|&v| v.is_finite() && v >= 0.0)
            && self.phse.iter().all(|v| v.is_finite())
    }
}

/// v2.1 stationary noise PSD, independently replaceable residual, and band envelopes.
#[derive(Debug, Clone)]
pub struct NoiseFrame {
    pub eenv: Vec<HarmonicFrame>,
    pub edc: Vec<f64>,
    /// Power dB, referred to a 44.1 kHz sample rate.
    pub psd: Vec<f64>,
    /// Additive power dB residual, on the same frequency grid as psd.
    pub psd_residual: Option<Vec<f64>>,
}
impl NoiseFrame {
    pub fn new(num_channels: usize, nhar_env: usize, npsd: usize) -> Self {
        Self {
            eenv: vec![HarmonicFrame::new(nhar_env); num_channels],
            edc: vec![1e-5; num_channels],
            psd: vec![-120.0; npsd],
            psd_residual: None,
        }
    }
}

/// Layer 1 tract amplitude response (dB) and source harmonic phases (radians).
#[derive(Debug, Clone)]
pub struct Layer1Data {
    pub vt_magn: Vec<f64>,
    pub vs_phse: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Frame {
    /// Pitch in Hz: zero (unvoiced), or MIN_VOICED_F0 through source Nyquist.
    pub f0: f64,
    pub rd: Option<f64>,
    pub hm: Option<HarmonicFrame>,
    pub layer1: Option<Layer1Data>,
    pub nm: NoiseFrame,
    pub pbp_enabled: bool,
}
impl Frame {
    pub fn is_voiced(&self) -> bool {
        self.f0 > 0.0
    }
    /// Shift all harmonic phases, including noise envelopes and Layer 1 phases.
    pub fn phase_shift(&mut self, theta: f64) {
        if let Some(hm) = &mut self.hm {
            hm.phase_shift(theta);
        }
        for hm in &mut self.nm.eenv {
            hm.phase_shift(theta);
        }
        if let Some(l1) = &mut self.layer1 {
            for (i, p) in l1.vs_phse.iter_mut().enumerate() {
                *p = wrap(*p + theta * (i + 1) as f64);
            }
        }
    }
    pub fn phase_sync_rps(&mut self, layer1_based: bool) {
        let source = if layer1_based {
            self.layer1
                .as_ref()
                .and_then(|l| l.vs_phse.first())
                .copied()
        } else {
            None
        };
        let reference = source
            .or_else(|| self.hm.as_ref().and_then(|hm| hm.phse.first()).copied())
            .unwrap_or(0.0);
        self.phase_shift(-reference);
    }
    pub fn clear_layer0(&mut self) {
        self.hm = None;
    }

    /// Rebuild Layer 0 from the current Layer 1 parameters. Harmonic count is
    /// limited by the stored phases, configured maximum and source Nyquist.
    pub fn to_layer0(&mut self, config: &ModelConfig) -> Result<(), LlsmError> {
        crate::layer1::validate_config(config)?;
        self.validate_layer1(config)
            .map_err(|reason| LlsmError::InvalidFrame { index: 0, reason })?;
        self.hm = crate::layer1::reconstruct_harmonics(
            &mut crate::dsp::fft::Transforms::default(),
            self,
            config,
        )
        .map_err(|reason| LlsmError::InvalidFrame { index: 0, reason })?;
        Ok(())
    }

    pub(crate) fn validate_layer0(&self, conf: &ModelConfig) -> Result<(), &'static str> {
        self.validate_common(conf)?;
        if self.is_voiced() && self.hm.is_none() {
            return Err("voiced Layer 0 frame has no harmonic model");
        }
        if self.hm.as_ref().is_some_and(|hm| !hm.valid(conf.max_nhar)) {
            return Err("invalid harmonic amplitudes, phases or count");
        }
        Ok(())
    }

    pub(crate) fn validate_layer1(&self, conf: &ModelConfig) -> Result<(), &'static str> {
        self.validate_common(conf)?;
        if self.rd.is_none_or(|rd| !(0.02..=3.0).contains(&rd)) {
            return Err("Layer 1 requires Rd in [0.02, 3.0] on every frame");
        }
        if self.is_voiced() && self.layer1.as_ref().is_none_or(|l| l.vs_phse.is_empty()) {
            return Err(
                "voiced Layer 1 frame requires a tract spectrum and at least one source phase",
            );
        }
        Ok(())
    }

    pub(crate) fn validate_common(&self, conf: &ModelConfig) -> Result<(), &'static str> {
        if !valid_f0(self.f0, conf.fnyq) {
            return Err("F0 must be zero or finite in [10 Hz, source Nyquist]");
        }
        if self.rd.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            return Err("Rd must be finite and positive");
        }
        let nm = &self.nm;
        if nm.eenv.len() != conf.num_channels() || nm.edc.len() != conf.num_channels() {
            return Err("noise channel count differs from configuration");
        }
        if nm.eenv.iter().any(|h| !h.valid(conf.max_nhar_env))
            || nm.edc.iter().any(|&v| !v.is_finite() || v < 0.0)
        {
            return Err("invalid noise envelope");
        }
        if nm.psd.len() != conf.npsd || nm.psd.iter().any(|v| !v.is_finite()) {
            return Err("noise PSD differs from configured dimension or contains nonfinite values");
        }
        if nm
            .psd_residual
            .as_ref()
            .is_some_and(|r| r.len() != conf.npsd || r.iter().any(|v| !v.is_finite()))
        {
            return Err("invalid noise PSD residual");
        }
        if let Some(l1) = &self.layer1
            && (Some(l1.vt_magn.len()) != conf.nspec
                || l1.vt_magn.iter().chain(&l1.vs_phse).any(|v| !v.is_finite()))
        {
            return Err("Layer 1 data differs from configuration or contains nonfinite values");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub config: ModelConfig,
    pub frames: Vec<Frame>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseDirection {
    Forward,
    Backward,
}

impl Chunk {
    /// Allocate unvoiced frames. Validation requires a nonempty frame sequence;
    /// an empty chunk can be used while assembling a model but not synthesized.
    pub fn new(config: ModelConfig, num_frames: usize) -> Self {
        let frame = Frame {
            f0: 0.0,
            rd: None,
            hm: None,
            layer1: None,
            nm: NoiseFrame::new(config.num_channels(), config.max_nhar_env, config.npsd),
            pbp_enabled: false,
        };
        Self {
            config,
            frames: vec![frame; num_frames],
        }
    }
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }
    pub fn f0_contour(&self) -> Vec<f64> {
        self.frames.iter().map(|f| f.f0).collect()
    }
    pub fn clear_layer0(&mut self) {
        for f in &mut self.frames {
            f.clear_layer0();
        }
    }
    pub fn phase_sync_rps(&mut self, layer1_based: bool) {
        for f in &mut self.frames {
            f.phase_sync_rps(layer1_based);
        }
    }
    pub fn phase_propagate(&mut self, direction: PhaseDirection) {
        crate::modify::phase_propagate(self, direction);
    }
    /// Estimate a continuous Rd track and decompose voiced Layer 0 frames.
    /// nfft must be even and at least 64. Existing harmonics are retained.
    pub fn to_layer1(&mut self, nfft: usize) -> Result<(), LlsmError> {
        crate::layer1::decompose(self, nfft)
    }
    /// Rebuild all harmonics from Layer 1, replacing stale Layer 0 data.
    pub fn to_layer0(&mut self) -> Result<(), LlsmError> {
        self.validate_layer1()?;
        let mut t = crate::dsp::fft::Transforms::default();
        let harmonics: Result<Vec<_>, _> = self
            .frames
            .iter()
            .enumerate()
            .map(|(index, f)| {
                crate::layer1::reconstruct_harmonics(&mut t, f, &self.config)
                    .map_err(|reason| LlsmError::InvalidFrame { index, reason })
            })
            .collect();
        for (frame, hm) in self.frames.iter_mut().zip(harmonics?) {
            frame.hm = hm;
        }
        Ok(())
    }
    /// Validate the Layer 1 source of truth; cached Layer 0 harmonics are ignored.
    pub fn validate_layer1(&self) -> Result<(), LlsmError> {
        crate::layer1::validate_config(&self.config)?;
        self.validate_nonempty()?;
        for (index, f) in self.frames.iter().enumerate() {
            f.validate_layer1(&self.config)
                .map_err(|reason| LlsmError::InvalidFrame { index, reason })?;
        }
        Ok(())
    }
    /// Validate caller-edited data before Layer 0 synthesis.
    pub fn validate_layer0(&self) -> Result<(), LlsmError> {
        self.config.validate()?;
        self.validate_nonempty()?;
        for (index, f) in self.frames.iter().enumerate() {
            f.validate_layer0(&self.config)
                .map_err(|reason| LlsmError::InvalidFrame { index, reason })?;
        }
        Ok(())
    }
    fn validate_nonempty(&self) -> Result<(), LlsmError> {
        if self.frames.is_empty() {
            return Err(LlsmError::InvalidInput {
                field: "frames",
                reason: "must contain at least one frame",
            });
        }
        Ok(())
    }
}
