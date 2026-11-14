//! Analysis options and the configuration retained with a model.
use crate::error::{LlsmError, config};

/// Lowest supported voiced pitch (Hz). Zero represents an unvoiced frame.
/// This also bounds pitch-dependent analysis windows at public entry points.
pub const MIN_VOICED_F0: f64 = 10.0;

pub(crate) fn valid_f0(f0: f64, nyquist: f64) -> bool {
    f0.is_finite() && (f0 == 0.0 || (f0 >= MIN_VOICED_F0 && f0 <= nyquist))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HarmonicAnalysisMethod {
    PeakPicking,
    #[default]
    ChirpZ,
}

/// Analysis parameters. Frequencies are Hz and time intervals are seconds.
#[derive(Debug, Clone)]
pub struct AnalysisOptions {
    pub thop: f64,
    pub max_nhar: usize,
    pub max_nhar_env: usize,
    pub npsd: usize,
    /// Strictly increasing channel boundaries, all below the input Nyquist.
    pub chan_freqs: Vec<f64>,
    /// Equivalent lip radius in cm.
    pub lip_radius: f64,
    pub refine_f0: bool,
    pub hm_method: HarmonicAnalysisMethod,
    /// Analysis window length in pitch periods; at least one period.
    pub rel_winsize: f64,
}
impl Default for AnalysisOptions {
    fn default() -> Self {
        Self {
            thop: 0.005,
            max_nhar: 100,
            max_nhar_env: 4,
            npsd: 256,
            chan_freqs: vec![2000.0, 4000.0, 8000.0],
            lip_radius: 1.5,
            refine_f0: true,
            hm_method: HarmonicAnalysisMethod::ChirpZ,
            rel_winsize: 4.0,
        }
    }
}
impl AnalysisOptions {
    /// Validate options and capture the model's analysis frequency grid.
    pub fn to_config(&self, fs: f64) -> Result<ModelConfig, LlsmError> {
        config(
            fs.is_finite() && fs > 0.0,
            "fs",
            "must be finite and positive",
        )?;
        config(
            self.rel_winsize.is_finite() && self.rel_winsize >= 1.0,
            "rel_winsize",
            "must be finite and at least one pitch period",
        )?;
        let conf = ModelConfig {
            thop: self.thop,
            max_nhar: self.max_nhar,
            max_nhar_env: self.max_nhar_env,
            npsd: self.npsd,
            fnyq: fs / 2.0,
            chan_freqs: self.chan_freqs.clone(),
            lip_radius: self.lip_radius,
            nspec: None,
        };
        conf.validate()?;
        Ok(conf)
    }
}

/// Configuration and source frequency extent shared by all frames in a chunk.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub thop: f64,
    pub max_nhar: usize,
    pub max_nhar_env: usize,
    pub npsd: usize,
    pub fnyq: f64,
    pub chan_freqs: Vec<f64>,
    pub lip_radius: f64,
    pub nspec: Option<usize>,
}
impl ModelConfig {
    pub fn num_channels(&self) -> usize {
        self.chan_freqs.len() + 1
    }
    /// Validate a configuration built or edited by a caller.
    pub fn validate(&self) -> Result<(), LlsmError> {
        config(
            self.fnyq.is_finite() && self.fnyq > 0.0,
            "fnyq",
            "must be finite and positive",
        )?;
        config(
            self.thop.is_finite()
                && self.thop > 0.0
                && (self.thop * self.fnyq * 2.0).is_finite()
                && self.thop * self.fnyq * 2.0 >= 2.0,
            "thop",
            "must be finite and span at least two source samples",
        )?;
        config(self.max_nhar > 0, "max_nhar", "must be positive")?;
        config(
            self.npsd >= 2,
            "npsd",
            "must contain at least two frequency points",
        )?;
        config(
            self.lip_radius.is_finite() && self.lip_radius > 0.0,
            "lip_radius",
            "must be finite and positive",
        )?;
        config(
            self.nspec.is_none_or(|n| n >= 3),
            "nspec",
            "must contain at least three frequency points",
        )?;
        let mut previous = 0.0;
        for &f in &self.chan_freqs {
            config(
                f.is_finite() && f > previous && f < self.fnyq,
                "chan_freqs",
                "must be finite, strictly increasing, positive and below Nyquist",
            )?;
            previous = f;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SynthesisMode {
    #[default]
    Layer0,
    Layer1,
}

/// Offline synthesis parameters. The seed controls every noise sample.
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    pub fs: f64,
    pub iczt_threshold: Option<(f64, f64)>,
    pub mode: SynthesisMode,
    pub random_seed: u64,
}
impl SynthesisOptions {
    pub fn new(fs: f64) -> Self {
        Self {
            fs,
            iczt_threshold: Some((0.275, 2.26)),
            mode: SynthesisMode::Layer0,
            random_seed: 1,
        }
    }
    pub(crate) fn validate(&self, thop: f64) -> Result<(), LlsmError> {
        config(
            self.fs.is_finite() && self.fs > 0.0,
            "fs",
            "must be finite and positive",
        )?;
        config(
            (thop * self.fs).is_finite() && thop * self.fs >= 2.0,
            "thop",
            "must span at least two output samples",
        )?;
        config(
            self.iczt_threshold
                .is_none_or(|(a, b)| a.is_finite() && b.is_finite()),
            "iczt_threshold",
            "parameters must be finite",
        )?;
        Ok(())
    }
}
