//! Low Level Speech Model v2.1, implemented in safe Rust with f64 arithmetic.
//!
//! Offline Layer 0/1 analysis/synthesis with editable harmonic/source/tract frames.
//!
//! ```
//! use llsm2::{analyze, synthesize, AnalysisOptions, SynthesisOptions, LlsmError};
//! use std::f64::consts::TAU;
//!
//! let fs = 44100.0;
//! let x: Vec<_> = (0..4410).map(|i| 0.1*(TAU*200.0*i as f64/fs).sin()).collect();
//! let f0 = vec![200.0;20]; // known pitch of this synthetic signal, Hz
//! let analysis = analyze(&AnalysisOptions::default(), &x, fs, &f0)?;
//! let output = synthesize(&SynthesisOptions::new(fs), &analysis.chunk)?;
//! assert!(output.y.iter().all(|v| v.is_finite()));
//! # Ok::<(), LlsmError>(())
//! ```
#![forbid(unsafe_code)]

pub mod coder;
pub mod config;
mod constants;
mod dsp;
pub mod error;
pub mod frame;
mod layer0;
mod layer1;
pub mod modify;
pub mod rtsynth;

pub use coder::Coder;
pub use config::{
    AnalysisOptions, HarmonicAnalysisMethod, MIN_VOICED_F0, ModelConfig, SynthesisMode,
    SynthesisOptions,
};
pub use error::LlsmError;
pub use frame::{Chunk, Frame, HarmonicFrame, Layer1Data, NoiseFrame, PhaseDirection};
pub use layer0::{AnalysisResult, SynthesisOutput, analyze, synthesize, synthesize_with_effect};
pub use layer1::{GlottalFlowParams, PbpEffect, PulseContext};
pub use modify::interpolate_frames;
pub use rtsynth::RtSynthesizer;
