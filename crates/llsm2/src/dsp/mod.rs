//! Internal DSP contracts: sizes/axes are supplied by the model layer, frequencies
//! are cycles/sample unless marked Hz, and harmonic phases refer to sample n/2.
//! Input validation belongs to the public analysis/synthesis boundary.
pub(crate) mod f0;
pub(crate) mod fft;
pub(crate) mod filter;
mod filter_coef;
pub(crate) mod interp;
pub(crate) mod lfmodel;
pub(crate) mod noise;
pub(crate) mod sinusoid;
pub(crate) mod spectral;
pub(crate) mod vector;
pub(crate) mod window;

#[cfg(test)]
mod tests;
