use crate::{Error, Result, input_slice};
use llsm2::{
    AnalysisOptions, Chunk, Coder, Frame, HarmonicAnalysisMethod, HarmonicFrame, Layer1Data,
    ModelConfig, NoiseFrame, RtSynthesizer, SynthesisOptions, SynthesisOutput,
};
use std::ffi::c_void;

pub const LLSM_LAYER0: u32 = 0;
pub const LLSM_LAYER1: u32 = 1;
pub const LLSM_PEAK_PICKING: u32 = 0;
pub const LLSM_CHIRP_Z: u32 = 1;

/// ABI status values are stable. Details of the last failure are thread-local.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlsmStatus {
    Ok = 0,
    NullPointer = 1,
    InvalidConfig = 2,
    InvalidFrame = 3,
    BufferTooSmall = 4,
    InternalError = 5,
    InvalidInput = 6,
    BufferFull = 7,
    InvalidPulse = 8,
}

/// Borrowed channel boundaries. nspec=0 means no Layer 1 spectral grid.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlsmModelConfig {
    pub thop: f64,
    pub max_nhar: usize,
    pub max_nhar_env: usize,
    pub npsd: usize,
    pub fnyq: f64,
    pub chan_freqs: *const f64,
    pub num_chan_freqs: usize,
    pub lip_radius: f64,
    pub nspec: usize,
}
impl LlsmModelConfig {
    pub(crate) unsafe fn owned(&self) -> Result<ModelConfig> {
        Ok(ModelConfig {
            thop: self.thop,
            max_nhar: self.max_nhar,
            max_nhar_env: self.max_nhar_env,
            npsd: self.npsd,
            fnyq: self.fnyq,
            chan_freqs: unsafe { input_slice(self.chan_freqs, self.num_chan_freqs)? }.to_vec(),
            lip_radius: self.lip_radius,
            nspec: if self.nspec == 0 {
                None
            } else {
                Some(self.nspec)
            },
        })
    }
    pub(crate) fn view(c: &ModelConfig) -> Self {
        Self {
            thop: c.thop,
            max_nhar: c.max_nhar,
            max_nhar_env: c.max_nhar_env,
            npsd: c.npsd,
            fnyq: c.fnyq,
            chan_freqs: c.chan_freqs.as_ptr(),
            num_chan_freqs: c.chan_freqs.len(),
            lip_radius: c.lip_radius,
            nspec: c.nspec.unwrap_or(0),
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlsmAnalysisOptions {
    pub thop: f64,
    pub max_nhar: usize,
    pub max_nhar_env: usize,
    pub npsd: usize,
    pub chan_freqs: *const f64,
    pub num_chan_freqs: usize,
    pub lip_radius: f64,
    /// 0 or 1. Methods: LLSM_PEAK_PICKING or LLSM_CHIRP_Z.
    pub refine_f0: u32,
    pub hm_method: u32,
    pub rel_winsize: f64,
}
impl LlsmAnalysisOptions {
    pub(crate) unsafe fn owned(&self) -> Result<AnalysisOptions> {
        let hm_method = match self.hm_method {
            0 => HarmonicAnalysisMethod::PeakPicking,
            1 => HarmonicAnalysisMethod::ChirpZ,
            _ => return Err(Error::Input("invalid harmonic analysis method")),
        };
        Ok(AnalysisOptions {
            thop: self.thop,
            max_nhar: self.max_nhar,
            max_nhar_env: self.max_nhar_env,
            npsd: self.npsd,
            chan_freqs: unsafe { input_slice(self.chan_freqs, self.num_chan_freqs)? }.to_vec(),
            lip_radius: self.lip_radius,
            refine_f0: crate::flag(self.refine_f0)?,
            hm_method,
            rel_winsize: self.rel_winsize,
        })
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlsmSynthesisOptions {
    pub fs: f64,
    /// LLSM_LAYER0 or LLSM_LAYER1; use_iczt is 0 or 1.
    pub mode: u32,
    pub use_iczt: u32,
    pub iczt_a: f64,
    pub iczt_b: f64,
    pub random_seed: u64,
}
impl LlsmSynthesisOptions {
    pub(crate) fn owned(&self) -> Result<SynthesisOptions> {
        Ok(SynthesisOptions {
            fs: self.fs,
            mode: crate::mode(self.mode)?,
            iczt_threshold: if crate::flag(self.use_iczt)? {
                Some((self.iczt_a, self.iczt_b))
            } else {
                None
            },
            random_seed: self.random_seed,
        })
    }
}
/// Read-only arrays. A zero-length array may have a null pointer.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlsmHarmonicView {
    pub ampl: *const f64,
    pub phse: *const f64,
    pub len: usize,
}
impl LlsmHarmonicView {
    fn view(h: &HarmonicFrame) -> Self {
        Self {
            ampl: h.ampl.as_ptr(),
            phse: h.phse.as_ptr(),
            len: h.ampl.len(),
        }
    }
    unsafe fn owned(&self) -> Result<HarmonicFrame> {
        Ok(HarmonicFrame {
            ampl: unsafe { input_slice(self.ampl, self.len)? }.to_vec(),
            phse: unsafe { input_slice(self.phse, self.len)? }.to_vec(),
        })
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlsmNoiseView {
    pub eenv: *const LlsmHarmonicView,
    pub edc: *const f64,
    pub num_channels: usize,
    pub psd: *const f64,
    pub npsd: usize,
    pub psd_residual: *const f64,
    pub residual_len: usize,
}
/// A frame view is borrowed until its owning handle is mutated or destroyed.
/// Input views are deep-copied. rd=0 means absent, has_harmonic/pbp_enabled=0/1.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LlsmFrameView {
    pub f0: f64,
    pub rd: f64,
    pub has_harmonic: u32,
    pub harmonic: LlsmHarmonicView,
    pub vt_magn: *const f64,
    pub nspec: usize,
    pub vs_phse: *const f64,
    pub nsource: usize,
    pub noise: LlsmNoiseView,
    pub pbp_enabled: u32,
}
impl LlsmFrameView {
    pub(crate) unsafe fn owned(&self) -> Result<Frame> {
        let nm = &self.noise;
        let eenv = unsafe { input_slice(nm.eenv, nm.num_channels)? }
            .iter()
            .map(|h| unsafe { h.owned() })
            .collect::<Result<_>>()?;
        Ok(Frame {
            f0: self.f0,
            rd: if self.rd == 0.0 { None } else { Some(self.rd) },
            hm: if crate::flag(self.has_harmonic)? {
                Some(unsafe { self.harmonic.owned()? })
            } else {
                None
            },
            layer1: if self.nspec == 0 && self.nsource == 0 {
                None
            } else {
                Some(Layer1Data {
                    vt_magn: unsafe { input_slice(self.vt_magn, self.nspec)? }.to_vec(),
                    vs_phse: unsafe { input_slice(self.vs_phse, self.nsource)? }.to_vec(),
                })
            },
            nm: NoiseFrame {
                eenv,
                edc: unsafe { input_slice(nm.edc, nm.num_channels)? }.to_vec(),
                psd: unsafe { input_slice(nm.psd, nm.npsd)? }.to_vec(),
                psd_residual: if nm.residual_len == 0 {
                    None
                } else {
                    Some(unsafe { input_slice(nm.psd_residual, nm.residual_len)? }.to_vec())
                },
            },
            pbp_enabled: crate::flag(self.pbp_enabled)?,
        })
    }
}

pub struct LlsmChunk {
    pub(crate) inner: Chunk,
}
pub struct LlsmFrame {
    pub(crate) inner: Frame,
    envelope_views: Vec<LlsmHarmonicView>,
}
pub struct LlsmCoder {
    pub(crate) inner: Coder,
}
pub struct LlsmRtSynth {
    pub(crate) inner: RtSynthesizer,
}
pub struct LlsmOutput {
    pub(crate) inner: SynthesisOutput,
}
impl LlsmFrame {
    pub(crate) fn new(inner: Frame) -> Self {
        let envelope_views = inner.nm.eenv.iter().map(LlsmHarmonicView::view).collect();
        Self {
            inner,
            envelope_views,
        }
    }
    pub(crate) fn view(&self) -> LlsmFrameView {
        let f = &self.inner;
        let nm = &f.nm;
        let empty = LlsmHarmonicView {
            ampl: std::ptr::null(),
            phse: std::ptr::null(),
            len: 0,
        };
        LlsmFrameView {
            f0: f.f0,
            rd: f.rd.unwrap_or(0.0),
            has_harmonic: u32::from(f.hm.is_some()),
            harmonic: f.hm.as_ref().map_or(empty, LlsmHarmonicView::view),
            vt_magn: f
                .layer1
                .as_ref()
                .map_or(std::ptr::null(), |l| l.vt_magn.as_ptr()),
            nspec: f.layer1.as_ref().map_or(0, |l| l.vt_magn.len()),
            vs_phse: f
                .layer1
                .as_ref()
                .map_or(std::ptr::null(), |l| l.vs_phse.as_ptr()),
            nsource: f.layer1.as_ref().map_or(0, |l| l.vs_phse.len()),
            noise: LlsmNoiseView {
                eenv: self.envelope_views.as_ptr(),
                edc: nm.edc.as_ptr(),
                num_channels: nm.edc.len(),
                psd: nm.psd.as_ptr(),
                npsd: nm.psd.len(),
                psd_residual: nm
                    .psd_residual
                    .as_ref()
                    .map_or(std::ptr::null(), |r| r.as_ptr()),
                residual_len: nm.psd_residual.as_ref().map_or(0, |r| r.len()),
            },
            pbp_enabled: u32::from(f.pbp_enabled),
        }
    }
}
/// Borrowed output arrays, valid until llsm2_output_destroy.
#[repr(C)]
pub struct LlsmOutputView {
    pub fs: f64,
    pub len: usize,
    pub y: *const f64,
    pub periodic: *const f64,
    pub aperiodic: *const f64,
}
#[repr(C)]
pub struct LlsmGlottalFlowParams {
    pub fa: f64,
    pub rk: f64,
    pub rg: f64,
    pub t0: f64,
    pub ee: f64,
}
#[repr(C)]
pub struct LlsmPulseContext {
    pub frame_index: usize,
    pub pulse_index: usize,
    pub f0: f64,
    pub rd: f64,
}
/// Returns a time offset in seconds; modifies the supplied LF parameters.
pub type LlsmPbpCallback = Option<
    unsafe extern "C" fn(*mut LlsmGlottalFlowParams, *const LlsmPulseContext, *mut c_void) -> f64,
>;
