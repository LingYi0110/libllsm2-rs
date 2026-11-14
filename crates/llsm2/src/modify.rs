use crate::constants::{LOG_RESIDUAL_BIAS, log_to_power_db};
use crate::dsp::interp::circular;
use crate::dsp::interp::wrap;
use crate::{Chunk, PhaseDirection};
use crate::{Frame, HarmonicFrame, Layer1Data, LlsmError, ModelConfig, NoiseFrame};
use std::f64::consts::TAU;

fn blend(a: f64, b: f64, t: f64) -> f64 {
    a * (1.0 - t) + b * t
}

fn interpolate_harmonics(a: &HarmonicFrame, b: &HarmonicFrame, t: f64) -> HarmonicFrame {
    let n = a.num_harmonics().max(b.num_harmonics());
    let ampl = (0..n)
        .map(|i| {
            blend(
                a.ampl.get(i).copied().unwrap_or(0.0),
                b.ampl.get(i).copied().unwrap_or(0.0),
                t,
            )
        })
        .collect();
    let phse = (0..n)
        .map(|i| match (a.phse.get(i), b.phse.get(i)) {
            (Some(&a), Some(&b)) => circular(a, b, t),
            (Some(&a), None) => a,
            (None, Some(&b)) => b,
            (None, None) => unreachable!("one harmonic model contains this index"),
        })
        .collect();
    HarmonicFrame { ampl, phse }
}

/// Interpolate valid frames on the same configuration. `ratio` is in `[0,1]`;
/// endpoints are exact clones. At voiced/unvoiced boundaries, keep the voiced
/// F0 and fade its amplitude. Layer 1 tract spectra are floored at -80 dB.
/// Phases follow the shortest circular arc; absent harmonics fade from/to zero.
/// Interior Layer 1 results clear cached hm. PSD residuals interpolate their
/// effective dB correction, with None meaning zero correction.
/// With `config.nspec` set, both inputs must satisfy the Layer 1 contract,
/// including Rd on unvoiced frames; otherwise both must be valid Layer 0 frames.
pub fn interpolate_frames(
    config: &ModelConfig,
    left: &Frame,
    right: &Frame,
    ratio: f64,
) -> Result<Frame, LlsmError> {
    let layer1_config = config.nspec.is_some();
    if layer1_config {
        crate::layer1::validate_config(config)?;
    } else {
        config.validate()?;
    }
    if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
        return Err(LlsmError::InvalidInput {
            field: "ratio",
            reason: "must be finite and in [0, 1]",
        });
    }
    for (index, frame) in [left, right].iter().enumerate() {
        let result = if layer1_config {
            frame.validate_layer1(config)
        } else {
            frame.validate_layer0(config)
        };
        result.map_err(|reason| LlsmError::InvalidFrame { index, reason })?;
    }
    if ratio == 0.0 {
        return Ok(left.clone());
    }
    if ratio == 1.0 {
        return Ok(right.clone());
    }
    let av = left.is_voiced();
    let bv = right.is_voiced();
    let f0 = match (av, bv) {
        (true, true) => blend(left.f0, right.f0, ratio),
        (true, false) => left.f0,
        (false, true) => right.f0,
        (false, false) => 0.0,
    };
    let rd = match (left.rd, right.rd) {
        (Some(a), Some(b)) => Some(blend(a, b, ratio)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let empty = HarmonicFrame::default();
    let eenv = left
        .nm
        .eenv
        .iter()
        .zip(&right.nm.eenv)
        .map(|(a, b)| {
            interpolate_harmonics(
                if av { a } else { &empty },
                if bv { b } else { &empty },
                ratio,
            )
        })
        .collect();
    let edc = left
        .nm
        .edc
        .iter()
        .zip(&right.nm.edc)
        .map(|(&a, &b)| blend(a, b, ratio))
        .collect();
    let psd = left
        .nm
        .psd
        .iter()
        .zip(&right.nm.psd)
        .map(|(&a, &b)| blend(a, b, ratio))
        .collect();
    let neutral = log_to_power_db(LOG_RESIDUAL_BIAS);
    let psd_residual = match (&left.nm.psd_residual, &right.nm.psd_residual) {
        (None, None) => None,
        (a, b) => Some(
            (0..config.npsd)
                .map(|i| {
                    blend(
                        a.as_ref().map_or(neutral, |a| a[i]),
                        b.as_ref().map_or(neutral, |b| b[i]),
                        ratio,
                    )
                })
                .collect(),
        ),
    };
    let voiced = if av { left } else { right };
    let layer1 = if av && bv {
        match (&left.layer1, &right.layer1) {
            (Some(a), Some(b)) => {
                let n = a.vs_phse.len().max(b.vs_phse.len());
                let vs_phse = (0..n)
                    .map(|i| match (a.vs_phse.get(i), b.vs_phse.get(i)) {
                        (Some(&a), Some(&b)) => circular(a, b, ratio),
                        (Some(&a), None) => a,
                        (None, Some(&b)) => b,
                        _ => unreachable!(),
                    })
                    .collect();
                Some(Layer1Data {
                    vt_magn: a
                        .vt_magn
                        .iter()
                        .zip(&b.vt_magn)
                        .map(|(&a, &b)| blend(a, b, ratio).max(-80.0))
                        .collect(),
                    vs_phse,
                })
            }
            _ => None,
        }
    } else if av || bv {
        voiced.layer1.as_ref().map(|l| {
            let gain = 20.0 * (if av { 1.0 - ratio } else { ratio }).max(1e-4).log10();
            Layer1Data {
                vt_magn: l.vt_magn.iter().map(|v| (v + gain).max(-80.0)).collect(),
                vs_phse: l.vs_phse.clone(),
            }
        })
    } else {
        None
    };
    let hm = if layer1.is_some() || (!av && !bv) {
        None
    } else {
        let mut h = interpolate_harmonics(
            if av {
                left.hm.as_ref().unwrap()
            } else {
                &empty
            },
            if bv {
                right.hm.as_ref().unwrap()
            } else {
                &empty
            },
            ratio,
        );
        let n = config.max_nhar.min((config.fnyq / f0).floor() as usize);
        h.ampl.truncate(n);
        h.phse.truncate(n);
        Some(h)
    };
    let pbp_enabled = if av && bv {
        if ratio < 0.5 {
            left.pbp_enabled
        } else {
            right.pbp_enabled
        }
    } else {
        (av || bv) && voiced.pbp_enabled
    };
    Ok(Frame {
        f0,
        rd,
        hm,
        layer1,
        nm: NoiseFrame {
            eenv,
            edc,
            psd,
            psd_residual,
        },
        pbp_enabled,
    })
}

pub(crate) fn phase_propagate(chunk: &mut Chunk, direction: PhaseDirection) {
    let sign = match direction {
        PhaseDirection::Forward => 1.0,
        PhaseDirection::Backward => -1.0,
    };
    let mut phase = 0.0;
    for frame in &mut chunk.frames {
        // Inclusive rectangular integration, matching llsm_chunk_phasepropagate.
        phase = wrap(phase + frame.f0 * chunk.config.thop * TAU);
        frame.phase_shift(sign * phase);
    }
}
