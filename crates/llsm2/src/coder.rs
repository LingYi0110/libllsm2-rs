//! Fixed-dimensional statistical features for phase-free speech modeling.
use crate::constants::{db_to_log, log_to_db, log_to_power_db, power_db_to_log};
use crate::dsp::{
    fft::Transforms,
    interp::interp1,
    lfmodel::LfModel,
    spectral::{harmonic_minphase, lipfilter},
};
use crate::error::config;
use crate::{Frame, HarmonicFrame, Layer1Data, LlsmError, MIN_VOICED_F0, ModelConfig, NoiseFrame};
use rustdct::{DctPlanner, TransformType2And3};
use std::sync::Arc;

/// Lossy Layer 1 codec with features `[VUV, F0, Rd, MelSpec..., BAP...]`.
///
/// MelSpec contains natural-log amplitudes sampled on a compressed Mel axis,
/// not cepstral coefficients. BAP contains linear noise/total power ratios.
/// Original phases, noise envelope modulation and PSD residuals are discarded.
/// Transform plans are immutable; one coder supports concurrent `&self` calls.
///
/// ```
/// use llsm2::{AnalysisOptions, Coder, LlsmError};
/// let mut config = AnalysisOptions::default().to_config(44100.0)?;
/// config.nspec = Some(1025); // Layer 1 grid for a 2048-sample FFT
/// let coder = Coder::new(&config, 64, 5)?;
/// let mut features = vec![-3.0; coder.feature_dim()];
/// features[..3].copy_from_slice(&[1.0, 200.0, 1.0]); // VUV, F0, Rd
/// features[67..].fill(0.2); // five BAP values
/// let frame = coder.decode_layer0(&features)?;
/// assert_eq!(frame.f0, 200.0);
/// assert!(frame.hm.is_some());
/// # Ok::<(), LlsmError>(())
/// ```
pub struct Coder {
    config: ModelConfig,
    order_spec: usize,
    order_bap: usize,
    faxis: Vec<f64>,
    psdaxis: Vec<f64>,
    melaxis: Vec<f64>,
    apaxis: Vec<f64>,
    full_dct2: Arc<dyn TransformType2And3<f64>>,
    full_dct3: Arc<dyn TransformType2And3<f64>>,
    mel_dct2: Arc<dyn TransformType2And3<f64>>,
    mel_dct3: Arc<dyn TransformType2And3<f64>>,
}

impl Coder {
    /// Capture a valid Layer 1 configuration. `1 <= order_spec < nspec` and
    /// `1 <= order_bap < nspec`; neither order needs to be a power of two.
    /// Source Nyquist must exceed 50 Hz, the lower endpoint of the Mel axis.
    pub fn new(conf: &ModelConfig, order_spec: usize, order_bap: usize) -> Result<Self, LlsmError> {
        crate::layer1::validate_config(conf)?;
        let ns = conf.nspec.unwrap();
        config(
            conf.fnyq > 50.0,
            "fnyq",
            "Coder requires Nyquist above 50 Hz",
        )?;
        config(
            (1..ns).contains(&order_spec),
            "order_spec",
            "must be positive and smaller than nspec",
        )?;
        config(
            (1..ns).contains(&order_bap),
            "order_bap",
            "must be positive and no larger than the number of spectral intervals",
        )?;
        let faxis = (0..ns)
            .map(|i| i as f64 / (ns - 1) as f64 * conf.fnyq)
            .collect();
        let psdaxis = (0..conf.npsd)
            .map(|i| i as f64 / (conf.npsd - 1) as f64 * conf.fnyq)
            .collect();
        let mel_floor = (50.0_f64 / 700.0).ln_1p();
        let mel_ceil = (conf.fnyq / 700.0).ln_1p();
        // Preserve coder.c's division by ns: the last sample is below Nyquist.
        // The factor 1125 cancels between freq2mel and mel2freq.
        let melaxis = (0..ns)
            .map(|i| 700.0 * (mel_floor + (mel_ceil - mel_floor) * i as f64 / ns as f64).exp_m1())
            .collect();
        let apaxis = (0..=order_bap)
            .map(|i| i as f64 / order_bap as f64 * conf.fnyq)
            .collect();
        let mut planner = DctPlanner::new();
        Ok(Self {
            config: conf.clone(),
            order_spec,
            order_bap,
            faxis,
            psdaxis,
            melaxis,
            apaxis,
            full_dct2: planner.plan_dct2(ns - 1),
            full_dct3: planner.plan_dct3(ns - 1),
            mel_dct2: planner.plan_dct2(order_spec),
            mel_dct3: planner.plan_dct3(order_spec),
        })
    }

    pub fn feature_dim(&self) -> usize {
        self.order_spec + self.order_bap + 3
    }

    /// Encode a valid Layer 1 frame. Unvoiced F0/Rd features are both zero,
    /// matching the reference format; unvoiced BAP features are all one.
    pub fn encode(&self, frame: &Frame) -> Result<Vec<f64>, LlsmError> {
        let mut dst = vec![0.0; self.feature_dim()];
        self.encode_into(frame, &mut dst)?;
        Ok(dst)
    }

    /// Encode into a buffer of exactly `feature_dim()` elements.
    pub fn encode_into(&self, frame: &Frame, dst: &mut [f64]) -> Result<(), LlsmError> {
        if dst.len() != self.feature_dim() {
            return Err(LlsmError::InvalidInput {
                field: "dst",
                reason: "length must equal Coder::feature_dim()",
            });
        }
        frame
            .validate_layer1(&self.config)
            .map_err(|reason| LlsmError::InvalidFrame { index: 0, reason })?;
        let ns = self.faxis.len();
        let noise: Vec<_> = interp1(&self.psdaxis, &frame.nm.psd, &self.faxis)
            .into_iter()
            .map(|v| power_db_to_log(v).exp())
            .collect();
        let mut total = noise.clone();
        let mut bap = vec![1.0; self.order_bap];
        if frame.is_voiced() {
            let source = LfModel::from_rd(frame.rd.unwrap(), 1.0 / frame.f0, 1.0);
            let spectrum = source.spectrum(&self.faxis);
            let fundamental = source.spectrum(&[frame.f0])[0].norm();
            let tract = &frame.layer1.as_ref().unwrap().vt_magn;
            let mut periodic = vec![0.0; ns];
            for j in 1..ns {
                periodic[j] = db_to_log(tract[j]).exp() * spectrum[j].norm() / fundamental
                    * frame.f0
                    / self.faxis[j];
            }
            lipfilter(
                self.config.lip_radius,
                self.config.fnyq / (ns - 1) as f64,
                &mut periodic[1..],
                None,
                false,
            );
            periodic[0] = periodic[1];
            for (power, amplitude) in total.iter_mut().zip(periodic) {
                *power += amplitude * amplitude * 44100.0 / (4.0 * frame.f0);
            }
            for (band, value) in bap.iter_mut().enumerate() {
                let start = band * (ns - 1) / self.order_bap;
                let end = (band + 1) * (ns - 1) / self.order_bap;
                *value =
                    (start..end).map(|k| noise[k] / total[k]).sum::<f64>() / (end - start) as f64;
            }
        }
        let logs: Vec<_> = total.iter().map(|v| 0.5 * v.ln()).collect();
        let mel = self.compress_mel(&interp1(&self.faxis, &logs, &self.melaxis));
        if mel.iter().chain(&bap).any(|v| !v.is_finite()) {
            return Err(LlsmError::InvalidFrame {
                index: 0,
                reason: "Coder encoding produced nonfinite features",
            });
        }
        dst[0] = f64::from(frame.is_voiced());
        dst[1] = frame.f0;
        dst[2] = if frame.is_voiced() {
            frame.rd.unwrap()
        } else {
            0.0
        };
        dst[3..3 + self.order_spec].copy_from_slice(&mel);
        dst[3 + self.order_spec..].copy_from_slice(&bap);
        Ok(())
    }

    /// Decode with LF source phases and a tract envelope. VUV > 0.5 is voiced.
    /// Finite model predictions are clamped: F0 to [20 Hz, source Nyquist],
    /// Rd to [0.02, 3], BAP to [0, 1]. Unvoiced output F0 is zero.
    pub fn decode_layer1(&self, features: &[f64]) -> Result<Frame, LlsmError> {
        self.decode(features, true)
    }

    /// Decode directly to radiated harmonic amplitudes and minimum-phase tract
    /// plus LF source phases. Uses the same feature bounds as `decode_layer1`.
    pub fn decode_layer0(&self, features: &[f64]) -> Result<Frame, LlsmError> {
        self.decode(features, false)
    }

    fn compress_mel(&self, mel: &[f64]) -> Vec<f64> {
        let n = self.faxis.len() - 1;
        let mut coefficients = mel[..n].to_vec();
        self.full_dct2.process_dct2(&mut coefficients);
        coefficients.truncate(self.order_spec);
        // C halves coefficient 0 before Ooura DCT-III, which gives it full
        // weight. rustdct already gives it half weight, so no adjustment here.
        self.mel_dct3.process_dct3(&mut coefficients);
        coefficients.iter_mut().for_each(|v| *v *= 2.0 / n as f64);
        coefficients
    }

    fn expand_mel(&self, mel: &[f64]) -> Vec<f64> {
        let n = self.faxis.len() - 1;
        let mut coefficients: Vec<_> = mel
            .iter()
            .map(|v| v * (n as f64 / self.order_spec as f64))
            .collect();
        self.mel_dct2.process_dct2(&mut coefficients);
        coefficients.resize(n, 0.0);
        self.full_dct3.process_dct3(&mut coefficients);
        coefficients.iter_mut().for_each(|v| *v *= 2.0 / n as f64);
        coefficients.push(coefficients[n - 1]);
        coefficients
    }

    fn decode(&self, features: &[f64], layer1: bool) -> Result<Frame, LlsmError> {
        if features.len() != self.feature_dim() || features.iter().any(|v| !v.is_finite()) {
            return Err(LlsmError::InvalidInput {
                field: "features",
                reason: "must have Coder::feature_dim() finite values",
            });
        }
        let voiced = features[0] > 0.5;
        let f0 = features[1].clamp(MIN_VOICED_F0.max(20.0), self.config.fnyq);
        let rd = features[2].clamp(0.02, 3.0);
        let ns = self.faxis.len();
        let mel = self.expand_mel(&features[3..3 + self.order_spec]);
        let logs = interp1(&self.melaxis, &mel, &self.faxis);
        let mut bap = Vec::with_capacity(self.order_bap + 1);
        bap.push(if voiced { 0.0 } else { 1.0 });
        bap.extend(
            features[3 + self.order_spec..]
                .iter()
                .map(|v| v.clamp(0.0, 1.0)),
        );
        let mut noise = interp1(&self.apaxis, &bap, &self.faxis);
        let mut amplitude = Vec::with_capacity(ns);
        for (j, (&log, ap)) in logs.iter().zip(&mut noise).enumerate() {
            let f = self.faxis[j];
            if voiced {
                if f < 500.0 {
                    *ap = 1e-3;
                } else if f < 2000.0 {
                    *ap = 1e-3 + (*ap - 1e-3) * (f - 500.0) / 1500.0;
                }
            }
            let power = (2.0 * log).exp();
            if !power.is_finite() {
                return Err(LlsmError::InvalidInput {
                    field: "features",
                    reason: "Mel spectrum exceeds representable linear power",
                });
            }
            amplitude.push((power * (1.0 - *ap) * (f0 * 4.0 / 44100.0)).sqrt());
            *ap *= power;
        }
        let mut nm = NoiseFrame::new(
            self.config.num_channels(),
            self.config.max_nhar_env,
            self.config.npsd,
        );
        nm.psd = interp1(&self.faxis, &noise, &self.psdaxis)
            .into_iter()
            .map(|p| log_to_power_db(p.max(f64::MIN_POSITIVE).ln()))
            .collect();
        let mut frame = Frame {
            f0: if voiced { f0 } else { 0.0 },
            rd: Some(rd),
            hm: None,
            layer1: None,
            nm,
            pbp_enabled: false,
        };
        if voiced {
            let n = self
                .config
                .max_nhar
                .min((self.config.fnyq / f0).floor() as usize);
            let frequencies: Vec<_> = (1..=n).map(|i| i as f64 * f0).collect();
            let source = LfModel::from_rd(rd, 1.0 / f0, 1.0);
            let harmonic_source = source.spectrum(&frequencies);
            let vs_phse = harmonic_source.iter().map(|v| v.arg()).collect();
            let fundamental = harmonic_source[0].norm();
            if layer1 {
                lipfilter(
                    self.config.lip_radius,
                    self.config.fnyq / (ns - 1) as f64,
                    &mut amplitude[1..],
                    None,
                    true,
                );
                let spectrum = source.spectrum(&self.faxis);
                for j in 1..ns {
                    amplitude[j] = log_to_db(
                        (amplitude[j] * self.faxis[j] / f0 * fundamental / spectrum[j].norm())
                            .max(f64::MIN_POSITIVE)
                            .ln(),
                    );
                }
                amplitude[0] = amplitude[1];
                frame.layer1 = Some(Layer1Data {
                    vt_magn: amplitude,
                    vs_phse,
                });
            } else {
                let radiated = interp1(&self.faxis, &amplitude, &frequencies);
                let mut tract = radiated.clone();
                lipfilter(self.config.lip_radius, f0, &mut tract, None, true);
                for (i, (v, source)) in tract.iter_mut().zip(&harmonic_source).enumerate() {
                    *v /= source.norm() / ((i + 1) as f64 * fundamental);
                }
                let mut phase = harmonic_minphase(&mut Transforms::default(), &tract);
                lipfilter(
                    self.config.lip_radius,
                    f0,
                    &mut tract,
                    Some(&mut phase),
                    false,
                );
                for (p, source) in phase.iter_mut().zip(vs_phse) {
                    *p += source;
                }
                frame.hm = Some(HarmonicFrame {
                    ampl: radiated,
                    phse: phase,
                });
            }
        }
        let validity = if layer1 {
            frame.validate_layer1(&self.config)
        } else {
            frame.validate_layer0(&self.config)
        };
        validity.map_err(|reason| LlsmError::InvalidInput {
            field: "features",
            reason,
        })?;
        Ok(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AnalysisOptions;
    use std::f64::consts::PI;

    fn configuration(ns: usize) -> ModelConfig {
        let mut config = AnalysisOptions::default().to_config(44100.0).unwrap();
        config.nspec = Some(ns);
        config
    }

    #[test]
    fn dct_compression_and_expansion_match_independent_cosine_sums() {
        for (n, m) in [(64, 1), (64, 40), (64, 60), (64, 64), (66, 60), (1024, 64)] {
            let coder = Coder::new(&configuration(n + 1), m, 5).unwrap();
            let mel: Vec<_> = (0..=n).map(|j| -2.0 + (0.17 * j as f64).sin()).collect();
            let coefficients: Vec<f64> = (0..m)
                .map(|k| {
                    (0..n)
                        .map(|j| mel[j] * (PI * k as f64 * (j as f64 + 0.5) / n as f64).cos())
                        .sum()
                })
                .collect();
            let expected: Vec<_> = (0..m)
                .map(|j| {
                    (0.5 * coefficients[0]
                        + (1..m)
                            .map(|k| {
                                coefficients[k]
                                    * (PI * k as f64 * (j as f64 + 0.5) / m as f64).cos()
                            })
                            .sum::<f64>())
                        * 2.0
                        / n as f64
                })
                .collect();
            let compressed = coder.compress_mel(&mel);
            for (a, b) in compressed.iter().zip(expected) {
                assert!((a - b).abs() < 1e-11);
            }
            let expanded = coder.expand_mel(&compressed);
            for (j, &value) in expanded[..n].iter().enumerate() {
                let expected = (0.5 * coefficients[0]
                    + (1..m)
                        .map(|k| {
                            coefficients[k] * (PI * k as f64 * (j as f64 + 0.5) / n as f64).cos()
                        })
                        .sum::<f64>())
                    * 2.0
                    / n as f64;
                assert!((value - expected).abs() < 1e-11);
            }
            assert_eq!(expanded[n], expanded[n - 1]);
            let recompressed = coder.compress_mel(&expanded);
            for (a, b) in recompressed.iter().zip(&compressed) {
                assert!((a - b).abs() < 1e-11);
            }
            let constant = coder.compress_mel(&vec![-3.0; n + 1]);
            assert!(constant.iter().all(|v| (v + 3.0).abs() < 1e-12));
        }
    }

    #[test]
    fn bap_encoding_uses_physical_radiation_bins_and_reference_power_scale() {
        let config = configuration(65);
        let coder = Coder::new(&config, 40, 5).unwrap();
        let mut frame = Frame {
            f0: 200.0,
            rd: Some(1.0),
            hm: None,
            layer1: Some(Layer1Data {
                vt_magn: vec![-20.0; 65],
                vs_phse: vec![0.0; 100],
            }),
            nm: NoiseFrame::new(config.num_channels(), config.max_nhar_env, config.npsd),
            pbp_enabled: false,
        };
        frame.nm.psd.fill(-20.0);
        let encoded = coder.encode(&frame).unwrap();
        let source = LfModel::from_rd(1.0, 1.0 / 200.0, 1.0);
        let fundamental = source.spectrum(&[200.0])[0].norm();
        let spectrum = source.spectrum(&coder.faxis);
        let mut periodic = vec![0.0; 65];
        for j in 1..65 {
            let f = j as f64 * config.fnyq / 64.0;
            // Independently evaluate the radiation magnitude at bin k*df.
            let r = 128.0 / (9.0 * PI * PI);
            let w_l = 2.0 * PI * f * 8.0 * config.lip_radius / (100.0 * 3.0 * PI * 340.0);
            let lip = w_l * r / (r * r + w_l * w_l).sqrt();
            let amplitude = 0.1 * spectrum[j].norm() / fundamental * 200.0 / f * lip;
            periodic[j] = amplitude.powi(2) * 44100.0 / (4.0 * 200.0);
        }
        periodic[0] = periodic[1];
        for band in 0..5 {
            let start = band * 64 / 5;
            let end = (band + 1) * 64 / 5;
            let expected = (start..end)
                .map(|k| 0.01 / (0.01 + periodic[k]))
                .sum::<f64>()
                / (end - start) as f64;
            assert!((encoded[43 + band] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn decode_preserves_power_units_and_voiced_low_frequency_noise_suppression() {
        let mut config = configuration(1025);
        config.npsd = 1025;
        let coder = Coder::new(&config, 64, 5).unwrap();
        let mut features = vec![0.0; coder.feature_dim()];
        features[..3].copy_from_slice(&[1.0, 200.0, 1.0]);
        features[67..].fill(0.6);
        let frame = coder.decode_layer1(&features).unwrap();
        for (j, &db) in frame.nm.psd.iter().enumerate() {
            let f = coder.faxis[j];
            let raw_ap = 0.6 * (f / (config.fnyq / 5.0)).min(1.0);
            let ap = if f < 500.0 {
                1e-3
            } else if f < 2000.0 {
                1e-3 + (raw_ap - 1e-3) * (f - 500.0) / 1500.0
            } else {
                raw_ap
            };
            assert!((db - 10.0 * ap.log10()).abs() < 1e-10);
        }
        let harmonics = coder.decode_layer0(&features).unwrap().hm.unwrap();
        let expected = (0.999_f64 * 200.0 * 4.0 / 44100.0).sqrt();
        assert!((harmonics.ampl[0] - expected).abs() < 1e-12);
        assert!((harmonics.ampl[1] - expected).abs() < 1e-12);
        let source = LfModel::from_rd(1.0, 1.0 / 200.0, 1.0);
        let fundamental = source.spectrum(&[200.0])[0].norm();
        let tract = &frame.layer1.as_ref().unwrap().vt_magn;
        assert_eq!(tract[0], tract[1]);
        for j in [1, 30, 1024] {
            let f = coder.faxis[j];
            let ap = power_db_to_log(frame.nm.psd[j]).exp();
            let amplitude = ((1.0 - ap) * 200.0 * 4.0 / 44100.0).sqrt();
            let lip = crate::dsp::spectral::lip_response(config.lip_radius, f).norm();
            let expected = 20.0
                * (amplitude / lip * f / 200.0 * fundamental / source.spectrum(&[f])[0].norm())
                    .log10();
            assert!((tract[j] - expected).abs() < 1e-10);
        }
        features[0] = 0.5; // The threshold itself is unvoiced.
        features[67..].fill(1.0);
        features[3..67].fill(-3.0);
        let frame = coder.decode_layer1(&features).unwrap();
        assert_eq!(frame.f0, 0.0);
        assert!(
            frame
                .nm
                .psd
                .iter()
                .all(|v| (v - log_to_power_db(-6.0)).abs() < 1e-10)
        );
    }
}
