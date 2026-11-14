//! v2.1 logarithmic noise statistics and amplitude/power conversions.
use std::f64::consts::{LN_10, PI};

pub(crate) const EULER_GAMMA: f64 = std::f64::consts::EULER_GAMMA;
pub(crate) const LOG_CHI2_VARIANCE: f64 = PI * PI / 6.0;
pub(crate) const LOG_RESIDUAL_BIAS: f64 = 0.375;

pub(crate) fn log_to_db(x: f64) -> f64 {
    x * 20.0 / LN_10
}
pub(crate) fn db_to_log(x: f64) -> f64 {
    x * LN_10 / 20.0
}
pub(crate) fn log_to_power_db(x: f64) -> f64 {
    x * 10.0 / LN_10
}
pub(crate) fn power_db_to_log(x: f64) -> f64 {
    x * LN_10 / 10.0
}
