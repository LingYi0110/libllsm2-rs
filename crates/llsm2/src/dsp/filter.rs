use super::filter_coef::*;

#[derive(Debug, Clone, Copy)]
pub(crate) enum FilterKind {
    Lowpass,
    Highpass,
}

/// Nearest tabulated cutoff: 0.01..0.48 cycles/sample, 48 rows per kind.
pub(crate) fn chebyshev(cutoff: f64, kind: FilterKind) -> (&'static [f64; 5], &'static [f64; 5]) {
    let i = (cutoff * 100.0 - 1.0).round().clamp(0.0, 47.0) as usize;
    match kind {
        FilterKind::Lowpass => (&CHEBY_L_B[i], &CHEBY_L_A[i]),
        FilterKind::Highpass => (&CHEBY_H_B[i], &CHEBY_H_A[i]),
    }
}

/// Causal direct-form I with zero initial conditions; a[0] must be nonzero.
pub(crate) fn filter(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for i in 0..x.len() {
        let feed: f64 = (0..b.len().min(i + 1)).map(|k| b[k] * x[i - k]).sum();
        let feedback: f64 = (1..a.len().min(i + 1)).map(|k| a[k] * y[i - k]).sum();
        y[i] = (feed - feedback) / a[0];
    }
    y
}

/// Forward/reverse zero-state filtering, matching ciglet (no reflected padding).
pub(crate) fn filtfilt(b: &[f64], a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = filter(b, a, x);
    y.reverse();
    y = filter(b, a, &y);
    y.reverse();
    y
}

pub(crate) fn bandpass(x: &[f64], low: f64, high: f64) -> Vec<f64> {
    let mut y = x.to_vec();
    if low > 0.0 {
        let (b, a) = chebyshev(low, FilterKind::Highpass);
        y = filtfilt(b, a, &y);
    }
    if high < 0.5 {
        let (b, a) = chebyshev(high, FilterKind::Lowpass);
        y = filtfilt(b, a, &y);
    }
    y
}

/// Cumulative-sum moving average with endpoint extension. Keeps ciglet's
/// fractional cumulative-axis convention, used by CheapTrick; integer widths
/// give a conventional box average. Half-width must be nonnegative.
pub(crate) fn moving_average(x: &[f64], half_width: f64) -> Vec<f64> {
    if x.is_empty() || half_width == 0.0 {
        return x.to_vec();
    }
    let pad = half_width as usize;
    let mut prefix = Vec::with_capacity(x.len() + 2 * pad);
    let mut sum = 0.0;
    for i in 0..x.len() + 2 * pad {
        sum += x[i.saturating_sub(pad).min(x.len() - 1)];
        prefix.push(sum);
    }
    let upper: Vec<_> = (0..x.len()).map(|i| i as f64 + half_width).collect();
    let lower: Vec<_> = (0..x.len()).map(|i| i as f64 - half_width).collect();
    let end = (x.len() + pad) as f64;
    let upper = super::interp::interp1u(-half_width, end, &prefix, &upper);
    let lower = super::interp::interp1u(-half_width, end, &prefix, &lower);
    upper
        .iter()
        .zip(lower)
        .map(|(a, b)| (a - b) / (2.0 * half_width))
        .collect()
}

pub(crate) struct KalmanOutput {
    pub(crate) values: Vec<f64>,
    pub(crate) variances: Vec<f64>,
    pub(crate) log_likelihood: f64,
}

/// Random-walk state, time-varying process Q and measurement R variances.
pub(crate) fn kalman_filter(z: &[f64], q: &[f64], r: &[f64], initial: f64) -> KalmanOutput {
    let mut out = KalmanOutput {
        values: Vec::with_capacity(z.len()),
        variances: Vec::with_capacity(z.len()),
        log_likelihood: 0.0,
    };
    let mut state = initial;
    let mut p = 0.0;
    for t in 0..z.len() {
        let prediction = p + q[t];
        let innovation = z[t] - state;
        let variance = prediction + r[t];
        let gain = prediction / variance;
        state += gain * innovation;
        p = prediction * (1.0 - gain);
        out.values.push(state);
        out.variances.push(p);
        out.log_likelihood -=
            0.5 * ((std::f64::consts::TAU * variance).ln() + innovation * innovation / variance);
    }
    out
}

pub(crate) fn rts_smooth(filtered: &[f64], p: &[f64], q: &[f64]) -> Vec<f64> {
    let mut x = filtered.to_vec();
    for t in (1..x.len()).rev() {
        let gain = p[t - 1] / (p[t - 1] + q[t]);
        x[t - 1] = filtered[t - 1] + gain * (x[t] - filtered[t - 1]);
    }
    x
}

/// v2.1 impulse-resistant local mean (order >= 1).
pub(crate) fn smoothing_filter(x: &[f64], order: usize) -> Vec<f64> {
    if x.len() < order {
        return x.to_vec();
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mut y = vec![0.0; x.len()];
    let half = order / 2;
    y[..half].fill(mean(&x[..order]));
    y[x.len() - half..].fill(mean(&x[x.len() - order..]));
    for i in half..x.len() - half {
        let slice = &x[i - half..i - half + order];
        let m = mean(slice);
        let pos = slice.iter().filter(|&&v| v >= m).count() as f64;
        let neg = slice.iter().filter(|&&v| v <= m).count() as f64;
        let total: f64 = slice.iter().map(|v| (v - m).max(0.0)).sum();
        y[i] = m + (pos - neg) * total / (order * order) as f64;
    }
    y
}
