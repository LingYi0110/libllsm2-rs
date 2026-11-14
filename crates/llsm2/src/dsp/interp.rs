use std::f64::consts::{PI, TAU};

pub(crate) fn wrap(x: f64) -> f64 {
    (x + PI).rem_euclid(TAU) - PI
}
pub(crate) fn circular(a: f64, b: f64, t: f64) -> f64 {
    wrap(a + wrap(b - a) * t)
}

/// Strictly increasing source axis; endpoint hold; queries can be unordered.
pub(crate) fn interp1(axis: &[f64], values: &[f64], queries: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; queries.len()];
    interp1_into(axis, values, queries, &mut out);
    out
}
pub(crate) fn interp1_into(axis: &[f64], values: &[f64], queries: &[f64], out: &mut [f64]) {
    for (&q, v) in queries.iter().zip(out) {
        let i = axis.partition_point(|&v| v <= q);
        *v = if i == 0 {
            values[0]
        } else if i == axis.len() {
            values[i - 1]
        } else {
            let t = (q - axis[i - 1]) / (axis[i] - axis[i - 1]);
            values[i - 1] + t * (values[i] - values[i - 1])
        };
    }
}

/// Uniform source coordinates are start + j*(end-start)/N (end is exclusive).
pub(crate) fn interp1u(start: f64, end: f64, values: &[f64], queries: &[f64]) -> Vec<f64> {
    queries
        .iter()
        .map(|&q| {
            let p = ((q - start) * values.len() as f64 / (end - start))
                .clamp(0.0, (values.len() - 1) as f64);
            let i = p as usize;
            if i + 1 == values.len() {
                values[i]
            } else {
                values[i] + (p - i as f64) * (values[i + 1] - values[i])
            }
        })
        .collect()
}

/// Fill missing values linearly with held endpoints. All-blank input stays blank.
pub(crate) fn fill_blanks(x: &[f64], blank: f64) -> Vec<f64> {
    let present = |v: f64| {
        if blank.is_nan() {
            !v.is_nan()
        } else {
            v != blank
        }
    };
    let indices: Vec<_> = x
        .iter()
        .enumerate()
        .filter(|(_, v)| present(**v))
        .map(|(i, _)| i as f64)
        .collect();
    if indices.is_empty() {
        return x.to_vec();
    }
    let values: Vec<_> = indices.iter().map(|&i| x[i as usize]).collect();
    interp1(
        &indices,
        &values,
        &(0..x.len()).map(|i| i as f64).collect::<Vec<_>>(),
    )
}

/// Three-point parabolic interpolation of a log spectrum at an interior peak.
pub(crate) fn qifft(x: &[f64], k: usize) -> (f64, f64) {
    let a = (x[k - 1] + x[k + 1]) * 0.5 - x[k];
    let b = (x[k + 1] - x[k - 1]) * 0.5;
    let delta = if a == 0.0 { 0.0 } else { -b / (2.0 * a) };
    let delta = if delta.abs() < 1.0 { delta } else { 0.0 };
    (
        k as f64 + delta,
        (x[k] + b * delta + a * delta * delta).min(x[k] + 0.2),
    )
}
