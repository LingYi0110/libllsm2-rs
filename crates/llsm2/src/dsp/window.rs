use std::f64::consts::TAU;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Window {
    Hann,
    #[cfg(test)]
    Hamming,
    Blackman,
    #[cfg(test)]
    BlackmanHarris,
    #[cfg(test)]
    Rectangular,
}

/// Hann/Hamming are symmetric; Blackman variants are periodic, matching ciglet.
pub(crate) fn window(kind: Window, n: usize) -> Vec<f64> {
    let mut values = vec![0.0; n];
    fill_window(kind, &mut values);
    values
}

pub(crate) fn fill_window(kind: Window, values: &mut [f64]) {
    let n = values.len();
    if n == 1 {
        values[0] = 1.0;
        return;
    }
    for (i, value) in values.iter_mut().enumerate() {
        let symmetric = TAU * i as f64 / (n - 1) as f64;
        let periodic = TAU * i as f64 / n as f64;
        *value = match kind {
            Window::Hann => 0.5 - 0.5 * symmetric.cos(),
            #[cfg(test)]
            Window::Hamming => 0.54 - 0.46 * symmetric.cos(),
            Window::Blackman => 0.42 - 0.5 * periodic.cos() + 0.08 * (2.0 * periodic).cos(),
            #[cfg(test)]
            Window::BlackmanHarris => {
                0.35875 - 0.48829 * periodic.cos() + 0.14128 * (2.0 * periodic).cos()
                    - 0.01168 * (3.0 * periodic).cos()
            }
            #[cfg(test)]
            Window::Rectangular => 1.0,
        };
    }
}

/// Zero padding outside the waveform, with sample n/2 at `center`.
pub(crate) fn fetch_frame(x: &[f64], center: isize, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let index = center + i as isize - (n / 2) as isize;
            if index >= 0 && (index as usize) < x.len() {
                x[index as usize]
            } else {
                0.0
            }
        })
        .collect()
}
