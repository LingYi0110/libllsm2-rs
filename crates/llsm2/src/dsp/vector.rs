//! Safe stable-Rust SIMD, with an explicitly buildable scalar implementation.
pub(crate) fn sum(dst: &mut [f64], a: &[f64], b: &[f64]) {
    #[cfg(feature = "simd")]
    {
        let (dst, tail) = dst.as_chunks_mut::<4>();
        let (a, at) = a.as_chunks::<4>();
        let (b, bt) = b.as_chunks::<4>();
        for ((dst, a), b) in dst.iter_mut().zip(a).zip(b) {
            *dst = (wide::f64x4::new(*a) + wide::f64x4::new(*b)).to_array();
        }
        for ((v, a), b) in tail.iter_mut().zip(at).zip(bt) {
            *v = a + b;
        }
    }
    #[cfg(not(feature = "simd"))]
    for ((v, a), b) in dst.iter_mut().zip(a).zip(b) {
        *v = a + b;
    }
}
pub(crate) fn add(dst: &mut [f64], src: &[f64]) {
    #[cfg(feature = "simd")]
    {
        let (dst, tail) = dst.as_chunks_mut::<4>();
        let (src, st) = src.as_chunks::<4>();
        for (dst, src) in dst.iter_mut().zip(src) {
            *dst = (wide::f64x4::new(*dst) + wide::f64x4::new(*src)).to_array();
        }
        for (v, src) in tail.iter_mut().zip(st) {
            *v += src;
        }
    }
    #[cfg(not(feature = "simd"))]
    for (v, src) in dst.iter_mut().zip(src) {
        *v += src;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn packed_sum_and_add_keep_all_tails_and_match_scalar_arithmetic() {
        for n in 0..35 {
            let a: Vec<_> = (0..n).map(|i| (i as f64 - 17.0) * 0.123).collect();
            let b: Vec<_> = (0..n).map(|i| -(i as f64 * 0.071)).collect();
            let expected: Vec<_> = a.iter().zip(&b).map(|(a, b)| a + b).collect();
            let mut dst = vec![0.0; n];
            super::sum(&mut dst, &a, &b);
            assert_eq!(dst, expected);
            let mut dst = a.clone();
            super::add(&mut dst, &b);
            assert_eq!(dst, expected);
        }
    }
}
