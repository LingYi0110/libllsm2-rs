//! Private, fixed-capacity sample buffers. Callers establish index bounds.

pub(super) struct RingBuffer {
    data: Vec<f64>,
    cursor: usize,
}

impl RingBuffer {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            cursor: 0,
        }
    }
    pub(super) fn capacity(&self) -> usize {
        self.data.len()
    }
    pub(super) fn clear(&mut self) {
        self.data.fill(0.0);
        self.cursor = 0;
    }
    #[cfg(test)]
    pub(super) fn read(&self, offset: isize) -> f64 {
        debug_assert!((-(self.capacity() as isize)..0).contains(&offset));
        self.data[(self.cursor as isize + offset).rem_euclid(self.capacity() as isize) as usize]
    }
    pub(super) fn read_into(&self, offset: isize, dst: &mut [f64]) {
        debug_assert!(offset + dst.len() as isize <= 0);
        let (a, b) = self.slices(offset, dst.len());
        let (head, tail) = dst.split_at_mut(a.len());
        head.copy_from_slice(a);
        tail.copy_from_slice(b);
    }
    pub(super) fn slices(&self, offset: isize, len: usize) -> (&[f64], &[f64]) {
        debug_assert!(offset >= -(self.capacity() as isize) && offset + len as isize <= 0);
        let start = (self.cursor as isize + offset).rem_euclid(self.capacity() as isize) as usize;
        let first = len.min(self.capacity() - start);
        (&self.data[start..start + first], &self.data[..len - first])
    }
    pub(super) fn forward(&mut self, size: usize) {
        self.cursor = (self.cursor + size) % self.capacity();
    }
    pub(super) fn write(&mut self, offset: isize, src: &[f64]) {
        debug_assert!(offset >= -(self.capacity() as isize) && offset + src.len() as isize <= 0);
        let start = (self.cursor as isize + offset).rem_euclid(self.capacity() as isize) as usize;
        let first = src.len().min(self.capacity() - start);
        self.data[start..start + first].copy_from_slice(&src[..first]);
        self.data[..src.len() - first].copy_from_slice(&src[first..]);
    }
    pub(super) fn append(&mut self, src: &[f64]) {
        debug_assert!(src.len() <= self.capacity());
        self.forward(src.len());
        self.write(-(src.len() as isize), src);
    }
    #[cfg(test)]
    fn append_blank(&mut self, size: usize) {
        for _ in 0..size {
            self.append(&[0.0]);
        }
    }
}

/// A past and a future ring sharing one logical origin, for centered OLA.
pub(super) struct DualBuffer {
    past: Vec<f64>,
    future: Vec<f64>,
    cursor: usize,
}
impl DualBuffer {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            past: vec![0.0; capacity],
            future: vec![0.0; capacity],
            cursor: 0,
        }
    }
    pub(super) fn clear(&mut self) {
        self.past.fill(0.0);
        self.future.fill(0.0);
        self.cursor = 0;
    }
    pub(super) fn read(&self, offset: isize) -> f64 {
        debug_assert!(offset >= -(self.past.len() as isize) && offset < self.future.len() as isize);
        let k = (self.cursor as isize + offset).rem_euclid(self.past.len() as isize) as usize;
        if offset < 0 {
            self.past[k]
        } else {
            self.future[k]
        }
    }
    pub(super) fn add(&mut self, offset: isize, src: &[f64]) {
        debug_assert!(
            offset >= -(self.past.len() as isize)
                && offset + src.len() as isize <= self.future.len() as isize
        );
        let mut i = 0;
        while i < src.len() {
            let relative = offset + i as isize;
            let k = (self.cursor as isize + relative).rem_euclid(self.past.len() as isize) as usize;
            let mut count = (src.len() - i).min(self.past.len() - k);
            if relative < 0 {
                count = count.min((-relative) as usize);
                crate::dsp::vector::add(&mut self.past[k..k + count], &src[i..i + count]);
            } else {
                crate::dsp::vector::add(&mut self.future[k..k + count], &src[i..i + count]);
            }
            i += count;
        }
    }
    pub(super) fn forward(&mut self, size: usize) {
        debug_assert!(size <= self.past.len());
        let mut left = size;
        while left > 0 {
            let count = left.min(self.past.len() - self.cursor);
            let range = self.cursor..self.cursor + count;
            self.past[range.clone()].copy_from_slice(&self.future[range.clone()]);
            self.future[range].fill(0.0);
            self.cursor = (self.cursor + count) % self.past.len();
            left -= count;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, VecDeque};
    #[test]
    fn ring_block_scalar_forward_and_zero_operations_match_history() {
        let mut a = RingBuffer::new(4096);
        let mut b = RingBuffer::new(4096);
        let mut history = VecDeque::from(vec![0.0; 4096]);
        for block in 0..100 {
            let x: Vec<_> = (0..100).map(|j| (block * 100 + j) as f64 / 100.0).collect();
            a.append(&x);
            a.append_blank(100);
            b.forward(100);
            b.write(-100, &x);
            for _ in 0..100 {
                b.append(&[0.0]);
            }
            for &v in x.iter().chain(&vec![0.0; 100]) {
                history.pop_front();
                history.push_back(v);
            }
            let mut chunk = vec![0.0; 4096];
            a.read_into(-4096, &mut chunk);
            assert_eq!(chunk, history.iter().copied().collect::<Vec<_>>());
            for j in -4096..0 {
                assert_eq!(a.read(j), b.read(j));
            }
        }
        a.append(&[]);
        a.write(0, &[]);
        a.read_into(0, &mut []);
        a.clear();
        assert!((-4096..0).all(|i| a.read(i) == 0.0));
    }
    #[test]
    fn dual_overlap_add_and_advance_match_an_absolute_sample_timeline() {
        let mut buffer = DualBuffer::new(64);
        let mut reference = BTreeMap::<isize, f64>::new();
        let mut center = 0;
        for block in 0..100 {
            let offset = -23;
            let samples: Vec<_> = (0..54).map(|j| ((block + j) as f64).sin()).collect();
            buffer.add(offset, &samples);
            for (j, &v) in samples.iter().enumerate() {
                *reference.entry(center + offset + j as isize).or_default() += v;
            }
            for i in -64..64 {
                assert_eq!(
                    buffer.read(i),
                    *reference.get(&(center + i)).unwrap_or(&0.0)
                );
            }
            let step = 3 + block % 13;
            buffer.forward(step);
            center += step as isize;
            reference.retain(|&i, _| i >= center - 64);
        }
        buffer.add(0, &[]);
        buffer.forward(0);
        buffer.clear();
        assert!((-64..64).all(|i| buffer.read(i) == 0.0));
    }
}
