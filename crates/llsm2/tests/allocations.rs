//! This binary isolates the allocator probe from unrelated test threads.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};
thread_local! {
    static ENABLED:Cell<bool>=const { Cell::new(false) };
    static COUNT:Cell<usize>=const { Cell::new(0) };
}
struct CountAlloc;
fn record() {
    if ENABLED.try_with(Cell::get).unwrap_or(false) {
        let _ = COUNT.try_with(|n| n.set(n.get() + 1));
    }
}
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

#[test]
fn layer0_feed_read_and_clear_reuse_buffers_after_construction() {
    use llsm2::{AnalysisOptions, Chunk, HarmonicFrame, RtSynthesizer, SynthesisOptions};
    for threshold in [None, Some((0.275, 2.26))] {
        let fs = 44100.0;
        let conf = AnalysisOptions {
            thop: 100.5 / fs,
            ..AnalysisOptions::default()
        }
        .to_config(fs)
        .unwrap();
        let mut frame = Chunk::new(conf.clone(), 1).frames.remove(0);
        frame.f0 = 200.0;
        frame.hm = Some(HarmonicFrame {
            ampl: vec![0.01; 100],
            phse: vec![0.0; 100],
        });
        frame.nm.psd.fill(-40.0);
        let mut rt = RtSynthesizer::new(
            SynthesisOptions {
                iczt_threshold: threshold,
                ..SynthesisOptions::new(fs)
            },
            conf,
            512,
        )
        .unwrap();
        let mut out = [0.0; 128];
        // First use is included: plans/scratch are prepared by new().
        COUNT.with(|n| n.set(0));
        ENABLED.with(|v| v.set(true));
        for i in 0..400 {
            let n = [1, 4, 25, 100][i % 4];
            let hm = frame.hm.as_mut().unwrap();
            hm.ampl.resize(n, 0.01);
            hm.phse.resize(n, 0.0);
            rt.feed(&frame).unwrap();
            rt.read(&mut out);
            if i == 200 {
                rt.clear();
            }
        }
        ENABLED.with(|v| v.set(false));
        assert_eq!(COUNT.with(Cell::get), 0, "Layer 0 hot path allocated");
    }
}
