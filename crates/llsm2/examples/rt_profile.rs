//! Reproducible allocation/latency probe. Run with --release on an idle machine.
use llsm2::{
    AnalysisOptions, Chunk, HarmonicFrame, Layer1Data, PhaseDirection, RtSynthesizer,
    SynthesisMode, SynthesisOptions,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed},
    time::Instant,
};

struct CountingAllocator;
static ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENABLED.load(Relaxed) {
            ALLOCS.fetch_add(1, Relaxed);
            BYTES.fetch_add(layout.size(), Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if ENABLED.load(Relaxed) {
            ALLOCS.fetch_add(1, Relaxed);
            BYTES.fetch_add(size, Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn main() {
    let frames = 1000;
    for (label, layer1, pbp, threshold) in [
        ("layer0-recursive", false, false, None),
        ("layer0-czt", false, false, Some((0.275, 2.26))),
        ("layer1-harmonic", true, false, Some((0.275, 2.26))),
        ("layer1-pbp", true, true, Some((0.275, 2.26))),
    ] {
        let fs = 44100.0;
        let mut conf = AnalysisOptions {
            thop: 128.0 / fs,
            ..AnalysisOptions::default()
        }
        .to_config(fs)
        .unwrap();
        if layer1 {
            conf.nspec = Some(1025);
        }
        let mut chunk = Chunk::new(conf.clone(), frames + 100);
        for f in &mut chunk.frames {
            f.f0 = 200.0;
            f.rd = Some(1.0);
            f.pbp_enabled = pbp;
            f.nm.psd.fill(-40.0);
            f.nm.edc.fill(0.01);
            if layer1 {
                f.layer1 = Some(Layer1Data {
                    vt_magn: vec![-20.0; 1025],
                    vs_phse: vec![0.0; 100],
                });
            } else {
                f.hm = Some(HarmonicFrame {
                    ampl: (1..=100).map(|k| 0.1 / k as f64).collect(),
                    phse: vec![0.0; 100],
                });
            }
        }
        chunk.phase_propagate(PhaseDirection::Forward);
        let options = SynthesisOptions {
            mode: if layer1 {
                SynthesisMode::Layer1
            } else {
                SynthesisMode::Layer0
            },
            iczt_threshold: threshold,
            ..SynthesisOptions::new(fs)
        };
        let mut rt = RtSynthesizer::new(options, conf, 4096).unwrap();
        let mut out = [0.0; 256];
        for f in &chunk.frames[..100] {
            rt.feed(f).unwrap();
            black_box(rt.read(&mut out));
        }
        let mut times = Vec::with_capacity(frames);
        let mut read_ns = 0;
        ALLOCS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
        ENABLED.store(true, Relaxed);
        for f in &chunk.frames[100..] {
            let at = Instant::now();
            rt.feed(black_box(f)).unwrap();
            times.push(at.elapsed().as_nanos());
            let at = Instant::now();
            black_box(rt.read(black_box(&mut out)));
            read_ns += at.elapsed().as_nanos();
        }
        ENABLED.store(false, Relaxed);
        times.sort_unstable();
        println!(
            "{label}: allocs/frame={:.2} bytes/frame={:.0} feed_median_us={:.2} feed_p99_us={:.2} read_ns/sample={:.2}",
            ALLOCS.load(Relaxed) as f64 / frames as f64,
            BYTES.load(Relaxed) as f64 / frames as f64,
            times[frames / 2] as f64 / 1000.0,
            times[frames * 99 / 100] as f64 / 1000.0,
            read_ns as f64 / (frames * 128) as f64
        );
    }
}
