# libllsm2-rs

A safe, idiomatic Rust rewrite of **libllsm2 v2.1** (Low Level Speech Model) for high-quality speech analysis, modification, parametric coding, and real-time streaming synthesis.

- **Pure Safe Rust Core**: `crates/llsm2` enforces `#![forbid(unsafe_code)]` with uniform `f64` DSP precision.
- **v2.1 Complete Feature Set**: Layer 0 (Harmonic + Noise) and Layer 1 (Source-Tract separation with LF glottal model, CheapTrick envelope, Harmonic + Pulse-by-Pulse synthesis), `Coder`, and lock-free `RtSynthesizer`.
- **C99 / C++ ABI**: `crates/llsm2-ffi` provides opaque-handle `cdylib`/`staticlib` bindings with a checked-in `cbindgen` header (`crates/llsm2-ffi/include/llsm2.h`).
- **License**: `GPL-3.0-or-later` (see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

---

## Workspace Structure

```text
.
├── Cargo.toml
├── LICENSE
├── THIRD_PARTY_NOTICES.md
└── crates/
    ├── llsm2/                     # Safe Rust core library (#![forbid(unsafe_code)])
    │   ├── src/
    │   │   ├── config.rs          # AnalysisOptions, SynthesisOptions, ModelConfig
    │   │   ├── frame.rs           # Chunk, Frame, HarmonicFrame, NoiseFrame, Layer1Data, phase ops
    │   │   ├── layer0.rs          # Layer 0 analysis & offline synthesis + STFT NoiseFilter
    │   │   ├── layer1.rs          # Source-tract decomposition, LF GlottalFlowParams, PbP synthesis
    │   │   ├── modify.rs          # Frame interpolation (time-stretching & crossfading)
    │   │   ├── coder.rs           # Fixed-dimension statistical feature Coder
    │   │   ├── rtsynth/           # Lock-free deterministic RtSynthesizer & ring buffers
    │   │   └── dsp/               # Built-in pure-Rust DSP (FFT/CZT, filters, LF, spectral, noise)
    │   ├── examples/
    │   │   ├── demo.rs            # End-to-end demos (roundtrip, pitch, stretch, coder, growl, stream)
    │   │   └── rt_profile.rs      # Allocation and per-frame latency profiler
    │   └── tests/                 # Unit, statistical acceptance, allocation & speech fixtures
    └── llsm2-ffi/                 # C99 ABI crate (cdylib / staticlib)
        ├── include/llsm2.h        # Generated C99/C++17 header
        ├── src/                   # Opaque handles, views, panic guards, C callbacks
        └── examples/stream.c      # C99/C++17 smoke test & usage example
```

---

## Core Capabilities

### 1. Two-Layer Speech Modeling
- **Layer 0 (Harmonic + Noise Model)**:
  - Harmonic analysis via **Peak-Picking** (QIFFT + circular phase interpolation) or **Chirp-Z Transform (CZT)**, plus Flanagan instantaneous-frequency F0 refinement (`refine_f0`).
  - Noise modeling with Kalman + RTS smoothing separating steady PSD (`nm.psd`) from v2.1 per-frame fluctuations (`nm.psd_residual`), plus multi-band subband temporal envelopes (`nm.edc`, `nm.eenv`).
  - Phase utilities: `phase_shift`, `phase_sync_rps` (Relative Phase Shift), and `phase_propagate` (`Forward` / `Backward`).
- **Layer 1 (Source-Tract Decomposition & Pulse-by-Pulse Synthesis)**:
  - Fits glottal shape parameter $R_d \in [0.02, 3.0]$ via Itakura–Saito distance over an LF cache, followed by trajectory interpolation and anti-impulse smoothing.
  - Separates vocal tract log-magnitude envelope (`vt_magn`, CheapTrick) and glottal source phase (`vs_phse`).
  - Seamless per-frame switching between **Harmonic Model (HM)** and **Pulse-by-Pulse (PbP)** synthesis with 3-period hysteresis crossfading.
  - Stateful or closure-based **`PbpEffect`** callbacks (`GlottalFlowParams { fa, rk, rg, t0, ee }` + pulse timing jitter in seconds).

### 2. Parametric Feature Codec (`Coder`)
- Encodes and decodes Layer 1 frames into compact fixed-dimension vectors for statistical or neural vocoding:
  `[VUV, F0, Rd, MelSpec(order_spec), BAP(order_bap)]`
- `Coder` is `Send + Sync` and exposes `&self` lock-free encoding/decoding methods (`encode`, `encode_into`, `decode_layer0`, `decode_layer1`).
- Supports arbitrary positive spectral/BAP orders and non-power-of-two FFT sizes.

### 3. Deterministic Streaming Synthesis (`RtSynthesizer`)
- Single-threaded, lock-free, `Send` state machine supporting both Layer 0 and Layer 1 (including `PbpEffect` callbacks).
- **Zero-allocation steady state** for Layer 0 (`feed` / `read` / `read_decomposed` / `clear` reuse pre-allocated scratch buffers and FFT plans).
- Atomic `LlsmError::BufferFull` checking before mutating internal state or invoking pulse callbacks.
- Exact sample clock tracking across fractional hop sizes (`round((i + 1) * thop * fs) - round(i * thop * fs)`).

### 4. SIMD & Performance
- Default `simd` Cargo feature uses [`wide`](https://github.com/Lokathor/wide) (safe Rust) to accelerate harmonic oscillator banks, overlap-add accumulation, and ring-buffer mixing.
- Disable with `--no-default-features` for pure scalar builds.

---

## Quick Start (Rust)

Add `llsm2` to your `Cargo.toml`:

```toml
[dependencies]
llsm2 = { path = "crates/llsm2" }
```

### Layer 0 / Layer 1 Analysis, Pitch Shifting, and Synthesis

```rust
use llsm2::{
    AnalysisOptions, LlsmError, PhaseDirection, SynthesisMode, SynthesisOptions,
    analyze, synthesize,
};
use std::f64::consts::TAU;

fn main() -> Result<(), LlsmError> {
    let fs = 44100.0;
    let x: Vec<f64> = (0..22050)
        .map(|i| 0.1 * (TAU * 200.0 * i as f64 / fs).cos())
        .collect();
    let f0 = vec![200.0; 172]; // External F0 contour in Hz (0.0 for unvoiced)

    let analysis_opts = AnalysisOptions {
        thop: 128.0 / fs,
        ..AnalysisOptions::default()
    };
    let mut chunk = analyze(&analysis_opts, &x, fs, &f0)?.chunk;

    // Convert to Layer 1 for source-tract pitch shifting (1.5x)
    chunk.to_layer1(2048)?;
    chunk.phase_sync_rps(true);
    chunk.clear_layer0();

    let factor = 1.5;
    for frame in &mut chunk.frames {
        frame.f0 = (frame.f0 * factor).min(chunk.config.fnyq);
        if let Some(layer1) = &mut frame.layer1 {
            for db in &mut layer1.vt_magn {
                *db -= 20.0 * factor.log10();
            }
        }
    }
    chunk.phase_propagate(PhaseDirection::Forward);

    let synth_opts = SynthesisOptions {
        mode: SynthesisMode::Layer1,
        ..SynthesisOptions::new(fs)
    };
    let out = synthesize(&synth_opts, &chunk)?;
    assert!(out.y.iter().all(|v| v.is_finite()));
    Ok(())
}
```

### Real-Time Streaming Synthesis

```rust
use llsm2::{Chunk, RtSynthesizer, SynthesisOptions};

fn stream_chunk(options: SynthesisOptions, chunk: &Chunk) -> Result<Vec<f64>, llsm2::LlsmError> {
    let mut rt = RtSynthesizer::new(options, chunk.config.clone(), 4096)?;
    let delay = rt.latency();
    let mut out = Vec::new();
    let mut buf = [0.0; 256];

    for frame in &chunk.frames {
        rt.feed(frame)?;
        loop {
            let n = rt.read(&mut buf);
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
    }
    // Feed valid silent frames at end-of-stream to flush `delay` + 1 OLA tail frame,
    // then slice `&out[delay..]`.
    Ok(out)
}
```

---

## C99 / C++ ABI (`llsm2-ffi`)

Build the shared and static libraries:

```text
cargo build --release -p llsm2-ffi
```

Include [`crates/llsm2-ffi/include/llsm2.h`](crates/llsm2-ffi/include/llsm2.h) and link against `llsm2_ffi`. All calls return `LlsmStatus`, guard Rust panics at the ABI boundary, and expose diagnostics via `llsm2_last_error_message()`. See [`crates/llsm2-ffi/examples/stream.c`](crates/llsm2-ffi/examples/stream.c) for full ownership rules and callback examples.

---

## Running Examples & Tests

Run the interactive speech demos:

```text
cargo run --release -p llsm2 --example demo -- roundtrip
cargo run --release -p llsm2 --example demo -- pitch
cargo run --release -p llsm2 --example demo -- stretch
cargo run --release -p llsm2 --example demo -- coder
cargo run --release -p llsm2 --example demo -- growl
cargo run --release -p llsm2 --example demo -- stream
```

Run the full verification suite:

```text
cargo test --workspace
cargo test --workspace --release
cargo test --workspace --no-default-features
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --workspace --no-deps
```

---

## Key Conventions & Notes

- **F0 Input**: `analyze` requires a caller-supplied F0 contour (`0.0` for unvoiced, or finite `MIN_VOICED_F0 (10.0) <= f0 <= fs / 2.0`). `llsm2` does not ship a production pitch tracker in its public API.
- **Output Length**: Offline synthesis produces `round((num_frames + 1) * thop * output_fs)` samples (including the final overlap-add tail).
- **Cross-Rate Synthesis**: Harmonics above `output_fs / 2.0` are omitted and noise shaping zeros bins above source `config.fnyq` to avoid high-frequency artifacts during upsampling.
- **v2.1 Scope**: Deprecated v2.0 symbols (`LLSM_CONF_NOSWARP`, frequency-warped PSDs, legacy `void*` container bags, and internal mutexes) are intentionally omitted.
