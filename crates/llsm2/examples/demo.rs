//! Runnable demos for llsm2 Layer 0/1 analysis, modification, coding, and streaming synthesis.
mod common;

use common::Result;
use llsm2::{
    AnalysisOptions, Chunk, Coder, Frame, GlottalFlowParams, NoiseFrame, PbpEffect, PhaseDirection,
    PulseContext, RtSynthesizer, SynthesisMode, SynthesisOptions, SynthesisOutput, analyze,
    interpolate_frames, synthesize,
};
use std::f64::consts::TAU;
use std::path::{Path, PathBuf};
use std::time::Instant;

const HOP: usize = 128;

fn save(dir: &Path, name: &str, output: &SynthesisOutput) -> Result<()> {
    for (suffix, samples) in [
        ("", &output.y),
        ("-periodic", &output.y_sin),
        ("-noise", &output.y_noise),
    ] {
        common::write_wav(
            &dir.join(format!("{name}{suffix}.wav")),
            samples,
            output.fs as u32,
        )?;
    }
    Ok(())
}

fn layer1(chunk: &mut Chunk, all_pulses: bool) -> Result<()> {
    chunk.to_layer1(2048)?;
    chunk.phase_sync_rps(true);
    chunk.clear_layer0();
    for (i, frame) in chunk.frames.iter_mut().enumerate() {
        frame.pbp_enabled = all_pulses || (i / 50) % 2 == 1;
    }
    chunk.phase_propagate(PhaseDirection::Forward);
    Ok(())
}

fn stretch(mut source: Chunk, factor: f64) -> Result<Chunk> {
    source.to_layer1(2048)?;
    source.phase_propagate(PhaseDirection::Backward);
    source.clear_layer0();
    let n = source.num_frames();
    let count = (n as f64 * factor).round().max(1.0) as usize;
    let mut frames = Vec::with_capacity(count);
    let mut random = 91u64;
    for i in 0..count {
        let at = (i as f64 / factor).min((n - 1) as f64);
        let left = at as usize;
        let right = (left + 1).min(n - 1);
        let mut frame = interpolate_frames(
            &source.config,
            &source.frames[left],
            &source.frames[right],
            at - left as f64,
        )?;
        // Sample local stochastic residuals instead of stretching the same
        // fluctuation across repeated frames.
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let neighbor =
            (left as isize + (random % 5) as isize - 2).clamp(0, (n - 1) as isize) as usize;
        frame.nm.psd_residual = source.frames[neighbor].nm.psd_residual.clone();
        frames.push(frame);
    }
    let mut output = Chunk {
        config: source.config,
        frames,
    };
    output.phase_propagate(PhaseDirection::Forward);
    output.to_layer0()?;
    Ok(output)
}

struct Growl {
    thop: f64,
    duration: f64,
    calls: usize,
}
impl PbpEffect for Growl {
    fn modify(&mut self, g: &mut GlottalFlowParams, ctx: &PulseContext<'_>) -> f64 {
        self.calls += 1;
        let time = ctx.frame_index as f64 * self.thop;
        let start = 0.2 * self.duration;
        let end = 0.85 * self.duration;
        let fade = 0.04 * self.duration;
        let envelope =
            ((time - start) / fade).clamp(0.0, 1.0) * ((end - time) / fade).clamp(0.0, 1.0);
        let lfo = (TAU * ctx.pulse_index as f64 / 6.0
            + (TAU * ctx.pulse_index as f64 / 50.0).sin())
        .sin();
        g.fa *= 1.0 + 0.16 * lfo * envelope;
        g.rk *= 1.0 + 0.04 * lfo * envelope;
        g.ee *= 1.0 - 0.18 * lfo * envelope;
        0.012 * g.t0 * (ctx.pulse_index as f64 * 2.39996).sin() * envelope
    }
}

fn stream(
    chunk: &Chunk,
    options: SynthesisOptions,
    mut effect: Option<&mut dyn PbpEffect>,
) -> Result<SynthesisOutput> {
    let hop = chunk.config.thop * options.fs;
    let wanted = ((chunk.num_frames() + 1) as f64 * hop).round() as usize;
    let mut rt = RtSynthesizer::new(options.clone(), chunk.config.clone(), 4096)?;
    let latency = rt.latency();
    println!(
        "Streaming latency: {latency} samples ({:.3} s)",
        latency as f64 / options.fs
    );
    let mut noise = NoiseFrame::new(
        chunk.config.num_channels(),
        chunk.config.max_nhar_env,
        chunk.config.npsd,
    );
    noise.edc.fill(0.0);
    let silence = Frame {
        f0: 0.0,
        rd: Some(1.0),
        hm: None,
        layer1: None,
        nm: noise,
        pbp_enabled: false,
    };
    let padding = ((latency as f64 + 3.0 * hop) / hop).ceil() as usize + 3;
    let mut periodic = Vec::new();
    let mut aperiodic = Vec::new();
    let mut p = [0.0; 512];
    let mut ap = [0.0; 512];
    for frame in chunk
        .frames
        .iter()
        .chain(std::iter::repeat_n(&silence, padding))
    {
        match effect.as_deref_mut() {
            Some(effect) => rt.feed_with_effect(frame, effect)?,
            None => rt.feed(frame)?,
        }
        loop {
            let n = rt.read_decomposed(&mut p, &mut ap);
            if n == 0 {
                break;
            }
            periodic.extend_from_slice(&p[..n]);
            aperiodic.extend_from_slice(&ap[..n]);
        }
    }
    let range = latency..latency + wanted;
    let y_sin = periodic[range.clone()].to_vec();
    let y_noise = aperiodic[range].to_vec();
    Ok(SynthesisOutput {
        y: y_sin.iter().zip(&y_noise).map(|(p, ap)| p + ap).collect(),
        y_sin,
        y_noise,
        fs: options.fs,
    })
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let usage = "demo <roundtrip|pitch|stretch|coder|growl|stream|f0> [input.wav] [output_dir] [factor]\n\
                 defaults: bundled fixture speech, target/demo/<mode>, pitch=1.5, stretch=2.0";
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        println!("{usage}");
        return Ok(());
    }
    let mode = args[0].as_str();
    if ![
        "roundtrip",
        "pitch",
        "stretch",
        "coder",
        "growl",
        "stream",
        "f0",
    ]
    .contains(&mode)
        || args.len() > 4
    {
        return Err(usage.into());
    }
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.join("../..");
    let input = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        manifest_dir
            .join("tests/fixtures/audio")
            .join(if mode == "growl" {
                "are-you-ready.wav"
            } else {
                "arctic_a0001.wav"
            })
    });
    let dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root.join("target/demo").join(mode));
    let factor = args
        .get(3)
        .map(|v| v.parse::<f64>())
        .transpose()?
        .unwrap_or(if mode == "stretch" { 2.0 } else { 1.5 });
    if !factor.is_finite() || factor <= 0.0 {
        return Err("factor must be finite and positive".into());
    }
    let (audio, fs) = common::read_wav(&input)?;
    std::fs::create_dir_all(&dir)?;
    println!(
        "Input: {} ({fs} Hz, {:.3} s)",
        input.display(),
        audio.len() as f64 / fs as f64
    );
    let thop = HOP as f64 / fs as f64;
    let start = Instant::now();
    let f0 = common::estimate_f0_yin(&audio, fs, HOP);
    println!(
        "YIN F0: {} frames ({} voiced) in {:.3} s",
        f0.len(),
        f0.iter().filter(|&&v| v > 0.0).count(),
        start.elapsed().as_secs_f64(),
    );
    common::write_f0_csv(&dir.join("f0.csv"), &f0, thop)?;
    if mode == "f0" {
        println!("Saved {}", dir.join("f0.csv").display());
        return Ok(());
    }
    common::write_wav(&dir.join("input.wav"), &audio, fs)?;
    let mut analysis_options = AnalysisOptions {
        thop,
        ..AnalysisOptions::default()
    };
    analysis_options.chan_freqs.retain(|&f| f < fs as f64 / 2.0);
    let start = Instant::now();
    let mut chunk = analyze(&analysis_options, &audio, fs as f64, &f0)?.chunk;
    println!(
        "Analysis: {} frames in {:.3} s",
        chunk.num_frames(),
        start.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    let options = SynthesisOptions::new(fs as f64);
    match mode {
        "roundtrip" => {
            save(&dir, "layer0", &synthesize(&options, &chunk)?)?;
            layer1(&mut chunk, false)?;
            let options = SynthesisOptions {
                mode: SynthesisMode::Layer1,
                ..options
            };
            save(&dir, "layer1", &synthesize(&options, &chunk)?)?;
        }
        "pitch" => {
            layer1(&mut chunk, false)?;
            chunk.phase_propagate(PhaseDirection::Backward);
            for frame in &mut chunk.frames {
                frame.f0 = (frame.f0 * factor).min(chunk.config.fnyq);
                if let Some(layer1) = &mut frame.layer1 {
                    for value in &mut layer1.vt_magn {
                        *value -= 20.0 * factor.log10();
                    }
                }
            }
            chunk.phase_propagate(PhaseDirection::Forward);
            let options = SynthesisOptions {
                mode: SynthesisMode::Layer1,
                ..options
            };
            save(&dir, "pitch", &synthesize(&options, &chunk)?)?;
        }
        "stretch" => {
            chunk = stretch(chunk, factor)?;
            save(&dir, "stretch", &synthesize(&options, &chunk)?)?;
        }
        "coder" => {
            chunk.to_layer1(2048)?;
            let coder = Coder::new(&chunk.config, 64, 5)?;
            let features: Vec<_> = chunk
                .frames
                .iter()
                .map(|f| coder.encode(f))
                .collect::<std::result::Result<_, _>>()?;
            let mut file =
                std::io::BufWriter::new(std::fs::File::create(dir.join("features.csv"))?);
            use std::io::Write;
            for frame in &features {
                for (i, feature) in frame.iter().enumerate() {
                    if i > 0 {
                        write!(file, ",")?;
                    }
                    write!(file, "{feature:.12}")?;
                }
                writeln!(file)?;
            }
            file.flush()?;
            for use_layer1 in [false, true] {
                let frames = features
                    .iter()
                    .map(|f| {
                        if use_layer1 {
                            coder.decode_layer1(f)
                        } else {
                            coder.decode_layer0(f)
                        }
                    })
                    .collect::<std::result::Result<_, _>>()?;
                let mut decoded = Chunk {
                    config: chunk.config.clone(),
                    frames,
                };
                if use_layer1 {
                    decoded.to_layer0()?;
                }
                decoded.phase_propagate(PhaseDirection::Forward);
                save(
                    &dir,
                    if use_layer1 {
                        "coder-layer1"
                    } else {
                        "coder-layer0"
                    },
                    &synthesize(&options, &decoded)?,
                )?;
            }
        }
        "growl" | "stream" => {
            layer1(&mut chunk, mode == "growl")?;
            let options = SynthesisOptions {
                mode: SynthesisMode::Layer1,
                ..options
            };
            let baseline = synthesize(&options, &chunk)?;
            save(&dir, "offline", &baseline)?;
            let output = if mode == "growl" {
                let mut effect = Growl {
                    thop: chunk.config.thop,
                    duration: audio.len() as f64 / fs as f64,
                    calls: 0,
                };
                let output = stream(&chunk, options, Some(&mut effect))?;
                println!("Growl modified {} pulses", effect.calls);
                output
            } else {
                let output = stream(&chunk, options, None)?;
                let error = (output
                    .y
                    .iter()
                    .zip(&baseline.y)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    / output.y.len() as f64)
                    .sqrt();
                println!("Streaming vs offline RMS difference: {error:.8}");
                output
            };
            save(&dir, mode, &output)?;
        }
        _ => unreachable!(),
    }
    println!(
        "Synthesis and output: {:.3} s",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
