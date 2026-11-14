use std::error::Error;
use std::io::{BufWriter, Write};
use std::path::Path;

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// Read PCM or float32 WAV, mixing multichannel audio down to mono.
pub fn read_wav(path: &Path) -> Result<(Vec<f64>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|v| v.map(f64::from))
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f64.powi(spec.bits_per_sample as i32 - 1);
            reader
                .samples::<i32>()
                .map(|v| v.map(|v| v as f64 / scale))
                .collect::<std::result::Result<_, _>>()?
        }
    };
    if samples.is_empty() || spec.sample_rate == 0 || samples.iter().any(|v| !v.is_finite()) {
        return Err("WAV must contain nonempty finite audio and a positive sample rate".into());
    }
    let audio = samples
        .chunks_exact(spec.channels as usize)
        .map(|frame| frame.iter().sum::<f64>() / spec.channels as f64)
        .collect();
    Ok((audio, spec.sample_rate))
}

/// Write mono float32 WAV without automatic normalization.
pub fn write_wav(path: &Path, audio: &[f64], fs: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: fs,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in audio {
        let sample = sample as f32;
        if !sample.is_finite() {
            return Err(format!("non-finite output sample in {}", path.display()).into());
        }
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    let peak = audio.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let rms = (audio.iter().map(|v| v * v).sum::<f64>() / audio.len() as f64).sqrt();
    println!(
        "{}: {:.3} s, peak={peak:.4}, RMS={rms:.4}",
        path.display(),
        audio.len() as f64 / fs as f64
    );
    Ok(())
}

/// Write per-frame F0 contour (Hz, 0.0 for unvoiced) to CSV.
pub fn write_f0_csv(path: &Path, f0: &[f64], thop: f64) -> Result<()> {
    let mut file = BufWriter::new(std::fs::File::create(path)?);
    writeln!(file, "time_s,f0_hz,voiced")?;
    for (i, &freq) in f0.iter().enumerate() {
        writeln!(file, "{:.6},{:.6},{}", i as f64 * thop, freq, freq > 0.0)?;
    }
    file.flush()?;
    Ok(())
}

/// Lightweight YIN pitch estimator for example playback only (not part of the library API).
/// Uses a 4x box-averaged decimation front-end and CMNDF < 0.2 threshold in 50..500 Hz.
pub fn estimate_f0_yin(x: &[f64], fs: u32, hop: usize) -> Vec<f64> {
    let decimate = (fs as f64 / 11025.0).round().max(1.0) as usize;
    let down: Vec<f64> = x
        .chunks_exact(decimate)
        .map(|chunk| chunk.iter().sum::<f64>() / decimate as f64)
        .collect();
    let ds = fs as f64 / decimate as f64;
    let minimum = (ds / 500.0).floor().max(2.0) as usize;
    let maximum = (ds / 50.0).ceil().max(minimum as f64 + 2.0) as usize;
    let size = (ds * 0.025).ceil().max(4.0) as usize;
    let sample = |i: isize| {
        if i < 0 || i as usize >= down.len() {
            0.0
        } else {
            down[i as usize]
        }
    };
    let nframes = (x.len() / hop).max(1);
    let mut contour = Vec::with_capacity(nframes);
    for frame in 0..nframes {
        let center = (frame as f64 * hop as f64 / decimate as f64).round() as isize;
        let energy = (0..size)
            .map(|j| sample(center + j as isize - (size / 2) as isize).powi(2))
            .sum::<f64>()
            / size as f64;
        if energy < 1e-6 {
            contour.push(0.0);
            continue;
        }
        let mut difference = vec![0.0; maximum + 1];
        for (tau, diff) in difference.iter_mut().enumerate().skip(1) {
            let start = center - ((size + tau) / 2) as isize;
            *diff = (0..size)
                .map(|j| (sample(start + j as isize) - sample(start + (j + tau) as isize)).powi(2))
                .sum();
        }
        let mut cmndf = vec![1.0; maximum + 1];
        let mut cumulative = 0.0;
        for tau in 1..=maximum {
            cumulative += difference[tau];
            if cumulative > 0.0 {
                cmndf[tau] = difference[tau] * tau as f64 / cumulative;
            }
        }
        let mut tau = minimum;
        while tau < maximum && cmndf[tau] >= 0.2 {
            tau += 1;
        }
        if tau >= maximum {
            contour.push(0.0);
            continue;
        }
        while tau + 1 < maximum && cmndf[tau + 1] < cmndf[tau] {
            tau += 1;
        }
        let denom = cmndf[tau - 1] - 2.0 * cmndf[tau] + cmndf[tau + 1];
        let delta = if denom.abs() > 1e-12 {
            (0.5 * (cmndf[tau - 1] - cmndf[tau + 1]) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        contour.push((ds / (tau as f64 + delta)).clamp(50.0, (fs as f64 / 2.0).min(500.0)));
    }
    contour
}
