use anyhow::{anyhow, Result};
use rubato::{FftFixedInOut, Resampler};
use std::path::Path;

/// Resample single-channel f32 audio samples from `from_sr` to `to_sr`
pub fn resample(samples: &[f32], from_sr: usize, to_sr: usize) -> Result<Vec<f32>> {
    if from_sr == to_sr || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    let chunk_size = 1024;
    let mut resampler = FftFixedInOut::<f32>::new(from_sr, to_sr, chunk_size, 1)
        .map_err(|e| anyhow!("Failed to create FFT resampler: {}", e))?;

    let mut output = Vec::new();
    let mut offset = 0;

    while offset < samples.len() {
        let required_in = resampler.input_frames_next();
        let end = (offset + required_in).min(samples.len());
        let mut chunk = samples[offset..end].to_vec();
        if chunk.len() < required_in {
            chunk.resize(required_in, 0.0);
        }

        let waves_in = vec![chunk];
        let waves_out = resampler
            .process(&waves_in, None)
            .map_err(|e| anyhow!("Resampling processing error: {}", e))?;

        output.extend_from_slice(&waves_out[0]);
        offset += required_in;
    }

    // Remove filter delay / phase latency
    let delay = resampler.output_delay();
    if output.len() > delay {
        output.drain(0..delay);
    }

    // Trim tail padding to exact ratio
    let expected_len = (samples.len() as f64 * to_sr as f64 / from_sr as f64) as usize;
    if output.len() > expected_len {
        output.truncate(expected_len);
    }

    Ok(output)
}

/// Load WAV audio file as single-channel f32 audio normalized to [-1.0, 1.0]
pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path.as_ref())
        .map_err(|e| anyhow!("Failed to open WAV file {:?}: {}", path.as_ref(), e))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let samples_f32: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    // Convert multi-channel to mono
    let mono_samples = if channels > 1 {
        let mut mono = Vec::with_capacity(samples_f32.len() / channels);
        for chunk in samples_f32.chunks_exact(channels) {
            let sum: f32 = chunk.iter().sum();
            mono.push(sum / channels as f32);
        }
        mono
    } else {
        samples_f32
    };

    Ok((mono_samples, sample_rate))
}
