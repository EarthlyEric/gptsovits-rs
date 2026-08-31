use anyhow::{anyhow, Result};
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Write};
use std::process::{Command, Stdio};
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    #[default]
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl AudioFormat {
    pub fn content_type(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Opus => "audio/opus",
            AudioFormat::Aac => "audio/aac",
            AudioFormat::Flac => "audio/flac",
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Pcm => "application/octet-stream",
        }
    }
}

/// Peak normalization to prevent digital clipping / harsh distortion (max target amplitude = 0.98)
fn normalize_peak(samples: &[f32]) -> Vec<f32> {
    let max_amp = samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_amp > 0.98 {
        let scale = 0.98 / max_amp;
        samples.iter().map(|&s| s * scale).collect()
    } else {
        samples.to_vec()
    }
}

/// Encode single-channel f32 audio samples to the requested format
pub fn encode_audio(samples: &[f32], sample_rate: u32, format: AudioFormat) -> Result<Vec<u8>> {
    let normalized = normalize_peak(samples);
    match format {
        AudioFormat::Pcm => encode_pcm_s16le(&normalized),
        AudioFormat::Wav => encode_wav(&normalized, sample_rate),
        AudioFormat::Mp3 => encode_ffmpeg(
            &normalized,
            sample_rate,
            "mp3",
            &["-c:a", "libmp3lame", "-q:a", "2"],
        ),
        AudioFormat::Opus => encode_ffmpeg(
            &normalized,
            sample_rate,
            "opus",
            &["-c:a", "libopus", "-b:a", "128k"],
        ),
        AudioFormat::Aac => encode_ffmpeg(
            &normalized,
            sample_rate,
            "adts",
            &["-c:a", "aac", "-b:a", "192k"],
        ),
        AudioFormat::Flac => encode_ffmpeg(&normalized, sample_rate, "flac", &["-c:a", "flac"]),
    }
}

/// Pure Rust 16-bit PCM (s16le) encoder
pub fn encode_pcm_s16le(samples: &[f32]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * 32767.0) as i16;
        bytes.extend_from_slice(&sample_i16.to_le_bytes());
    }
    Ok(bytes)
}

/// Pure Rust WAV encoder
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)
            .map_err(|e| anyhow!("Failed to create WavWriter: {}", e))?;

        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let sample_i16 = (clamped * 32767.0) as i16;
            writer
                .write_sample(sample_i16)
                .map_err(|e| anyhow!("Failed to write WAV sample: {}", e))?;
        }
        writer
            .finalize()
            .map_err(|e| anyhow!("Failed to finalize WAV: {}", e))?;
    }

    Ok(cursor.into_inner())
}

/// Encode using system ffmpeg for MP3, AAC, OPUS, FLAC
fn encode_ffmpeg(
    samples: &[f32],
    sample_rate: u32,
    out_format: &str,
    extra_args: &[&str],
) -> Result<Vec<u8>> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-f",
        "s16le",
        "-ar",
        &sample_rate.to_string(),
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]);
    cmd.args(extra_args);
    cmd.args(["-f", out_format, "pipe:1"]);
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    let mut child = cmd
        .spawn()
        .map_err(|e| anyhow!("Failed to spawn ffmpeg for audio encoding: {}", e))?;

    let pcm_bytes = encode_pcm_s16le(samples)?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&pcm_bytes)?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| anyhow!("ffmpeg encoding error: {}", e))?;

    if !output.status.success() {
        // If ffmpeg fails, fallback to WAV
        return encode_wav(samples, sample_rate);
    }

    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_wav() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = encode_wav(&samples, 32000).unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], b"RIFF");
    }

    #[test]
    fn test_encode_pcm() {
        let samples = vec![0.0f32, 1.0, -1.0];
        let bytes = encode_pcm_s16le(&samples).unwrap();
        assert_eq!(bytes.len(), 6);
    }
}
