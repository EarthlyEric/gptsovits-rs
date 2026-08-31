use anyhow::{anyhow, Result};
use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

const SAMPLE_RATE: f32 = 16_000.0;
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const FFT_SIZE: usize = 512;
const MEL_BINS: usize = 80;
const PREEMPHASIS: f32 = 0.97;
const LOW_FREQ: f32 = 20.0;
const HIGH_FREQ: f32 = 8_000.0;
const EPSILON: f32 = f32::EPSILON;

/// Compute the 80-bin log Mel filterbank used by GPT-SoVITS' SV model.
///
/// This follows `GPT_SoVITS/eres2net/kaldi.py::fbank`: 25 ms Povey-windowed
/// frames, 10 ms shift, 512-point FFT, no dithering, and no mean subtraction
/// after the filterbank projection. The returned layout is `[frames, 80]`.
pub fn kaldi_fbank(audio_16k: &[f32]) -> Result<Array2<f32>> {
    if audio_16k.len() < FRAME_LENGTH {
        return Err(anyhow!(
            "SV reference audio must contain at least {} samples at 16 kHz",
            FRAME_LENGTH
        ));
    }

    let frame_count = 1 + (audio_16k.len() - FRAME_LENGTH) / FRAME_SHIFT;
    let window = povey_window();
    let mel_filters = mel_filterbank();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut spectrum = vec![Complex::new(0.0, 0.0); FFT_SIZE];
    let mut output = Vec::with_capacity(frame_count * MEL_BINS);

    for frame_idx in 0..frame_count {
        let start = frame_idx * FRAME_SHIFT;
        let frame = &audio_16k[start..start + FRAME_LENGTH];
        let mean = frame.iter().copied().sum::<f32>() / FRAME_LENGTH as f32;

        for (i, value) in spectrum.iter_mut().enumerate() {
            *value = if i < FRAME_LENGTH {
                let centered = frame[i] - mean;
                let previous = if i == 0 {
                    centered
                } else {
                    frame[i - 1] - mean
                };
                Complex::new((centered - PREEMPHASIS * previous) * window[i], 0.0)
            } else {
                Complex::new(0.0, 0.0)
            };
        }

        fft.process(&mut spectrum);

        for filter in &mel_filters {
            let energy = filter
                .iter()
                .zip(spectrum.iter().take(FFT_SIZE / 2 + 1))
                .map(|(&weight, bin)| weight * bin.norm_sqr())
                .sum::<f32>();
            output.push(energy.max(EPSILON).ln());
        }
    }

    Array2::from_shape_vec((frame_count, MEL_BINS), output)
        .map_err(|error| anyhow!("Failed to construct SV filterbank: {}", error))
}

fn povey_window() -> Vec<f32> {
    (0..FRAME_LENGTH)
        .map(|i| {
            let phase = 2.0 * PI * i as f32 / (FRAME_LENGTH - 1) as f32;
            (0.5 - 0.5 * phase.cos()).powf(0.85)
        })
        .collect()
}

fn mel_filterbank() -> Vec<Vec<f32>> {
    let low_mel = mel_scale(LOW_FREQ);
    let high_mel = mel_scale(HIGH_FREQ);
    let mel_delta = (high_mel - low_mel) / (MEL_BINS + 1) as f32;

    (0..MEL_BINS)
        .map(|bin_idx| {
            let left = low_mel + bin_idx as f32 * mel_delta;
            let center = low_mel + (bin_idx + 1) as f32 * mel_delta;
            let right = low_mel + (bin_idx + 2) as f32 * mel_delta;

            (0..=FFT_SIZE / 2)
                .map(|fft_idx| {
                    // kaldi.py builds filters for bins [0, 255] and pads the
                    // Nyquist column (bin 256) with zero.
                    if fft_idx == FFT_SIZE / 2 {
                        return 0.0;
                    }
                    let frequency = SAMPLE_RATE * fft_idx as f32 / FFT_SIZE as f32;
                    let mel = mel_scale(frequency);
                    ((mel - left) / (center - left))
                        .min((right - mel) / (right - center))
                        .max(0.0)
                })
                .collect()
        })
        .collect()
}

fn mel_scale(frequency: f32) -> f32 {
    1127.0 * (1.0 + frequency / 700.0).ln()
}

#[cfg(test)]
mod tests {
    use super::kaldi_fbank;

    #[test]
    fn produces_expected_frame_and_feature_dimensions() {
        let features = kaldi_fbank(&vec![0.0; 16_000]).unwrap();
        assert_eq!(features.shape(), &[98, 80]);
        assert!(features.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_audio_shorter_than_one_frame() {
        assert!(kaldi_fbank(&vec![0.0; 399]).is_err());
    }

    #[test]
    fn matches_upstream_kaldi_fbank_for_a_deterministic_frame() {
        let audio: Vec<f32> = (0..16_000)
            .map(|i| -0.8 + 1.6 * i as f32 / 15_999.0)
            .collect();
        let features = kaldi_fbank(&audio).unwrap();
        let expected = [
            -7.3628030,
            -8.1426048,
            -12.9475498,
            -13.9233694,
            -14.3024788,
            -14.3260603,
            -15.2077885,
            -15.9423847,
        ];

        for (actual, expected) in features.row(0).iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
        }
    }
}
