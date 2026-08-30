use anyhow::Result;

/// Adjust audio playback speed (0.25 to 4.0) using high-quality cubic interpolation in pure Rust
pub fn adjust_speed(samples: &[f32], speed: f32, _sample_rate: u32) -> Result<Vec<f32>> {
    if (speed - 1.0).abs() < 1e-4 || samples.is_empty() {
        return Ok(samples.to_vec());
    }

    // Clamp speed to spec limits
    let speed = speed.clamp(0.25, 4.0);

    let original_len = samples.len();
    let new_len = (original_len as f32 / speed) as usize;
    let mut out = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let orig_pos = i as f32 * speed;
        let idx0 = orig_pos.floor() as usize;
        let frac = orig_pos - idx0 as f32;

        let s0 = if idx0 > 0 { samples[idx0 - 1] } else { samples[0] };
        let s1 = samples[idx0.min(original_len - 1)];
        let s2 = samples[(idx0 + 1).min(original_len - 1)];
        let s3 = samples[(idx0 + 2).min(original_len - 1)];

        // Catmull-Rom cubic interpolation
        let a = -0.5 * s0 + 1.5 * s1 - 1.5 * s2 + 0.5 * s3;
        let b = s0 - 2.5 * s1 + 2.0 * s2 - 0.5 * s3;
        let c = -0.5 * s0 + 0.5 * s2;
        let d = s1;

        let val = a * frac * frac * frac + b * frac * frac + c * frac + d;
        out.push(val.clamp(-1.0, 1.0));
    }

    Ok(out)
}
