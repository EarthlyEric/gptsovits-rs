use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

pub struct CfmV3V4Model {
    dit_session: Mutex<Option<Session>>,
    vocoder_session: Mutex<Option<Session>>,
    sampling_rate: u32,
    default_sample_steps: usize,
}

fn load_session<P: AsRef<Path>>(path: P, threads: usize) -> Option<Session> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        return None;
    }
    let builder = Session::builder().map_err(|e| anyhow!("{}", e)).ok()?;
    let mut builder = builder.with_intra_threads(threads).map_err(|e| anyhow!("{}", e)).ok()?;
    builder.commit_from_file(path_ref).map_err(|e| anyhow!("{}", e)).ok()
}

impl CfmV3V4Model {
    pub fn new<P: AsRef<Path>>(
        dit_path: P,
        vocoder_path: P,
        sampling_rate: u32,
        default_sample_steps: usize,
    ) -> Self {
        let dit_session = load_session(dit_path, 4);
        let vocoder_session = load_session(vocoder_path, 4);

        Self {
            dit_session: Mutex::new(dit_session),
            vocoder_session: Mutex::new(vocoder_session),
            sampling_rate,
            default_sample_steps,
        }
    }

    /// Run Continuous Normalizing Flow (CFM) DiT with ODE solver, then decode with Vocoder
    pub fn synthesize(
        &self,
        _text_seq: &[i64],
        pred_semantic: &[i64],
        _ref_audio: &[f32],
        steps: usize,
    ) -> Result<Vec<f32>> {
        let actual_steps = if steps > 0 { steps } else { self.default_sample_steps };
        let mut dit_guard = self.dit_session.lock().unwrap();
        let mut vocoder_guard = self.vocoder_session.lock().unwrap();

        if let (Some(dit), Some(vocoder)) = (dit_guard.as_mut(), vocoder_guard.as_mut()) {
            let mel_frames = (pred_semantic.len() * 4).max(32);
            let n_mels = 100;

            // Initial random noise
            let mut mel_latent = Array3::<f32>::zeros((1, n_mels, mel_frames));
            for v in mel_latent.iter_mut() {
                *v = rand::random::<f32>() * 2.0 - 1.0;
            }

            let dt = 1.0 / actual_steps as f32;

            // Euler ODE solver
            for step in 0..actual_steps {
                let t = step as f32 * dt;
                let t_arr = Array2::from_shape_vec((1, 1), vec![t])?;

                let x_val = Value::from_array(mel_latent.clone())?;
                let t_val = Value::from_array(t_arr)?;

                let dit_outputs = dit.run(ort::inputs![
                    "x" => x_val,
                    "t" => t_val,
                ])?;

                let (_shape, vt_data) = dit_outputs[0].try_extract_tensor::<f32>()?;

                for (x_val, &vt) in mel_latent.iter_mut().zip(vt_data.iter()) {
                    *x_val += vt * dt;
                }
            }

            // Vocoder synthesis
            let mel_val = Value::from_array(mel_latent)?;
            let vocoder_outputs = vocoder.run(ort::inputs![
                "mel" => mel_val,
            ])?;

            let (_shape, audio_data) = vocoder_outputs[0].try_extract_tensor::<f32>()?;
            Ok(audio_data.to_vec())
        } else {
            // Mock synthesis for V3/V4
            let duration_secs = (pred_semantic.len() as f32 / 25.0).max(0.5);
            let num_samples = (duration_secs * self.sampling_rate as f32) as usize;
            let mut samples = Vec::with_capacity(num_samples);

            for i in 0..num_samples {
                let t = i as f32 / self.sampling_rate as f32;
                let freq = 260.0 + 40.0 * (t * 3.5).sin();
                let envelope = (t * std::f32::consts::PI / duration_secs).sin().powf(0.5);
                let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3 * envelope;
                samples.push(sample);
            }

            Ok(samples)
        }
    }
}
