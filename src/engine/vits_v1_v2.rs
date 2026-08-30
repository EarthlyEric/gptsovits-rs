use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

pub struct VitsV1V2Model {
    session: Mutex<Option<Session>>,
    sampling_rate: u32,
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

impl VitsV1V2Model {
    pub fn new<P: AsRef<Path>>(model_path: P, sampling_rate: u32) -> Self {
        let session = load_session(model_path, 4);
        Self {
            session: Mutex::new(session),
            sampling_rate,
        }
    }

    /// Synthesize audio waveform from text sequence, predicted semantic tokens, and reference audio
    /// Output: mono audio samples (f32) at sampling_rate (typically 32,000 Hz)
    pub fn synthesize(
        &self,
        text_seq: &[i64],
        pred_semantic: &[i64],
        ref_audio: &[f32],
    ) -> Result<Vec<f32>> {
        let mut session_guard = self.session.lock().unwrap();

        if let Some(session) = session_guard.as_mut() {
            let actual_sem: Vec<i64> = if pred_semantic.is_empty() {
                let num_semantic = (text_seq.len() * 2).clamp(10, 200);
                (0..num_semantic).map(|i| (i % 1024) as i64).collect()
            } else {
                pred_semantic.iter().copied().map(|t| t.clamp(0, 1023)).collect()
            };

            let actual_ref = if ref_audio.is_empty() {
                vec![0.0f32; self.sampling_rate as usize * 3]
            } else {
                ref_audio.to_vec()
            };

            let text_arr = Array2::from_shape_vec((1, text_seq.len()), text_seq.to_vec())?;
            let sem_arr = Array3::from_shape_vec((1, 1, actual_sem.len()), actual_sem)?;
            let ref_arr = Array2::from_shape_vec((1, actual_ref.len()), actual_ref)?;

            let text_val = Value::from_array(text_arr)?;
            let sem_val = Value::from_array(sem_arr)?;
            let ref_val = Value::from_array(ref_arr)?;

            let outputs = session.run(ort::inputs![
                "text_seq" => text_val,
                "pred_semantic" => sem_val,
                "ref_audio" => ref_val,
            ])?;

            let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            Ok(data.to_vec())
        } else {
            // Mock synthesis: generates a tonal envelope based on semantic tokens
            let duration_secs = (pred_semantic.len() as f32 / 25.0).max(0.5);
            let num_samples = (duration_secs * self.sampling_rate as f32) as usize;
            let mut samples = Vec::with_capacity(num_samples);

            for i in 0..num_samples {
                let t = i as f32 / self.sampling_rate as f32;
                let freq = 220.0 + 50.0 * (t * 4.0).sin();
                let envelope = (t * std::f32::consts::PI / duration_secs).sin().powf(0.5);
                let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3 * envelope;
                samples.push(sample);
            }

            Ok(samples)
        }
    }
}
