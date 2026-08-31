use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

pub struct CNHuBERTModel {
    session: Mutex<Option<Session>>,
}

fn load_session<P: AsRef<Path>>(path: P, threads: usize) -> Option<Session> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        tracing::warn!(path = %path_ref.display(), "CNHuBERT ONNX model file not found");
        return None;
    }
    let builder = match Session::builder() {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to create CNHuBERT ONNX session");
            return None;
        }
    };
    let mut builder = match builder.with_intra_threads(threads) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to configure CNHuBERT ONNX session");
            return None;
        }
    };
    match builder.commit_from_file(path_ref) {
        Ok(session) => Some(session),
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to load CNHuBERT ONNX model");
            None
        }
    }
}

impl CNHuBERTModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        let session = load_session(model_path, 4);
        Self {
            session: Mutex::new(session),
        }
    }

    /// Extract 768-dimensional SSL features from 16kHz audio samples
    /// Input: 16kHz mono audio samples [num_samples]
    /// Output: SSL features Array3<f32> [1, 768, ssl_len] where ssl_len ~ num_samples / 320
    pub fn extract(&self, audio_16k: &[f32]) -> Result<Array3<f32>> {
        let mut session_guard = self.session.lock().unwrap();

        let session = session_guard
            .as_mut()
            .ok_or_else(|| anyhow!("CNHuBERT model is not loaded"))?;
        let input_len = audio_16k.len();
        let audio_arr = Array2::from_shape_vec((1, input_len), audio_16k.to_vec())?;
        let audio_val = Value::from_array(audio_arr)?;

        let outputs = session.run(ort::inputs!["ref_audio_16k" => audio_val])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let shape_slice = shape.as_ref();
        if shape_slice.len() != 3 {
            return Err(anyhow!(
                "CNHuBERT returned an unexpected tensor rank: {}",
                shape_slice.len()
            ));
        }

        let d0 = shape_slice[0] as usize;
        let d1 = shape_slice[1] as usize;
        let d2 = shape_slice[2] as usize;
        Array3::from_shape_vec((d0, d1, d2), data.to_vec())
            .map_err(|e| anyhow!("Failed to reshape SSL output: {}", e))
    }
}
