use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

pub const SPEAKER_EMBEDDING_DIM: usize = 20_480;
const FBANK_DIM: usize = 80;

pub struct SpeakerEmbeddingModel {
    session: Mutex<Option<Session>>,
}

fn load_session<P: AsRef<Path>>(path: P, intra_threads: usize, inter_threads: usize) -> Option<Session> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        tracing::warn!(path = %path_ref.display(), "Speaker embedding ONNX model file not found");
        return None;
    }

    let builder = match Session::builder() {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to create speaker embedding ONNX session");
            return None;
        }
    };
    let builder = match builder.with_intra_threads(intra_threads) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to configure speaker embedding ONNX session");
            return None;
        }
    };
    let mut builder = match builder.with_inter_threads(inter_threads) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to configure speaker inter-op threads");
            return None;
        }
    };

    match builder.commit_from_file(path_ref) {
        Ok(session) => Some(session),
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to load speaker embedding ONNX model");
            None
        }
    }
}

impl SpeakerEmbeddingModel {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        intra_threads: usize,
        inter_threads: usize,
    ) -> Self {
        let session = load_session(model_path, intra_threads, inter_threads);
        Self {
            session: Mutex::new(session),
        }
    }

    /// Extract the `[1, 20480]` ERes2NetV2 embedding from 16 kHz mono audio.
    pub fn extract(&self, audio_16k: &[f32]) -> Result<Array2<f32>> {
        let fbank = crate::audio::kaldi_fbank(audio_16k)?;
        let (frames, features) = fbank.dim();
        if features != FBANK_DIM || frames == 0 {
            return Err(anyhow!(
                "SV filterbank has shape [{}, {}]; expected [frames, {}]",
                frames,
                features,
                FBANK_DIM
            ));
        }

        let fbank_data: Vec<f32> = fbank.iter().copied().collect();
        let fbank_arr = Array3::from_shape_vec((1, frames, FBANK_DIM), fbank_data)?;
        let fbank_val = Value::from_array(fbank_arr)?;

        let mut session_guard = self.session.lock().unwrap();
        let session = session_guard
            .as_mut()
            .ok_or_else(|| anyhow!("Speaker embedding model is not loaded"))?;
        let outputs = session.run(ort::inputs!["fbank" => fbank_val])?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_slice = shape.as_ref();
        if shape_slice.len() != 2
            || shape_slice[0] != 1
            || shape_slice[1] != SPEAKER_EMBEDDING_DIM as i64
        {
            return Err(anyhow!(
                "Speaker embedding model returned unexpected shape: {:?}",
                shape_slice
            ));
        }

        Array2::from_shape_vec((1, SPEAKER_EMBEDDING_DIM), data.to_vec())
            .map_err(|error| anyhow!("Failed to reshape speaker embedding: {}", error))
    }
}
