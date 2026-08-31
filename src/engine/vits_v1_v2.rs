use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

pub struct VitsV1V2Model {
    session: Mutex<Option<Session>>,
    has_speaker_embedding_input: bool,
}

fn load_session<P: AsRef<Path>>(path: P, intra_threads: usize, inter_threads: usize) -> Option<Session> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        tracing::warn!(path = %path_ref.display(), "VITS ONNX model file not found");
        return None;
    }
    let builder = match Session::builder() {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to create VITS ONNX session");
            return None;
        }
    };
    let builder = match builder.with_intra_threads(intra_threads) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to configure VITS ONNX session");
            return None;
        }
    };
    let mut builder = match builder.with_inter_threads(inter_threads) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to configure VITS inter-op threads");
            return None;
        }
    };
    match builder.commit_from_file(path_ref) {
        Ok(session) => Some(session),
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to load VITS ONNX model");
            None
        }
    }
}

impl VitsV1V2Model {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        _sampling_rate: u32,
        intra_threads: usize,
        inter_threads: usize,
    ) -> Self {
        let session = load_session(model_path, intra_threads, inter_threads);
        let has_speaker_embedding_input = session
            .as_ref()
            .map(|session| {
                session
                    .inputs()
                    .iter()
                    .any(|input| input.name() == "sv_emb")
            })
            .unwrap_or(false);
        Self {
            session: Mutex::new(session),
            has_speaker_embedding_input,
        }
    }

    pub fn has_speaker_embedding_input(&self) -> bool {
        self.has_speaker_embedding_input
    }

    /// Synthesize audio waveform from text sequence, predicted semantic tokens, and reference audio
    /// Output: mono audio samples (f32) at sampling_rate (typically 32,000 Hz)
    pub fn synthesize(
        &self,
        text_seq: &[i64],
        pred_semantic: &[i64],
        ref_audio: &[f32],
        sv_emb: Option<&Array2<f32>>,
    ) -> Result<Vec<f32>> {
        if pred_semantic.is_empty() {
            return Err(anyhow!("T2S produced no semantic tokens"));
        }
        if ref_audio.is_empty() {
            return Err(anyhow!("Reference audio is required for VITS synthesis"));
        }

        let mut session_guard = self.session.lock().unwrap();

        if let Some(session) = session_guard.as_mut() {
            let actual_sem: Vec<i64> = pred_semantic
                .iter()
                .copied()
                .map(|t| t.clamp(0, 1023))
                .collect();
            let actual_ref = ref_audio.to_vec();

            let text_arr = Array2::from_shape_vec((1, text_seq.len()), text_seq.to_vec())?;
            let sem_arr = Array3::from_shape_vec((1, 1, actual_sem.len()), actual_sem)?;
            let ref_arr = Array2::from_shape_vec((1, actual_ref.len()), actual_ref)?;

            let text_val = Value::from_array(text_arr)?;
            let sem_val = Value::from_array(sem_arr)?;
            let ref_val = Value::from_array(ref_arr)?;

            let outputs = if self.has_speaker_embedding_input {
                let sv_emb = sv_emb
                    .ok_or_else(|| anyhow!("V2Pro/V2ProPlus VITS requires a speaker embedding"))?;
                if sv_emb.shape() != [1, 20_480] {
                    return Err(anyhow!(
                        "Speaker embedding has shape {:?}; expected [1, 20480]",
                        sv_emb.shape()
                    ));
                }
                let sv_val = Value::from_array(sv_emb.clone())?;
                session.run(ort::inputs![
                    "text_seq" => text_val,
                    "pred_semantic" => sem_val,
                    "ref_audio" => ref_val,
                    "sv_emb" => sv_val,
                ])?
            } else {
                session.run(ort::inputs![
                    "text_seq" => text_val,
                    "pred_semantic" => sem_val,
                    "ref_audio" => ref_val,
                ])?
            };

            let (_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            Ok(data.to_vec())
        } else {
            Err(anyhow!("VITS model is not loaded"))
        }
    }
}
