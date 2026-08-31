use anyhow::{anyhow, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

pub struct RoBERTaModel {
    session: Mutex<Option<Session>>,
}

fn load_session<P: AsRef<Path>>(path: P, intra_threads: usize, inter_threads: usize) -> Option<Session> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        tracing::warn!(path = %path_ref.display(), "RoBERTa ONNX model file not found");
        return None;
    }
    let builder = match Session::builder() {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to create RoBERTa ONNX session");
            return None;
        }
    };
    let builder = match builder.with_intra_threads(intra_threads) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to configure RoBERTa ONNX session");
            return None;
        }
    };
    let mut builder = match builder.with_inter_threads(inter_threads) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to configure RoBERTa inter-op threads");
            return None;
        }
    };
    match builder.commit_from_file(path_ref) {
        Ok(session) => Some(session),
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to load RoBERTa ONNX model");
            None
        }
    }
}

impl RoBERTaModel {
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

    /// Extract 1024-dimensional character-level BERT embeddings
    /// Output: Array2<f32> [num_chars, 1024]
    pub fn extract(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
    ) -> Result<Array2<f32>> {
        let mut session_guard = self.session.lock().unwrap();

        let session = session_guard
            .as_mut()
            .ok_or_else(|| anyhow!("RoBERTa model is not loaded"))?;
        let seq_len = input_ids.len();
        let ids_arr = Array2::from_shape_vec((1, seq_len), input_ids.to_vec())?;
        let mask_arr = Array2::from_shape_vec((1, seq_len), attention_mask.to_vec())?;
        let type_arr = Array2::from_shape_vec((1, seq_len), token_type_ids.to_vec())?;

        let ids_val = Value::from_array(ids_arr)?;
        let mask_val = Value::from_array(mask_arr)?;
        let type_val = Value::from_array(type_arr)?;

        let outputs = session.run(ort::inputs![
            "input_ids" => ids_val,
            "attention_mask" => mask_val,
            "token_type_ids" => type_val,
        ])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_slice = shape.as_ref();
        if shape_slice.len() != 3 || shape_slice[0] != 1 || shape_slice[1] < 2 {
            return Err(anyhow!(
                "RoBERTa returned an unexpected tensor shape: {:?}",
                shape_slice
            ));
        }

        let seq_dim = shape_slice[1] as usize;
        let hidden_dim = shape_slice[2] as usize;
        let num_chars = seq_dim - 2; // exclude [CLS] and [SEP]
        let mut char_embeddings = Vec::with_capacity(num_chars * hidden_dim);

        // Slice data[1..-1] to remove the tokenizer's special tokens.
        for char_idx in 1..seq_dim - 1 {
            let start = char_idx * hidden_dim;
            let end = start + hidden_dim;
            char_embeddings.extend_from_slice(&data[start..end]);
        }

        Array2::from_shape_vec((num_chars, hidden_dim), char_embeddings)
            .map_err(|e| anyhow!("Failed to reshape BERT embeddings: {}", e))
    }
}
