use anyhow::{anyhow, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

pub struct RoBERTaModel {
    session: Mutex<Option<Session>>,
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

impl RoBERTaModel {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Self {
        let session = load_session(model_path, 4);
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

        if let Some(session) = session_guard.as_mut() {
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

            if shape_slice.len() == 3 && shape_slice[1] >= 2 {
                let seq_dim = shape_slice[1] as usize;
                let hidden_dim = shape_slice[2] as usize;
                let num_chars = seq_dim - 2; // exclude [CLS] and [SEP]
                let mut char_embeddings = Vec::with_capacity(num_chars * hidden_dim);

                // Slice data[1..-1]
                for char_idx in 1..seq_dim - 1 {
                    let start = char_idx * hidden_dim;
                    let end = start + hidden_dim;
                    char_embeddings.extend_from_slice(&data[start..end]);
                }

                Array2::from_shape_vec((num_chars, hidden_dim), char_embeddings)
                    .map_err(|e| anyhow!("Failed to reshape BERT embeddings: {}", e))
            } else {
                let num_chars = seq_len.saturating_sub(2).max(1);
                Ok(Array2::zeros((num_chars, 1024)))
            }
        } else {
            let num_chars = input_ids.len().saturating_sub(2).max(1);
            Ok(Array2::zeros((num_chars, 1024)))
        }
    }
}
