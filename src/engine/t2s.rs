use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3, Array4};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

use crate::engine::sampler::sample_next_token;

pub struct T2SModel {
    encoder_session: Mutex<Option<Session>>,
    fsdec_session: Mutex<Option<Session>>,
    sdec_session: Mutex<Option<Session>>,
}

fn load_session<P: AsRef<Path>>(path: P, threads: usize) -> Option<Session> {
    let path_ref = path.as_ref();
    if !path_ref.exists() {
        tracing::warn!(path = %path_ref.display(), "T2S ONNX model file not found");
        return None;
    }
    let builder = match Session::builder() {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to create T2S ONNX session");
            return None;
        }
    };
    let mut builder = match builder.with_intra_threads(threads) {
        Ok(builder) => builder,
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to configure T2S ONNX session");
            return None;
        }
    };
    match builder.commit_from_file(path_ref) {
        Ok(session) => Some(session),
        Err(error) => {
            tracing::warn!(path = %path_ref.display(), %error, "Failed to load T2S ONNX model");
            None
        }
    }
}

impl T2SModel {
    pub fn new<P: AsRef<Path>>(encoder_path: P, fsdec_path: P, sdec_path: P) -> Self {
        let encoder_session = load_session(encoder_path, 4);
        let fsdec_session = load_session(fsdec_path, 4);
        let sdec_session = load_session(sdec_path, 4);

        Self {
            encoder_session: Mutex::new(encoder_session),
            fsdec_session: Mutex::new(fsdec_session),
            sdec_session: Mutex::new(sdec_session),
        }
    }

    /// Autoregressively generate semantic tokens
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        ref_seq: &[i64],
        text_seq: &[i64],
        ref_bert: &Array2<f32>,
        text_bert: &Array2<f32>,
        ssl_content: &Array3<f32>,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
    ) -> Result<Vec<i64>> {
        let mut enc_guard = self.encoder_session.lock().unwrap();
        let mut fsdec_guard = self.fsdec_session.lock().unwrap();
        let mut sdec_guard = self.sdec_session.lock().unwrap();

        if let (Some(encoder), Some(fsdec), Some(sdec)) = (
            enc_guard.as_mut(),
            fsdec_guard.as_mut(),
            sdec_guard.as_mut(),
        ) {
            // 1. T2S Encoder
            let ref_seq_arr = Array2::from_shape_vec((1, ref_seq.len()), ref_seq.to_vec())?;
            let text_seq_arr = Array2::from_shape_vec((1, text_seq.len()), text_seq.to_vec())?;

            let ref_seq_val = Value::from_array(ref_seq_arr)?;
            let text_seq_val = Value::from_array(text_seq_arr)?;
            let ref_bert_val = Value::from_array(ref_bert.clone())?;
            let text_bert_val = Value::from_array(text_bert.clone())?;
            let ssl_val = Value::from_array(ssl_content.clone())?;

            let enc_outputs = encoder.run(ort::inputs![
                "ref_seq" => ref_seq_val,
                "text_seq" => text_seq_val,
                "ref_bert" => ref_bert_val,
                "text_bert" => text_bert_val,
                "ssl_content" => ssl_val,
            ])?;

            let (shape_x, data_x) = enc_outputs[0].try_extract_tensor::<f32>()?;
            let (shape_p, data_p) = enc_outputs[1].try_extract_tensor::<i64>()?;

            let sx = shape_x.as_ref();
            let sp = shape_p.as_ref();

            let x_arr = Array3::from_shape_vec(
                (sx[0] as usize, sx[1] as usize, sx[2] as usize),
                data_x.to_vec(),
            )?;
            let p_arr = Array2::from_shape_vec((sp[0] as usize, sp[1] as usize), data_p.to_vec())?;

            let x_val = Value::from_array(x_arr)?;
            let p_val = Value::from_array(p_arr)?;

            // 2. T2S First-stage decoder
            let fsdec_outputs = fsdec.run(ort::inputs![
                "x" => x_val,
                "prompts" => p_val,
            ])?;

            let (s_y, d_y) = fsdec_outputs[0].try_extract_tensor::<i64>()?;
            let (s_k, d_k) = fsdec_outputs[1].try_extract_tensor::<f32>()?;
            let (s_v, d_v) = fsdec_outputs[2].try_extract_tensor::<f32>()?;
            let (s_emb, d_emb) = fsdec_outputs[3].try_extract_tensor::<f32>()?;
            let (s_ex, d_ex) = fsdec_outputs[4].try_extract_tensor::<f32>()?;

            let sy = s_y.as_ref();
            let sk = s_k.as_ref();
            let sv = s_v.as_ref();
            let semb = s_emb.as_ref();
            let sex = s_ex.as_ref();

            let mut y_arr = Array2::from_shape_vec((sy[0] as usize, sy[1] as usize), d_y.to_vec())?;
            let mut k_arr = Array4::from_shape_vec(
                (
                    sk[0] as usize,
                    sk[1] as usize,
                    sk[2] as usize,
                    sk[3] as usize,
                ),
                d_k.to_vec(),
            )?;
            let mut v_arr = Array4::from_shape_vec(
                (
                    sv[0] as usize,
                    sv[1] as usize,
                    sv[2] as usize,
                    sv[3] as usize,
                ),
                d_v.to_vec(),
            )?;
            let mut y_emb_arr = Array3::from_shape_vec(
                (semb[0] as usize, semb[1] as usize, semb[2] as usize),
                d_emb.to_vec(),
            )?;

            let sex_dims: Vec<usize> = sex.iter().map(|&dim| dim as usize).collect();
            let x_example_arr =
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&sex_dims), d_ex.to_vec())?;

            let mut generated_tokens = Vec::new();
            // The first-stage decoder has already appended one warm-up token. The
            // decoder cache does not contain the next sampled token yet, so each
            // externally sampled token can safely replace y's last element before
            // the following step.
            let mut history_tokens = d_y.to_vec();

            // 3. Autoregressive generation loop
            const MAX_STEPS: usize = 1500;
            const EOS_TOKEN: i64 = 1024;
            const MIN_GENERATED_TOKENS: usize = 10;

            for _ in 0..MAX_STEPS {
                let iy_val = Value::from_array(y_arr.clone())?;
                let ik_val = Value::from_array(k_arr.clone())?;
                let iv_val = Value::from_array(v_arr.clone())?;
                let iy_emb_val = Value::from_array(y_emb_arr.clone())?;
                let ix_val = Value::from_array(x_example_arr.clone())?;

                let step_outputs = sdec.run(ort::inputs![
                    "iy" => iy_val,
                    "ik" => ik_val,
                    "iv" => iv_val,
                    "iy_emb" => iy_emb_val,
                    "ix_example" => ix_val,
                ])?;

                let (s_y, d_y) = step_outputs[0].try_extract_tensor::<i64>()?;
                let (s_k, d_k) = step_outputs[1].try_extract_tensor::<f32>()?;
                let (s_v, d_v) = step_outputs[2].try_extract_tensor::<f32>()?;
                let (s_emb, d_emb) = step_outputs[3].try_extract_tensor::<f32>()?;
                let (_s_logits, d_logits) = step_outputs[4].try_extract_tensor::<f32>()?;

                let sy = s_y.as_ref();
                let sk = s_k.as_ref();
                let sv = s_v.as_ref();
                let semb = s_emb.as_ref();

                y_arr = Array2::from_shape_vec((sy[0] as usize, sy[1] as usize), d_y.to_vec())?;
                k_arr = Array4::from_shape_vec(
                    (
                        sk[0] as usize,
                        sk[1] as usize,
                        sk[2] as usize,
                        sk[3] as usize,
                    ),
                    d_k.to_vec(),
                )?;
                v_arr = Array4::from_shape_vec(
                    (
                        sv[0] as usize,
                        sv[1] as usize,
                        sv[2] as usize,
                        sv[3] as usize,
                    ),
                    d_v.to_vec(),
                )?;
                y_emb_arr = Array3::from_shape_vec(
                    (semb[0] as usize, semb[1] as usize, semb[2] as usize),
                    d_emb.to_vec(),
                )?;

                let current_logits = if d_logits.len() >= 1025 {
                    &d_logits[d_logits.len() - 1025..]
                } else {
                    return Err(anyhow!(
                        "T2S decoder returned {} logits; expected at least 1025",
                        d_logits.len()
                    ));
                };

                let mut sampling_logits = current_logits.to_vec();
                if generated_tokens.len() < MIN_GENERATED_TOKENS {
                    sampling_logits[EOS_TOKEN as usize] = f32::NEG_INFINITY;
                }

                let greedy_token = current_logits
                    .iter()
                    .enumerate()
                    .filter(|(_, score)| score.is_finite())
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(index, _)| index as i64);
                if generated_tokens.len() >= MIN_GENERATED_TOKENS && greedy_token == Some(EOS_TOKEN)
                {
                    break;
                }

                let next_token = sample_next_token(
                    &sampling_logits,
                    &history_tokens,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                );

                if next_token == EOS_TOKEN {
                    break;
                }

                let safe_token = next_token.clamp(0, 1023);
                generated_tokens.push(safe_token);
                history_tokens.push(safe_token);

                // sdec internally appends its own sample to y, but that sample
                // has not been consumed by k/v/y_emb yet. Replace it so the
                // next invocation follows the token sequence sent to VITS.
                let y_slice = y_arr
                    .as_slice_mut()
                    .ok_or_else(|| anyhow!("T2S decoder returned a non-contiguous y tensor"))?;
                let last = y_slice
                    .last_mut()
                    .ok_or_else(|| anyhow!("T2S decoder returned an empty y tensor"))?;
                *last = safe_token;
            }

            if generated_tokens.is_empty() {
                return Err(anyhow!("T2S decoder generated no semantic tokens"));
            }

            Ok(generated_tokens)
        } else {
            Err(anyhow!("T2S model is not fully loaded"))
        }
    }
}
