use anyhow::{anyhow, Result};
use std::collections::HashMap;

use crate::audio::{adjust_speed, resample};
use crate::config::AppConfig;
use crate::engine::cfm_v3_v4::CfmV3V4Model;
use crate::engine::cnhubert::CNHuBERTModel;
use crate::engine::roberta::RoBERTaModel;
use crate::engine::t2s::T2SModel;
use crate::engine::types::{InferenceRequest, InferenceResult, ModelVersion};
use crate::engine::vits_v1_v2::VitsV1V2Model;
use crate::text::{
    align_bert_to_phones, cleaned_text_to_sequence, text_to_phonemes, BertTokenizer,
};

pub struct CustomModelInstance {
    pub version: ModelVersion,
    pub t2s: T2SModel,
    pub vits: Option<VitsV1V2Model>,
    pub cfm: Option<CfmV3V4Model>,
    pub sampling_rate: u32,
    pub sample_steps: usize,
}

pub struct ModelManager {
    cnhubert: CNHuBERTModel,
    roberta: RoBERTaModel,
    tokenizer: Option<BertTokenizer>,
    t2s_models: HashMap<ModelVersion, T2SModel>,
    vits_models: HashMap<ModelVersion, VitsV1V2Model>,
    cfm_models: HashMap<ModelVersion, CfmV3V4Model>,
    custom_models: HashMap<String, CustomModelInstance>,
    default_version: ModelVersion,
}

impl ModelManager {
    pub fn new(config: &AppConfig) -> Self {
        let default_version = ModelVersion::from_str_loose(&config.models.default_version);

        let cnhubert = CNHuBERTModel::new(&config.models.cnhubert_path);
        let roberta = RoBERTaModel::new(&config.models.bert_path);
        let tokenizer = BertTokenizer::from_file(&config.models.bert_tokenizer_path).ok();

        let mut t2s_models = HashMap::new();
        let mut vits_models = HashMap::new();
        let mut cfm_models = HashMap::new();
        let mut custom_models = HashMap::new();

        // 1. Base Models (V1)
        t2s_models.insert(
            ModelVersion::V1,
            T2SModel::new(
                &config.models.v1.t2s_encoder_path,
                &config.models.v1.t2s_fsdec_path,
                &config.models.v1.t2s_sdec_path,
            ),
        );
        vits_models.insert(
            ModelVersion::V1,
            VitsV1V2Model::new(&config.models.v1.vits_path, config.models.v1.sampling_rate),
        );

        // 2. Base Models (V2)
        t2s_models.insert(
            ModelVersion::V2,
            T2SModel::new(
                &config.models.v2.t2s_encoder_path,
                &config.models.v2.t2s_fsdec_path,
                &config.models.v2.t2s_sdec_path,
            ),
        );
        vits_models.insert(
            ModelVersion::V2,
            VitsV1V2Model::new(&config.models.v2.vits_path, config.models.v2.sampling_rate),
        );

        // 3. Base Models (V2Pro)
        t2s_models.insert(
            ModelVersion::V2Pro,
            T2SModel::new(
                &config.models.v2_pro.t2s_encoder_path,
                &config.models.v2_pro.t2s_fsdec_path,
                &config.models.v2_pro.t2s_sdec_path,
            ),
        );
        vits_models.insert(
            ModelVersion::V2Pro,
            VitsV1V2Model::new(&config.models.v2_pro.vits_path, config.models.v2_pro.sampling_rate),
        );

        // 4. Base Models (V2ProPlus)
        t2s_models.insert(
            ModelVersion::V2ProPlus,
            T2SModel::new(
                &config.models.v2_pro_plus.t2s_encoder_path,
                &config.models.v2_pro_plus.t2s_fsdec_path,
                &config.models.v2_pro_plus.t2s_sdec_path,
            ),
        );
        vits_models.insert(
            ModelVersion::V2ProPlus,
            VitsV1V2Model::new(&config.models.v2_pro_plus.vits_path, config.models.v2_pro_plus.sampling_rate),
        );

        // 5. Base Models (V3)
        t2s_models.insert(
            ModelVersion::V3,
            T2SModel::new(
                &config.models.v3.t2s_encoder_path,
                &config.models.v3.t2s_fsdec_path,
                &config.models.v3.t2s_sdec_path,
            ),
        );
        cfm_models.insert(
            ModelVersion::V3,
            CfmV3V4Model::new(
                &config.models.v3.dit_path,
                &config.models.v3.vocoder_path,
                config.models.v3.sampling_rate,
                config.models.v3.sample_steps,
            ),
        );

        // 6. Base Models (V4)
        t2s_models.insert(
            ModelVersion::V4,
            T2SModel::new(
                &config.models.v4.t2s_encoder_path,
                &config.models.v4.t2s_fsdec_path,
                &config.models.v4.t2s_sdec_path,
            ),
        );
        cfm_models.insert(
            ModelVersion::V4,
            CfmV3V4Model::new(
                &config.models.v4.dit_path,
                &config.models.v4.vocoder_path,
                config.models.v4.sampling_rate,
                config.models.v4.sample_steps,
            ),
        );

        // 7. Custom Fine-tuned Models
        for (name, custom_cfg) in &config.models.custom {
            let ver = ModelVersion::from_str_loose(&custom_cfg.model_version);
            let sr = custom_cfg.sampling_rate.unwrap_or_else(|| ver.sampling_rate());
            let steps = custom_cfg.sample_steps.unwrap_or_else(|| ver.default_sample_steps());

            let dir = &custom_cfg.model_dir;
            let enc_path = if !custom_cfg.t2s_encoder_path.is_empty() {
                custom_cfg.t2s_encoder_path.clone()
            } else if !dir.is_empty() {
                format!("{}/t2s_encoder.onnx", dir)
            } else {
                String::new()
            };

            let fsdec_path = if !custom_cfg.t2s_fsdec_path.is_empty() {
                custom_cfg.t2s_fsdec_path.clone()
            } else if !dir.is_empty() {
                format!("{}/t2s_fsdec.onnx", dir)
            } else {
                String::new()
            };

            let sdec_path = if !custom_cfg.t2s_sdec_path.is_empty() {
                custom_cfg.t2s_sdec_path.clone()
            } else if !dir.is_empty() {
                format!("{}/t2s_sdec.onnx", dir)
            } else {
                String::new()
            };

            let vits_path = if !custom_cfg.vits_path.is_empty() {
                custom_cfg.vits_path.clone()
            } else if !dir.is_empty() {
                format!("{}/vits.onnx", dir)
            } else {
                String::new()
            };

            let dit_path = if !custom_cfg.dit_path.is_empty() {
                custom_cfg.dit_path.clone()
            } else if !dir.is_empty() {
                format!("{}/dit.onnx", dir)
            } else {
                String::new()
            };

            let vocoder_path = if !custom_cfg.vocoder_path.is_empty() {
                custom_cfg.vocoder_path.clone()
            } else if !dir.is_empty() {
                format!("{}/vocoder.onnx", dir)
            } else {
                String::new()
            };

            let t2s = T2SModel::new(&enc_path, &fsdec_path, &sdec_path);

            let (vits, cfm) = match ver {
                ModelVersion::V1 | ModelVersion::V2 | ModelVersion::V2Pro | ModelVersion::V2ProPlus => {
                    (Some(VitsV1V2Model::new(&vits_path, sr)), None)
                }
                ModelVersion::V3 | ModelVersion::V4 => {
                    (None, Some(CfmV3V4Model::new(&dit_path, &vocoder_path, sr, steps)))
                }
            };

            custom_models.insert(
                name.to_lowercase(),
                CustomModelInstance {
                    version: ver,
                    t2s,
                    vits,
                    cfm,
                    sampling_rate: sr,
                    sample_steps: steps,
                },
            );
        }

        Self {
            cnhubert,
            roberta,
            tokenizer,
            t2s_models,
            vits_models,
            cfm_models,
            custom_models,
            default_version,
        }
    }

    pub fn default_version(&self) -> ModelVersion {
        self.default_version
    }

    pub fn has_custom_model(&self, name: &str) -> bool {
        self.custom_models.contains_key(&name.to_lowercase())
    }

    pub fn list_custom_models(&self) -> Vec<(String, ModelVersion)> {
        self.custom_models
            .iter()
            .map(|(k, v)| (k.clone(), v.version))
            .collect()
    }

    /// Full end-to-end TTS Synthesis pipeline by Model Identifier (custom model name or base model)
    pub fn synthesize_by_model(
        &self,
        req: &InferenceRequest,
        model_name: &str,
    ) -> Result<InferenceResult> {
        let name_lower = model_name.to_lowercase();

        // 1. Check if it's a registered custom fine-tuned model
        if let Some(custom_model) = self.custom_models.get(&name_lower) {
            return self.synthesize_with_custom(req, custom_model);
        }

        // 2. Otherwise resolve base model version
        let base_version = if name_lower.contains("v1") {
            ModelVersion::V1
        } else if name_lower.contains("v2proplus") || name_lower.contains("v2pro+") {
            ModelVersion::V2ProPlus
        } else if name_lower.contains("v2pro") {
            ModelVersion::V2Pro
        } else if name_lower.contains("v3") {
            ModelVersion::V3
        } else if name_lower.contains("v4") {
            ModelVersion::V4
        } else if name_lower.contains("v2") {
            ModelVersion::V2
        } else if name_lower == "gpt-4o-mini-tts" || name_lower == "tts-1" || name_lower == "tts-1-hd" || name_lower == "gpt-sovits" {
            self.default_version
        } else {
            return Err(anyhow!("Model '{}' not found in base models or custom models", model_name));
        };

        self.synthesize(req, base_version)
    }

    /// Synthesize with a custom fine-tuned model instance
    fn synthesize_with_custom(
        &self,
        req: &InferenceRequest,
        custom: &CustomModelInstance,
    ) -> Result<InferenceResult> {
        let sym_version = custom.version.symbols_version();
        let target_sr = custom.sampling_rate;

        // 1. Target Text G2P & BERT
        let (text_phones, text_word2ph, text_norm) = text_to_phonemes(&req.text, &req.text_lang, sym_version);
        let text_seq = cleaned_text_to_sequence(&text_phones, sym_version);

        let text_bert = if let Some(tok) = &self.tokenizer {
            if let Ok((ids, mask, types)) = tok.encode(&text_norm) {
                if let Ok(char_bert) = self.roberta.extract(&ids, &mask, &types) {
                    align_bert_to_phones(Some(&char_bert), &text_word2ph, text_phones.len())
                } else {
                    align_bert_to_phones(None, &text_word2ph, text_phones.len())
                }
            } else {
                align_bert_to_phones(None, &text_word2ph, text_phones.len())
            }
        } else {
            align_bert_to_phones(None, &text_word2ph, text_phones.len())
        };

        // 2. Prompt Text G2P & BERT
        let prompt_text = if req.prompt_text.trim().is_empty() {
            "你好"
        } else {
            req.prompt_text.as_str()
        };

        let (ref_phones, ref_word2ph, ref_norm) = text_to_phonemes(prompt_text, &req.prompt_lang, sym_version);
        let ref_seq = cleaned_text_to_sequence(&ref_phones, sym_version);

        let ref_bert = if let Some(tok) = &self.tokenizer {
            if let Ok((ids, mask, types)) = tok.encode(&ref_norm) {
                if let Ok(char_bert) = self.roberta.extract(&ids, &mask, &types) {
                    align_bert_to_phones(Some(&char_bert), &ref_word2ph, ref_phones.len())
                } else {
                    align_bert_to_phones(None, &ref_word2ph, ref_phones.len())
                }
            } else {
                align_bert_to_phones(None, &ref_word2ph, ref_phones.len())
            }
        } else {
            align_bert_to_phones(None, &ref_word2ph, ref_phones.len())
        };

        // 3. Audio Preprocessing
        let ref_audio = if req.ref_audio.is_empty() {
            vec![0.0f32; (req.ref_sr as f32 * 1.5) as usize]
        } else {
            req.ref_audio.clone()
        };

        let ref_audio_16k = resample(&ref_audio, req.ref_sr as usize, 16000)?;
        let ref_audio_tgt_sr = resample(&ref_audio, req.ref_sr as usize, target_sr as usize)?;

        // 4. SSL Feature Extraction
        let ssl_content = self.cnhubert.extract(&ref_audio_16k)?;

        // 5. T2S Autoregressive Generation using Custom T2S Model
        let pred_semantic = custom.t2s.generate(
            &ref_seq,
            &text_seq,
            &ref_bert,
            &text_bert,
            &ssl_content,
            req.top_k,
            req.top_p,
            req.temperature,
            req.repetition_penalty,
        )?;

        // 6. Synthesis using Custom Synthesizer
        let raw_samples = if let Some(vits) = &custom.vits {
            vits.synthesize(&text_seq, &pred_semantic, &ref_audio_tgt_sr)?
        } else if let Some(cfm) = &custom.cfm {
            cfm.synthesize(&text_seq, &pred_semantic, &ref_audio_tgt_sr, req.sample_steps)?
        } else {
            return Err(anyhow!("Custom model has neither VITS nor CFM synthesizer"));
        };

        // 7. Speed Adjustment
        let processed_samples = if (req.speed - 1.0).abs() > 1e-4 {
            adjust_speed(&raw_samples, req.speed, target_sr)?
        } else {
            raw_samples
        };

        Ok(InferenceResult {
            samples: processed_samples,
            sample_rate: target_sr,
        })
    }

    /// Base Model TTS Synthesis pipeline in pure Rust
    pub fn synthesize(
        &self,
        req: &InferenceRequest,
        version: ModelVersion,
    ) -> Result<InferenceResult> {
        let sym_version = version.symbols_version();
        let target_sr = version.sampling_rate();

        // 1. Target Text G2P & BERT
        let (text_phones, text_word2ph, text_norm) = text_to_phonemes(&req.text, &req.text_lang, sym_version);
        let text_seq = cleaned_text_to_sequence(&text_phones, sym_version);

        let text_bert = if let Some(tok) = &self.tokenizer {
            if let Ok((ids, mask, types)) = tok.encode(&text_norm) {
                if let Ok(char_bert) = self.roberta.extract(&ids, &mask, &types) {
                    align_bert_to_phones(Some(&char_bert), &text_word2ph, text_phones.len())
                } else {
                    align_bert_to_phones(None, &text_word2ph, text_phones.len())
                }
            } else {
                align_bert_to_phones(None, &text_word2ph, text_phones.len())
            }
        } else {
            align_bert_to_phones(None, &text_word2ph, text_phones.len())
        };

        // 2. Reference / Prompt Text G2P & BERT
        let prompt_text = if req.prompt_text.trim().is_empty() {
            "你好"
        } else {
            req.prompt_text.as_str()
        };

        let (ref_phones, ref_word2ph, ref_norm) = text_to_phonemes(prompt_text, &req.prompt_lang, sym_version);
        let ref_seq = cleaned_text_to_sequence(&ref_phones, sym_version);

        let ref_bert = if let Some(tok) = &self.tokenizer {
            if let Ok((ids, mask, types)) = tok.encode(&ref_norm) {
                if let Ok(char_bert) = self.roberta.extract(&ids, &mask, &types) {
                    align_bert_to_phones(Some(&char_bert), &ref_word2ph, ref_phones.len())
                } else {
                    align_bert_to_phones(None, &ref_word2ph, ref_phones.len())
                }
            } else {
                align_bert_to_phones(None, &ref_word2ph, ref_phones.len())
            }
        } else {
            align_bert_to_phones(None, &ref_word2ph, ref_phones.len())
        };

        // 3. Reference Audio Preprocessing
        let ref_audio = if req.ref_audio.is_empty() {
            vec![0.0f32; (req.ref_sr as f32 * 1.5) as usize]
        } else {
            req.ref_audio.clone()
        };

        let ref_audio_16k = resample(&ref_audio, req.ref_sr as usize, 16000)?;
        let ref_audio_tgt_sr = resample(&ref_audio, req.ref_sr as usize, target_sr as usize)?;

        // 4. SSL Feature Extraction
        let ssl_content = self.cnhubert.extract(&ref_audio_16k)?;

        // 5. T2S Autoregressive Generation
        let t2s = self
            .t2s_models
            .get(&version)
            .ok_or_else(|| anyhow!("T2S model for {:?} not configured", version))?;

        let pred_semantic = t2s.generate(
            &ref_seq,
            &text_seq,
            &ref_bert,
            &text_bert,
            &ssl_content,
            req.top_k,
            req.top_p,
            req.temperature,
            req.repetition_penalty,
        )?;

        // 6. Acoustic Synthesizer (VITS for v1/v2/v2Pro/v2ProPlus, CFM for v3/v4)
        let raw_samples = match version {
            ModelVersion::V1 | ModelVersion::V2 | ModelVersion::V2Pro | ModelVersion::V2ProPlus => {
                let vits = self
                    .vits_models
                    .get(&version)
                    .ok_or_else(|| anyhow!("VITS model for {:?} not configured", version))?;
                vits.synthesize(&text_seq, &pred_semantic, &ref_audio_tgt_sr)?
            }
            ModelVersion::V3 | ModelVersion::V4 => {
                let cfm = self
                    .cfm_models
                    .get(&version)
                    .ok_or_else(|| anyhow!("CFM model for {:?} not configured", version))?;
                cfm.synthesize(
                    &text_seq,
                    &pred_semantic,
                    &ref_audio_tgt_sr,
                    req.sample_steps,
                )?
            }
        };

        // 7. Post-processing: Speed Adjustment
        let processed_samples = if (req.speed - 1.0).abs() > 1e-4 {
            adjust_speed(&raw_samples, req.speed, target_sr)?
        } else {
            raw_samples
        };

        Ok(InferenceResult {
            samples: processed_samples,
            sample_rate: target_sr,
        })
    }
}
